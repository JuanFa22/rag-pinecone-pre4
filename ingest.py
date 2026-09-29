"""Pipeline de ingesta: documentos -> chunks -> embeddings -> Pinecone Serverless.

Uso:
    python ingest.py                 # ingesta data/docs en el namespace configurado
    python ingest.py --reset         # borra el namespace antes de ingerir
"""
import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

import config

SUPPORTED = {".md", ".json", ".pdf"}


# ---------------------------------------------------------------- carga
def _parse_front_matter(raw: str) -> tuple[Dict[str, str], str]:
    """Front matter simple: '---\\ntags: a, b\\ntitulo: X\\n---'."""
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
    if not match:
        return {}, raw
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip().lower()] = v.strip()
    return meta, match.group(2)


def _tags(value: str) -> List[str]:
    return [t.strip().lower() for t in value.split(",") if t.strip()]


def load_documents(docs_dir: Path = config.DOCS_DIR) -> Iterator[Dict]:
    """Devuelve páginas/secciones: {doc_id, source, page, category, tags, text}."""
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED:
            continue
        rel = path.relative_to(docs_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "general"
        doc_id = path.stem

        if path.suffix.lower() == ".md":
            meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
            yield {"doc_id": doc_id, "source": rel.as_posix(), "page": 1,
                   "category": category, "tags": _tags(meta.get("tags", "")), "text": body.strip()}

        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else [data]
            for n, item in enumerate(items, start=1):
                yield {"doc_id": item.get("id", f"{doc_id}-{n}"), "source": rel.as_posix(), "page": n,
                       "category": item.get("category", category),
                       "tags": [str(t).lower() for t in item.get("tags", [])],
                       "text": item.get("text", "")}

        elif path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            for n, page in enumerate(PdfReader(str(path)).pages, start=1):
                yield {"doc_id": doc_id, "source": rel.as_posix(), "page": n,
                       "category": category, "tags": [], "text": (page.extract_text() or "").strip()}


# -------------------------------------------------------------- chunking
def build_chunks(pages: Iterator[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        if not page["text"]:
            continue
        for n, text in enumerate(splitter.split_text(page["text"])):
            chunks.append({
                "id": f'{page["doc_id"]}::p{page["page"]}::c{n}',
                "text": text,
                # Esquema de metadatos ESTRICTO (evita schema drift)
                "metadata": {
                    "doc_id": page["doc_id"],
                    "source": page["source"],
                    "page": page["page"],
                    "category": page["category"],
                    "tags": page["tags"],
                    "chunk": n,
                },
            })
    return chunks


# -------------------------------------------------------------- pinecone
def ensure_index(pc: Pinecone):
    """Crea el índice serverless si no existe y valida la dimensión."""
    if config.INDEX_NAME not in pc.list_indexes().names():
        print(f"Creando índice '{config.INDEX_NAME}' (dim={config.EMBED_DIM}, cosine)...")
        pc.create_index(
            name=config.INDEX_NAME, dimension=config.EMBED_DIM, metric="cosine",
            spec=ServerlessSpec(cloud=config.CLOUD, region=config.REGION),
        )
        while not pc.describe_index(config.INDEX_NAME).status["ready"]:
            time.sleep(1)
    info = pc.describe_index(config.INDEX_NAME)
    if info.dimension != config.EMBED_DIM:
        raise SystemExit(
            f"Mismatch de dimensiones: el índice tiene {info.dimension} y los embeddings "
            f"{config.EMBED_DIM}. Usá otro nombre de índice o cambiá EMBED_DIM."
        )
    return pc.Index(config.INDEX_NAME)


def upsert_in_batches(index, chunks: List[Dict], vectors: List[List[float]]) -> int:
    total = 0
    for start in range(0, len(chunks), config.UPSERT_BATCH_SIZE):
        batch = [
            {
                "id": c["id"],
                "values": v,
                # El texto original vive en los metadatos: no hace falta otra DB.
                "metadata": {**c["metadata"], "text": c["text"]},
            }
            for c, v in zip(chunks[start:start + config.UPSERT_BATCH_SIZE],
                            vectors[start:start + config.UPSERT_BATCH_SIZE])
        ]
        index.upsert(vectors=batch, namespace=config.NAMESPACE)
        total += len(batch)
        print(f"  upsert {total}/{len(chunks)}")
    return total


def save_chunks_locally(chunks: List[Dict]) -> None:
    """Copia local usada como corpus del recuperador léxico (BM25)."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with config.CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c["id"], "text": c["text"], "metadata": c["metadata"]},
                               ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="borra el namespace antes de ingerir")
    args = parser.parse_args()
    config.require_keys()

    from embeddings import GeminiEmbeddings

    chunks = build_chunks(load_documents())
    if not chunks:
        raise SystemExit(f"No se encontraron documentos en {config.DOCS_DIR}")
    print(f"{len(chunks)} chunks generados")

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = ensure_index(pc)
    if args.reset:
        try:
            index.delete(delete_all=True, namespace=config.NAMESPACE)
        except Exception as exc:  # el namespace puede no existir todavía
            print(f"(reset omitido: {exc})")

    vectors = GeminiEmbeddings().embed_documents([c["text"] for c in chunks])
    upsert_in_batches(index, chunks, vectors)
    save_chunks_locally(chunks)
    print("Estadísticas del índice:", index.describe_index_stats())


if __name__ == "__main__":
    main()
