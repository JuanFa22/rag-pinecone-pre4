"""Recuperador híbrido: Pinecone (vectorial) + BM25 (léxico) con EnsembleRetriever."""
import json
import re
import unicodedata
from typing import Any, List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import config

try:  # LangChain 0.x
    from langchain.retrievers import EnsembleRetriever
except ImportError:  # LangChain 1.x
    from langchain_classic.retrievers import EnsembleRetriever


def tokenize(text: str) -> List[str]:
    """Minúsculas, sin tildes y sin puntuación (mejor BM25 en español)."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9_\-]+", text)


def load_corpus() -> List[Document]:
    if not config.CHUNKS_FILE.exists():
        raise SystemExit("No existe data/processed/chunks.jsonl. Ejecutá primero: python ingest.py")
    docs = []
    with config.CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(page_content=row["text"], metadata={**row["metadata"], "id": row["id"]}))
    return docs


def build_bm25(k: int = 5) -> BM25Retriever:
    return BM25Retriever.from_documents(load_corpus(), k=k, preprocess_func=tokenize)


class WithChunkId(BaseRetriever):
    """Garantiza metadata['id'] en cada resultado (el EnsembleRetriever lo necesita).

    PineconeVectorStore devuelve el id del vector en `Document.id`, no en los
    metadatos; si tampoco está, se reconstruye desde doc_id/page/chunk.
    """

    inner: Any

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        docs = self.inner.invoke(query)
        for d in docs:
            if "id" not in d.metadata:
                d.metadata["id"] = d.id or (
                    f'{d.metadata.get("doc_id")}::p{int(d.metadata.get("page", 1))}'
                    f'::c{int(d.metadata.get("chunk", 0))}'
                )
        return docs


def build_vector_retriever(k: int = 5):
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore
    from embeddings import GeminiEmbeddings

    index = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.INDEX_NAME)
    store = PineconeVectorStore(index=index, embedding=GeminiEmbeddings(),
                                namespace=config.NAMESPACE, text_key="text")
    return WithChunkId(inner=store.as_retriever(search_kwargs={"k": k}))


def build_hybrid_retriever(k: int = 5, vector_weight: float = 0.6) -> EnsembleRetriever:
    """EnsembleRetriever fusiona los rankings con Reciprocal Rank Fusion (RRF) ponderado."""
    return EnsembleRetriever(
        retrievers=[build_vector_retriever(k), build_bm25(k)],
        weights=[vector_weight, 1.0 - vector_weight],
        id_key="id",
    )


if __name__ == "__main__":
    import sys
    config.require_keys()
    query = " ".join(sys.argv[1:]) or "¿Qué significa el error ERR_DIM_MISMATCH?"
    for i, d in enumerate(build_hybrid_retriever().invoke(query)[:5], start=1):
        print(f"{i}. [{d.metadata.get('doc_id')}] {d.page_content[:100]!r}")
