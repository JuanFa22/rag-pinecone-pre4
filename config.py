"""Configuración central del proyecto (lee variables de entorno / .env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "data" / "docs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
GOLDEN_SET_FILE = BASE_DIR / "data" / "golden_set.json"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-escalable")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "docs-v1")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
# La dimensión del índice DEBE coincidir con la de los embeddings.
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# Chunking: ~2400 caracteres ≈ 600 tokens (punto medio recomendado 500-800).
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
UPSERT_BATCH_SIZE = 100


def require_keys() -> None:
    missing = [n for n, v in (("PINECONE_API_KEY", PINECONE_API_KEY),
                              ("GOOGLE_API_KEY", GOOGLE_API_KEY)) if not v]
    if missing:
        raise SystemExit(f"Faltan variables de entorno: {', '.join(missing)} (ver .env.example)")
