"""Embeddings de Gemini expuestos como `Embeddings` de LangChain."""
import math
import time
from typing import List

from google import genai
from google.genai import types
from langchain_core.embeddings import Embeddings

import config


class GeminiEmbeddings(Embeddings):
    def __init__(self, model: str = config.EMBED_MODEL, dimension: int = config.EMBED_DIM,
                 batch_size: int = 50):
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        # gemini-embedding-001 solo devuelve vectores normalizados en 3072 dims;
        # con dimensiones reducidas hay que normalizar para que coseno sea correcto.
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    def _embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            for attempt in range(5):
                try:
                    res = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=self.dimension,
                        ),
                    )
                    break
                except Exception:  # límites de cuota / errores transitorios
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            out.extend(self._normalize(list(e.values)) for e in res.embeddings)
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]
