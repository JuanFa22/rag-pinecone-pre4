# Sistema RAG escalable en la nube con Pinecone (Pre-entrega 4)

Módulo de recuperación escalable en Python: ingesta a **Pinecone Serverless** con metadatos avanzados,
**recuperador híbrido** (vectorial + BM25 con `EnsembleRetriever`) y **evaluación** con Precision@k / Recall@k.

## Arquitectura

```
data/docs (.md/.json/.pdf)
   └─► ingest.py ─► chunking (\~600 tokens) ─► embeddings Gemini (768 dim) ─► Pinecone (namespace)
                          └─► data/processed/chunks.jsonl (corpus para BM25)
consulta ─► hybrid\_retriever.py ─► \[Pinecone vectorial + BM25] ─► RRF ponderado (EnsembleRetriever)
data/golden\_set.json ─► evaluate.py ─► Precision@k / Recall@k (vectorial vs BM25 vs híbrido)
```

Decisiones clave:

* **Embeddings:** `gemini-embedding-001` con `output\_dimensionality=768` (vectores normalizados). El índice se crea con **dimensión 768 y métrica coseno**. Si usaras OpenAI `text-embedding-3-small` habría que crear el índice con 1536.
* **Metadatos (esquema estricto):** `doc\_id`, `source`, `page`, `category`, `tags`, `chunk` y `text` (el texto original queda dentro del metadato, sin necesidad de otra base de datos).
* **Namespace:** `docs-v1` (configurable) para aislar datos por entorno/cliente.
* **Chunking:** `RecursiveCharacterTextSplitter`, 2400 caracteres (\~600 tokens) con solape de 300.
* **Ingesta por lotes:** `upsert` en batches de 100 vectores.
* **Híbrido:** `EnsembleRetriever` (Reciprocal Rank Fusion ponderado, 0.6 vectorial / 0.4 BM25).

## Pasos para replicar el índice

```bash
git clone https://github.com/<usuario>/<repo>.git \&\& cd <repo>
python -m venv .venv \&\& source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

cp .env.example .env        # completá PINECONE\_API\_KEY y GOOGLE\_API\_KEY (Gemini)

python ingest.py            # crea el índice serverless (aws/us-east-1) y sube los chunks
python evaluate.py --k 5    # imprime Precision@5 y Recall@5
python hybrid\_retriever.py "¿Qué significa el error ERR\_DIM\_MISMATCH?"   # prueba rápida
```

Para volver a construir el namespace desde cero: `python ingest.py --reset`.

## Evaluación

`data/golden\_set.json` contiene pares `{"pregunta": "...", "documento\_id\_esperado": "..."}`
(también acepta una lista de ids). Para cada pregunta se toman los top-k chunks:

* **Precision@k** = chunks del top-k que pertenecen a un documento esperado / k
* **Recall@k** = documentos esperados presentes en el top-k / documentos esperados

`evaluate.py` imprime la tabla en consola comparando **vectorial**, **BM25** e **híbrido**.
Con un único documento esperado por pregunta y un chunk por documento, Precision@5 está acotada
en 0.2, por lo que la métrica más informativa en este set pequeño es Recall@5.

| Modo | Precision@5 | Recall@5 |

|------|-------------|----------|

| vectorial (Pinecone) | 0.200 | 1.000 |

| léxico (BM25) | 0.200 | 1.000 |

| híbrido (Ensemble) | 0.200 | 1.000 |



Los tres modos encuentran el documento esperado en las 8 preguntas (Recall@5 = 1.0). Con un solo documento relevante por pregunta y un chunk por documento, Precision@5 tiene un máximo de 0.2. Con un golden set más grande y difícil, el recuperador híbrido debería mostrar ventaja en consultas con códigos o nombres exactos.Errores comunes que contempla el proyecto

* **Mismatch de dimensiones:** `ingest.py` valida que la dimensión del índice coincida con `EMBED\_DIM` y aborta con un mensaje claro.
* **Namespaces:** todas las operaciones (upsert, búsqueda) usan `PINECONE\_NAMESPACE`.
* **Chunking:** tamaño intermedio con solape para no perder contexto ni diluir el embedding.

## Estructura

```
config.py  embeddings.py  ingest.py  hybrid\_retriever.py  evaluate.py
data/docs/…   data/golden\_set.json   requirements.txt   .env.example
```

