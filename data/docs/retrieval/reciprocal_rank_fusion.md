---
tags: rrf, ranking, fusion
---
# Reciprocal Rank Fusion (RRF)

RRF fusiona listas ordenadas de distintos recuperadores usando solo la posición de cada documento: el puntaje es la suma de 1 / (k + rank) en cada lista. No requiere normalizar puntajes que provienen de escalas diferentes, como la distancia coseno y el puntaje BM25. Por eso es la estrategia preferida para combinar resultados heterogéneos. En LangChain, el EnsembleRetriever aplica RRF ponderado con el parámetro weights.
