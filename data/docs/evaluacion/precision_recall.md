---
tags: metricas, precision, recall
---
# Precision@k y Recall@k

Precision@k mide qué proporción de los k documentos recuperados es realmente relevante: evita el ruido. Recall@k mide qué proporción de los documentos relevantes existentes aparece en el top-k: evita dejar información útil afuera. En un sistema RAG, un recall bajo provoca alucinaciones por falta de contexto y una precisión baja introduce ruido que confunde al modelo. Ambas se calculan contra un golden set de preguntas con su documento esperado.
