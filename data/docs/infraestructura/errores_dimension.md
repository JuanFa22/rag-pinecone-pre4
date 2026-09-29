---
tags: error, dimension, troubleshooting
---
# Errores de dimensión

El error ERR_DIM_MISMATCH aparece cuando la dimensión del vector no coincide con la del índice. Por ejemplo, subir embeddings de 1536 dimensiones a un índice creado con 768 falla de inmediato. La solución es alinear la dimensión del índice con el modelo de embeddings, o recrear el índice. Con gemini-embedding-001 se puede fijar output_dimensionality en 768 y crear el índice con esa misma dimensión.
