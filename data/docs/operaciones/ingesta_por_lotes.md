---
tags: batching, upsert, ingesta
---
# Ingesta por lotes

Subir vectores uno por uno es ineficiente por el overhead de red y los límites de la API. La estrategia correcta es el batch upsert con lotes de entre 100 y 200 vectores, que reduce la latencia y aprovecha el paralelismo interno. Los metadatos deben seguir un esquema estricto: si un proceso guarda ingest_date y otro date_created, los filtros fallan (schema drift). Filtrar por metadatos como category con el operador $eq reduce el espacio de búsqueda.
