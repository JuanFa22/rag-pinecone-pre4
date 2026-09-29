---
tags: chunking, splitter, contexto
---
# Estrategias de chunking

Los chunks demasiado pequeños pierden el contexto semántico y los demasiado grandes diluyen la precisión del embedding. Un punto medio razonable es de 500 a 800 tokens con un solape de entre 10 y 15 por ciento. El RecursiveCharacterTextSplitter corta primero por párrafos, luego por líneas y por último por palabras, respetando la estructura del texto. Conviene guardar el texto original del chunk en los metadatos para evitar consultas adicionales.
