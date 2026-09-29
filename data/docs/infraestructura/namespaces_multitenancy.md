---
tags: namespace, multitenancy, seguridad
---
# Namespaces y multi-tenancy

Un namespace es una partición lógica dentro de un mismo índice de Pinecone. Permite separar los datos de distintos clientes (multi-tenancy) o los entornos dev, staging y prod sin crear un índice por cada uno. Cada consulta se ejecuta dentro de un único namespace, de modo que un usuario nunca ve vectores de otro. Ignorar los namespaces mezcla datos de prueba con producción, vuelve las búsquedas ruidosas y compromete la seguridad.
