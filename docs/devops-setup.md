# DevOps Setup

This repo's Jenkins job is parameterized for the ADR 0008 provider factory contract.

This repo keeps the original operational split:

- PR and `main` builds run quality gates only.
- Ingestion, validation, and backup are manual operations against configured backends.
- All job execution stays inside Docker; local-host assumptions are out of scope.

---

## Parameters

| Parameter | Purpose |
| --- | --- |
| `EMBEDDING_PROVIDER` | Selects the embedding provider |
| `EMBEDDING_MODEL` | Selects the provider-specific embedding model |
| `EMBEDDING_DIMENSION` | Optional output dimension override |
| `EMBEDDING_API_KEY` | Optional override for OpenAI, Google, or Ollama cloud |
| `BACKUP_FORMAT` | Backup format archived by Jenkins, currently `chromadb` |
| `OLLAMA_BASE_URL` | Docker-reachable Ollama URL when `EMBEDDING_PROVIDER=ollama` |
| `VECTOR_COLLECTION_PREFIX` | Optional prefix override; default is `movies_<git sha8>` |
| `VECTOR_STORE` | Selects the ingestion backend |
| `CHROMADB_PERSIST_PATH` | Persistent path when `VECTOR_STORE=chromadb` |
| `PINECONE_INDEX_NAME` | Pinecone index name |
| `PINECONE_INDEX_HOST` | Optional Pinecone host override |
| `PINECONE_CLOUD` | Pinecone serverless cloud |
| `PINECONE_REGION` | Pinecone serverless region |
| `PGVECTOR_DSN` | Optional pgvector DSN override |
| `PGVECTOR_SCHEMA` | Schema token for pgvector table derivation |
| `WITH_PROVIDERS` | Docker build extra, typically `local` |
| `VALIDATION_QUERY` | Smoke-test query for post-ingest validation |

---

## Jenkins Credentials

| Credential ID | Used For |
| --- | --- |
| `qdrant-url` | Default `QDRANT_URL` when `VECTOR_STORE=qdrant` |
| `qdrant-api-key-rw` | Default `QDRANT_API_KEY_RW` when `VECTOR_STORE=qdrant` |
| `openai-api-key` | Default `OPENAI_API_KEY` when `EMBEDDING_PROVIDER=openai` |
| `google-api-key` | Default `GOOGLE_API_KEY` when `EMBEDDING_PROVIDER=google` |
| `ollama-api-key` | Default `OLLAMA_API_KEY` for Ollama cloud |
| `pinecone-api-key` | Default `PINECONE_API_KEY` when `VECTOR_STORE=pinecone` |
| `pgvector-dsn` | Default `PGVECTOR_DSN` when `VECTOR_STORE=pgvector` |
| `kaggle-api-token` | Required dataset download token |

---

## Backup Scope

The Jenkins backup path uses `BACKUP_FORMAT=chromadb`: the job exports the configured source
collection into a portable ChromaDB directory under `outputs/backups/<backend>/<collection>/`.
Jenkins archives that directory directly so the backup is downloadable from the Jenkins build.
Portable backup is available for every supported source backend: `qdrant`, `chromadb`, `pinecone`,
and `pgvector`.

---

## Archived Artifacts

After manual ingest or backup runs, the job archives:

- `ingestion-outputs.env`
- `outputs/reports/cost-report.json`
- `outputs/reports/skipped-movies.json`
- `outputs/reports/validation-report.json`
- `outputs/reports/qdrant-live-eval/**`
- `outputs/backups/**`
