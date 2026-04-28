# Movie Finder RAG

Offline ingestion pipeline for Movie Finder. It downloads the
[Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
dataset, generates embeddings through the ADR 0008 provider factory, and writes vectors into the
configured store.

Contributor workflow in this repo is Docker-only. Lint, tests, type checks, ingestion, validation,
and backup all run through the committed `Makefile`. Host-managed Python environments are not part
of the supported developer workflow.

---

## Overview

```text
Kaggle CSV -> CSV Loader -> Embedding Provider Factory -> Vector Store Factory
               (pandas)      (OpenAI/Ollama/HuggingFace/Google) (Qdrant/Chroma/Pinecone/PGVector)
```

ADR 0008 contract:

- Embedding runtime is environment-driven via `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL`.
- Heavy local SDKs are installed only when `WITH_PROVIDERS=local`.
- Final target name is resolved as
  `{VECTOR_COLLECTION_PREFIX}_{sanitized_model}_{dimension}`.
- Query-time and ingestion-time embeddings must still be coordinated with `chain/`.

---

## Local Workflow

### Prerequisites

- Docker 24+ with the Compose plugin
- GNU Make
- Kaggle API token (`KGAT_...`)
- Provider credentials or a local runtime, depending on `EMBEDDING_PROVIDER`
- Vector store credentials for the selected `VECTOR_STORE`

### Setup

```bash
make init
make editor-up
```

If you want local HuggingFace / sentence-transformers support, set `WITH_PROVIDERS=local` before
building:

```bash
WITH_PROVIDERS=local make init
```

### Common Commands

```bash
make init
make editor-up
make editor-down
make shell

make lint
make fix
make format
make typecheck
make test
make test-coverage
make pre-commit
make check

make ingest
make validate
make backup
make retrieve
```

### Ingestion

```bash
make ingest
```

`make ingest` runs the runtime image against the configured vector store using variables from
`.env`. For remote backends, credentials stay external to the repo. For `chromadb`, persistence is
through the mounted workspace path inside Docker.

### Validation

```bash
make validate
```

`make validate` performs a post-ingestion smoke check using the configured provider and vector
store. It is intended to confirm that the freshly written target can be queried end-to-end, not to
replace a broader retrieval evaluation suite.

### Backup

```bash
make backup
```

`make backup` runs inside Docker and exports the configured source backend into a portable ChromaDB
artifact. This lets CI archive one stable backup format even when the source backend is `qdrant`,
`chromadb`, `pinecone`, or `pgvector`.

---

## Provider Configuration

The canonical embedding settings are:

| Variable | Required | Description |
| --- | --- | --- |
| `EMBEDDING_PROVIDER` | Yes | `openai`, `ollama`, `huggingface`, `sentence-transformers`, or `google` |
| `EMBEDDING_MODEL` | Yes | Model identifier for the selected provider |
| `EMBEDDING_DIMENSION` | No | Optional dimension override for providers that support it |
| `WITH_PROVIDERS` | No | Docker build extra, usually `local` for sentence-transformers + torch |

Provider-specific credentials:

| Variable | Required When | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | `EMBEDDING_PROVIDER=openai` | OpenAI embeddings API key |
| `GOOGLE_API_KEY` | `EMBEDDING_PROVIDER=google` | Google embeddings API key |
| `OLLAMA_BASE_URL` | `EMBEDDING_PROVIDER=ollama` | Ollama base URL, local or cloud |
| `OLLAMA_API_KEY` | Optional | Required for Ollama cloud |
| `SENTENCE_TRANSFORMERS_CACHE_DIR` | Optional | Cache directory for local HuggingFace models |

---

## Vector Store Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `VECTOR_STORE` | Yes | `qdrant`, `chromadb`, `pinecone`, or `pgvector` |
| `VECTOR_COLLECTION_PREFIX` | Yes | Prefix used for ADR 0008 target naming |
| `QDRANT_URL` | If qdrant | Qdrant endpoint |
| `QDRANT_API_KEY_RW` | If qdrant | Qdrant write-capable key |
| `PINECONE_INDEX_HOST` | If pinecone | Pinecone host |
| `PINECONE_API_KEY` | If pinecone | Pinecone API key |
| `PGVECTOR_DSN` | If pgvector | pgvector PostgreSQL DSN |

Backend-specific settings:

| Variable | Required When | Description |
| --- | --- | --- |
| `QDRANT_URL` | `VECTOR_STORE=qdrant` | Qdrant cluster URL |
| `QDRANT_API_KEY_RW` | `VECTOR_STORE=qdrant` | Write-capable Qdrant API key |
| `CHROMADB_PERSIST_PATH` | `VECTOR_STORE=chromadb` | Local persistence path inside Docker, default `outputs/chromadb/local` |
| `PINECONE_API_KEY` | `VECTOR_STORE=pinecone` | Pinecone API key |
| `PINECONE_INDEX_NAME` | `VECTOR_STORE=pinecone` | Pinecone index name |
| `PINECONE_INDEX_HOST` | Optional | Existing Pinecone host override |
| `PINECONE_CLOUD` / `PINECONE_REGION` | Optional | Serverless creation settings |
| `PGVECTOR_DSN` | `VECTOR_STORE=pgvector` | PostgreSQL DSN with pgvector enabled |
| `PGVECTOR_SCHEMA` | Optional | Schema token used for pgvector target names |

---

## Ingestion Outputs

`make ingest` now emits:

- `ingestion-outputs.env`
- `outputs/reports/cost-report.json`
- `outputs/reports/skipped-movies.json`

`make validate` emits:

- `outputs/reports/validation-report.json`

`make backup` emits:

- `outputs/backups/**`
- `outputs/backups/**/backup-manifest.json`

Backups always emit a portable ChromaDB artifact, regardless of whether the source vector store is
`qdrant`, `chromadb`, `pinecone`, or `pgvector`. The manifest records the source backend together
with the embedding provider, model, and final dimension so the artifact can be traced back to the
ingestion run that created it.

These artifacts are consumed by the Jenkins job for archiving and for backing up the dynamically
resolved target name after ingestion.

---

## VS Code

The committed `.vscode/` config assumes this workflow:

1. Run `make editor-up` from this repo.
2. Use `Dev Containers: Attach to Running Container...`.
3. Attach to the `rag` service container started from this repo.
4. Use the committed tasks for `make ...` targets.
5. Use the committed launch configs for ingestion or pytest inside the container.

The interpreter path inside the container is `/opt/venv/bin/python`.

---

## CI/CD

The repo-local `Jenkinsfile` and `.github/workflows/ci.yml` both accept the ADR 0008 provider
settings directly:

- `EMBEDDING_PROVIDER`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSION`
- `EMBEDDING_API_KEY`
- `VECTOR_STORE`
- `OLLAMA_BASE_URL`
- `CHROMADB_PERSIST_PATH`
- `PINECONE_INDEX_NAME`
- `PINECONE_INDEX_HOST`
- `PINECONE_CLOUD`
- `PINECONE_REGION`
- `PGVECTOR_SCHEMA`
- `WITH_PROVIDERS`

Post-ingest CI flow:

1. `make ingest`
2. `make validate`
3. `make backup`
4. Archive `ingestion-outputs.env`, `outputs/reports/**`, and `outputs/backups/**`

Live CI operations remain manual-only for ingestion-style runs. Credentials should be provided by
the CI system rather than committed env files.

---

## Sharing Outputs With Chain

After a successful ingestion run, share these values with the chain team:

```text
VECTOR_STORE=<backend>
VECTOR_STORE_TARGET_NAME=<resolved-target-name>
EMBEDDING_PROVIDER=<provider>
EMBEDDING_MODEL=<model>
EMBEDDING_DIMENSION=<final-dimension>
```
