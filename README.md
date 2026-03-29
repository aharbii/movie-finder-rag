# movie-finder-rag

Offline ingestion pipeline for Movie Finder. Downloads the [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) Kaggle dataset, generates OpenAI embeddings, and upserts them into a Qdrant Cloud collection.

Contributor workflow in this repo is **strictly Docker-only**: all quality gates (lint,
test, typecheck, test-coverage, pre-commit) execute through the provided `Makefile`.
Host-managed Python environments are not supported for development.

---

## Overview

```text
Kaggle CSV -> CSV Loader -> Embedding Provider -> Qdrant Cloud
               (pandas)      (OpenAI/Gemini)      (write-capable key)
```

The pipeline is re-runnable whenever the dataset changes or a fresh collection needs to be built
for the `chain` package.

---

## Local Workflow

### Prerequisites

- Docker 24+ with the Compose plugin
- GNU Make
- Qdrant Cloud write-capable credentials
- OpenAI API key
- Kaggle API token (`KGAT_` prefixed token)

### Setup

```bash
cp .env.example .env
$EDITOR .env   # Fill in API keys

make init
make editor-up
```

`make editor-up` starts the long-lived workspace container used for VS Code attach/debug flows.
The actual ingestion pipeline is still a one-shot command.

### Common Commands

```bash
make init           # build dev + runtime images
make editor-up      # start the attached-container workspace
make editor-down    # stop the local workspace container
make shell          # shell into the workspace container

make lint           # ruff check
make format         # ruff format
make typecheck      # mypy src (hardened DNA)
make test           # pytest tests/
make test-coverage  # pytest + coverage.xml + htmlcov
make pre-commit     # repo hooks inside Docker
make check          # lint + typecheck + test
```

### Ingestion

```bash
make ingest
```

`make ingest` runs the runtime image against external Qdrant Cloud using the variables in `.env`.
No local Qdrant compose workflow exists in this repo.

---

## VS Code

The committed `.vscode/` config assumes this workflow:

1. Run `make editor-up` from the host workspace.
2. Use `Dev Containers: Attach to Running Container...`.
3. Attach to the `rag` service container started from this repo.
4. Use the committed tasks for `make ...` targets.
5. Use the committed launch configs for ingestion or pytest inside the container.

The interpreter path inside the container is `/opt/venv/bin/python`.

---

## Environment

Canonical variables for this repo (DNA aligned with `infrastructure#9`):

| Variable | Required | Description |
|---|---|---|
| `QDRANT_URL` | Yes | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY_RW` | Yes | Write-capable Qdrant API key |
| `QDRANT_COLLECTION_NAME` | Yes | Shared collection name consumed by `chain` |
| `OPENAI_API_KEY` | Yes | OpenAI API key for embeddings |
| `KAGGLE_API_TOKEN` | Yes | Standalone token prefixed with `KGAT_` |

---

## CI/CD Pipeline

See `Jenkinsfile` for the full pipeline. The current stages line up with the repo-local
developer contract:

| Stage | Command / trigger | Notes |
|---|---|---|
| Lint + Typecheck | `make lint` + `make typecheck` | PRs, `main`, tags |
| Test | `make test-coverage` | PRs, `main`, tags |
| Build Image | `docker build --target runtime ...` | `main` and tags |
| Ingest | Manual `RUN_INGESTION=true` | Triggered via Jenkins parameters |

---

## Sharing Outputs With Chain

After a successful ingestion run, share these values with the chain team:

```text
QDRANT_URL=<qdrant-cloud-endpoint>
QDRANT_COLLECTION_NAME=<collection-name>
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072
```
