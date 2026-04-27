# Contributing to movie-finder-rag

`rag_ingestion` is the offline Movie Finder ingestion pipeline. It downloads the Wikipedia Movie
Plots dataset from Kaggle, embeds each record with OpenAI/Gemini, and writes vectors into Qdrant Cloud.

---

## Table of Contents

1. [Development setup](#development-setup)
2. [Project structure](#project-structure)
3. [Running the pipeline](#running-the-pipeline)
4. [Adding an embedding provider](#adding-an-embedding-provider)
5. [Adding a vector store backend](#adding-a-vector-store-backend)
6. [Testing strategy](#testing-strategy)
7. [CI/CD pipeline](#ci-cd-pipeline)
8. [Sharing outputs with the chain team](#sharing-outputs-with-the-chain-team)

---

## Development Setup

This repository is intentionally outside the backend `uv` workspace, but local development is
strictly Docker-only. Do not document or rely on host Python environments here.

```bash
cd rag_ingestion/
cp .env.example .env
$EDITOR .env

make init
make editor-up
```

Use `make shell` if you need an interactive shell inside the workspace container, then use the
committed VS Code tasks or launch configs from an attached container.

### Required environment variables

```bash
# Qdrant Cloud (write-capable key)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY_RW=
QDRANT_COLLECTION_NAME=movies

# OpenAI embeddings
OPENAI_API_KEY=sk-...

# Kaggle dataset download
KAGGLE_API_TOKEN=KGAT_...
```

---

## Project Structure

```text
movie-finder-rag/
├── src/
│   └── rag/             # Package source (embeddings, vectorstore, ingestion, config)
├── tui/                 # Textual TUI for retrieval evaluation (tui.app, tui.widgets, tui.constants)
├── scripts/             # Operational scripts (backup, cost-report, evaluate, launch_tui, retrieve)
├── tests/               # Unit tests
├── Dockerfile           # Multi-stage build (dev, builder, runtime)
├── docker-compose.yml   # Local development stack
├── Makefile             # Docker-backed developer contract
└── .vscode/             # Attached-container editor configuration
```

### Import note

This project uses the `src-layout`. Imports must use the absolute package name:
`from rag.module import member`.

---

## Running the Pipeline

### Supported local path

```bash
make ingest
```

This runs the runtime image against external Qdrant Cloud using the values in `.env`.

### Common quality commands

```bash
make lint           # ruff check
make format         # ruff format
make typecheck      # mypy src (hardened DNA)
make test           # pytest
make test-coverage  # pytest + reports
make check          # lint + typecheck + test
```

### Backup the pipeline in ChromaDB

```bash
make backup
```

This target runs through Docker Compose, not the host Python environment. For the current Qdrant
adapter, the existing `.env` values (`QDRANT_URL`, `QDRANT_API_KEY_RW`, `QDRANT_COLLECTION_NAME`)
are enough.

The backup utility is intentionally script-local and CI-friendly. It also accepts generic
vector-store inputs through CLI args or env vars:

```bash
make backup BACKUP_ARGS="--vector-store qdrant --collection-name movies"
```

Script-specific env fallbacks:

| Variable                 | Purpose                            |
| ------------------------ | ---------------------------------- |
| `VECTOR_STORE`           | Backup source type                 |
| `VECTOR_STORE_URL`       | Live vector store endpoint         |
| `VECTOR_STORE_API_KEY`   | Live vector store API key          |
| `BACKUP_COLLECTION_NAME` | Collection name to back up         |
| `BACKUP_OUTPUT_ROOT`     | Output root for archived artifacts |
| `BACKUP_BATCH_SIZE`      | Source pagination batch size       |

Generated artifacts are written under `outputs/backups/chromadb/`.

### Evaluate Qdrant retrieval

The retrieval evaluator is operational tooling under `scripts/`; it is not part of the `src/rag`
runtime package and is not unit-tested as application code. It is intended for post-ingest Jenkins
validation and retroactive checks of existing Qdrant collections:

```bash
make qdrant-live-eval
```

The evaluator writes one artifact folder per collection under `outputs/reports/qdrant-live-eval/`.

### Run interactive retrieval app

```bash
make retrieve
```

### Launch the Textual TUI

The full Textual TUI (`tui/app.py`) provides slash-command navigation for switching providers,
models, stores, and top-k, and displays cosine similarity scores alongside each result:

```bash
make tui
```

### Refresh the ingestion cost report

After ingestion, regenerate `outputs/reports/cost-report.json` from `ingestion-outputs.env`:

```bash
make cost-report
```

---

## Adding an Embedding Provider

This project follows the **Strategy Pattern** for embedding providers. To add a new one:

1.  **Interface**: Ensure your provider implements the `EmbeddingProvider` abstract base class found in `src/rag/embeddings/base.py`.
2.  **Implementation**: Create a new file in `src/rag/embeddings/` (e.g., `my_provider.py`).
3.  **Wiring**: Add the provider to the selection logic in `src/rag/main.py`.
4.  **Configuration**: If the provider requires new secrets, update `src/rag/config.py` (Pydantic `RAGConfig`), `.env.example`, and the documentation.

Example:

```python
from rag.embeddings.base import EmbeddingProvider, EmbeddingModel, EmbeddingModelUsage

class MyProvider(EmbeddingProvider):
    @property
    def model_info(self) -> EmbeddingModel:
        return EmbeddingModel(name="my-model", embedding_dimension=1536)

    def embed(self, text: str) -> list[float]:
        # Implementation...
        pass

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Implementation...
        pass

    def get_model_usage(self) -> EmbeddingModelUsage:
        return self._usage
```

---

## Adding a Vector Store Backend

Vector store backends are abstracted via the `VectorStore` interface in `src/rag/vectorstore/base.py`.

1.  **Implementation**: Create a new module in `src/rag/vectorstore/` implementing the required `upsert`, `upsert_batch`, and `search` methods.
2.  **DNA Alignment**: Ensure your implementation uses `from rag.utils.logger import get_logger` and handles configuration via `from rag.config import settings`.
3.  **Error Handling**: Do not use `sys.exit()`. Raise descriptive exceptions or implement retry logic using the project's standard patterns.

---

## Testing Strategy

The CI contract for this repo is stubbed. Tests must pass without live OpenAI, Qdrant, or Kaggle
credentials.

Follow these rules:

- No real OpenAI, Qdrant, or Kaggle network calls in unit tests
- Mock at the SDK/client boundary
- Prefer focused tests for env resolution and provider/store behavior

---

## CI/CD Pipeline

The Jenkins pipeline executes through the Docker Makefile to ensure environment parity between
local and CI runs.

| Stage                 | Command / trigger                       | Notes                                                  |
| --------------------- | --------------------------------------- | ------------------------------------------------------ |
| Lint + Typecheck      | `make lint` + `make typecheck`          | PRs, `main`, tags                                      |
| Test                  | `make test-coverage`                    | PRs, `main`, tags                                      |
| Ingest                | Manual `RUN_INGESTION=true`             | Parameterized Jenkins / Actions run                    |
| Cost report           | `make cost-report` (post-ingest)        | Writes `outputs/reports/cost-report.json`              |
| Post-ingest validate  | `make validate` (post-ingest)           | Smoke-test query against the new collection            |
| Qdrant live eval      | `make qdrant-live-eval` (post-ingest)   | Hit@k + similarity report; Qdrant only                 |
| Backup                | Manual `RUN_BACKUP=true` or post-ingest | Archives `outputs/backups/**`                          |
| Archive artifacts     | Automatic after ingest / backup         | All outputs downloadable from the Jenkins / Actions UI |

Live CI operations must stay manual and parameterized. Secret handling is owned by the CI system:

- Jenkins credentials: `qdrant-url`, `qdrant-api-key-rw`, `openai-api-key`,
  `kaggle-api-token`
- GitHub Actions secrets: `QDRANT_URL`, `QDRANT_API_KEY_RW`, `OPENAI_API_KEY`,
  `KAGGLE_API_TOKEN`

---

## Sharing Outputs With The Chain Team

After a successful ingest, all relevant metadata is written to `ingestion-outputs.env` and archived
as a Jenkins / GitHub Actions build artifact. Download it from the build page and share:

```text
QDRANT_URL=<cloud-endpoint>
VECTOR_STORE_TARGET_NAME=<adr0008-collection-name>
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072
INGESTION_TOTAL_TOKENS=<n>
INGESTION_ESTIMATED_COST_USD=<n>
```

The human-readable cost summary is also available as `outputs/reports/cost-report.json`.
For Qdrant ingestion runs, `outputs/reports/qdrant-live-eval/` contains per-collection HTML
reports and a `summary.json` with hit@k and mean cosine similarity metrics.
