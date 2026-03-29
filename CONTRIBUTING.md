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
rag_ingestion/
├── src/
│   └── rag/             # Package source
├── scripts/             # Utility scripts (backup, retrieve)
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

| Stage | Command / trigger | Notes |
|---|---|---|
| Lint + Typecheck | `make lint` + `make typecheck` | PRs, `main`, tags |
| Test | `make test-coverage` | PRs, `main`, tags |
| Build Image | `docker build --target runtime ...` | `main` and tags |
| Ingest | Manual `RUN_INGESTION=true` | Triggered via Jenkins parameters |

---

## Sharing Outputs With The Chain Team

After a successful ingest, share these metadata values with the chain team:

```text
QDRANT_URL=<cloud-endpoint>
QDRANT_COLLECTION_NAME=<new-collection>
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072
```
