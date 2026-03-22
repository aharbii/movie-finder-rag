# movie-finder-rag

AI/Data ingestion pipeline — downloads the [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) Kaggle dataset, embeds it with OpenAI, and loads it into a Qdrant vector store.

This package is maintained by the **AI/Data Engineering team** and is a prerequisite for the `chain` package.

---

## Overview

```
Kaggle CSV → CSV Loader → Embedding Provider → Vector Store
               (pandas)    (OpenAI / Gemini)   (Qdrant / ChromaDB)
```

The pipeline is designed to be re-run whenever the dataset is updated or a new embedding model is tested.

---

## Repository

- GitHub: `https://github.com/aharbii/movie-finder-rag`
- Consumed by: `chain` (reads from the resulting Qdrant collection)

---

## Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for local Qdrant)
- Kaggle account + API token

### Install

```bash
# Production only
uv sync --frozen --no-dev

# With dev tools (lint + test)
uv sync --group dev
```

### Configure environment

```bash
cp .env.example .env
$EDITOR .env   # fill in QDRANT_*, OPENAI_API_KEY, KAGGLE_*
```

---

## Running

### Local (with local Qdrant)

```bash
docker compose up qdrant -d
QDRANT_ENDPOINT=http://localhost:6333 QDRANT_API_KEY="" python -m src.main
```

### Against Qdrant Cloud

```bash
# Fill QDRANT_ENDPOINT + QDRANT_API_KEY in .env
python -m src.main
```

### Via Docker

```bash
docker build -t movie-finder-rag .
docker run --rm --env-file .env movie-finder-rag

# Or with docker compose (local Qdrant):
docker compose --profile run up --build ingestion
```

---

## Testing

```bash
uv sync --group test && uv run pytest tests/ -v
```

## Linting

```bash
uv sync --group lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run mypy src/
```

---

## Configuration

All configuration via environment variables (see `.env.example`):

| Variable | Required | Default | Description |
|---|---|---|---|
| `QDRANT_ENDPOINT` | Yes | — | Qdrant endpoint URL |
| `QDRANT_API_KEY` | Yes | — | Qdrant API key (empty for local) |
| `QDRANT_COLLECTION` | No | `movies` | Target collection name |
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for embeddings |
| `EMBEDDING_MODEL` | No | `text-embedding-3-large` | Embedding model |
| `EMBEDDING_DIMENSION` | No | `3072` | Vector dimension |
| `KAGGLE_USERNAME` | Yes | — | Kaggle username |
| `KAGGLE_KEY` | Yes | — | Kaggle API key |
| `VECTOR_STORE` | No | `qdrant` | `qdrant` or `chromadb` |

---

## Sharing outputs with the chain team

After a successful ingestion run, share these values with the chain team **via Jenkins credentials store** (not plain text):

```
QDRANT_ENDPOINT=<qdrant-cloud-endpoint>
QDRANT_API_KEY=<qdrant-api-key>
QDRANT_COLLECTION=<collection-name>
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072
```

---

## CI/CD

See `Jenkinsfile` for the full pipeline.

| Stage | Trigger | Description |
|---|---|---|
| Lint | Every PR + tag | ruff + mypy |
| Test | Every PR + tag | pytest + coverage |
| Build Image | `main` + tags | Docker build + push |
| Ingest | Manual (`RUN_INGESTION=true`) | Full dataset ingestion |

Manual ingest trigger: Jenkins → Build with Parameters → check `RUN_INGESTION`.
