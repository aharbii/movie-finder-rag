# Contributing to movie-finder-rag

The rag_ingestion project is a **one-shot batch pipeline** that downloads the Wikipedia Movie Plots dataset from Kaggle, embeds it with an embedding provider, and loads it into a vector store (Qdrant or ChromaDB). It is the data backbone for the `chain` package.

For org-wide conventions (branching, commits, PRs, release process, issue/PR template inspection,
and AI disclosure in PR descriptions or AI-assisted reviews) see the
[backend CONTRIBUTING.md](../CONTRIBUTING.md).

---

## Table of contents

1. [Development setup](#development-setup)
2. [Project structure](#project-structure)
3. [Running the pipeline](#running-the-pipeline)
4. [Adding an embedding provider](#adding-an-embedding-provider)
5. [Adding a vector store backend](#adding-a-vector-store-backend)
6. [Testing strategy](#testing-strategy)
7. [Triggering ingestion via Jenkins](#triggering-ingestion-via-jenkins)
8. [Sharing outputs with the chain team](#sharing-outputs-with-the-chain-team)

---

## Development setup

The rag_ingestion project is a **standalone repo** — it is NOT part of the backend uv workspace. It has its own `uv.lock`.

```bash
cd rag_ingestion/
uv sync --group dev
cp .env.example .env && $EDITOR .env
uv run pre-commit install

# Start local Qdrant for development
docker compose up qdrant -d
```

### Required environment variables

```
# Vector store
QDRANT_ENDPOINT=http://localhost:6333   # or Qdrant Cloud URL
QDRANT_API_KEY=                         # empty for local
QDRANT_COLLECTION=movies

# Embeddings
OPENAI_API_KEY=sk-
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072

# Kaggle (for dataset download)
KAGGLE_USERNAME=
KAGGLE_KEY=                             # from kaggle.com → Settings → API
```

---

## Project structure

```
rag_ingestion/
├── src/
│   ├── main.py               ← entry point: orchestrates the full pipeline
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── dataset.py        ← download_data() from Kaggle via kagglehub
│   ├── embeddings/
│   │   ├── base.py           ← EmbeddingProvider ABC
│   │   ├── openai_provider.py   ← OpenAI text-embedding-* implementation
│   │   └── gemini_provider.py   ← Google Gemini implementation
│   ├── ingestion/
│   │   ├── csv_loader.py     ← load_movies() from CSV → list[Movie]
│   │   └── pipeline.py       ← ingest_csv() orchestrator
│   ├── models/
│   │   └── movie.py          ← Movie Pydantic model
│   ├── vectorstore/
│   │   ├── base.py           ← VectorStore ABC
│   │   ├── qdrant_vectorstore.py    ← Qdrant Cloud implementation
│   │   └── chromadb_vectorstore.py  ← local ChromaDB implementation
│   └── utils/
│       └── logger.py         ← get_logger factory
├── scripts/
│   └── backup_vectorstore.py ← Qdrant → ChromaDB migration utility
└── tests/
    └── test_rag.py
```

### Import note

This project uses **flat imports** — modules import each other without the `src.` prefix (e.g. `from embeddings.base import EmbeddingProvider`). Always run Python with `PYTHONPATH=src` or use `uv run` which picks it up from `[tool.pytest.ini_options] pythonpath = ["src"]`.

```bash
# correct
PYTHONPATH=src python -m main
uv run python -m main        # uv handles PYTHONPATH automatically via pyproject.toml

# wrong
python src/main.py           # imports will fail
```

---

## Running the pipeline

### Local (against local Qdrant)

```bash
docker compose up qdrant -d
PYTHONPATH=src uv run python -m main
```

### Against Qdrant Cloud

Fill `QDRANT_ENDPOINT` and `QDRANT_API_KEY` in `.env`, then:

```bash
uv run python -m main
```

### Via Docker

```bash
docker build -t movie-finder-rag .
docker run --rm --env-file .env movie-finder-rag

# Using docker compose (local Qdrant):
docker compose --profile run up --build ingestion
```

---

## Adding an embedding provider

1. **Create** `src/embeddings/my_provider.py` implementing `EmbeddingProvider`:

```python
from embeddings.base import EmbeddingProvider

class MyEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str, **kwargs):
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per text."""
        # call your embedding API here
        return [[0.0] * 1536 for _ in texts]   # placeholder

    @property
    def dimension(self) -> int:
        return 1536
```

2. **Export** from `src/embeddings/__init__.py`.

3. **Wire up** in `src/main.py`:

```python
from embeddings.my_provider import MyEmbeddingProvider

# swap out the provider:
embedding_provider = MyEmbeddingProvider(model="my-model")
```

4. **Add tests** in `tests/` mocking the external API call.

5. **Add env vars** to `.env.example` and `rag_ingestion/.env.example` if the provider needs credentials.

---

## Adding a vector store backend

1. **Create** `src/vectorstore/my_store.py` implementing `VectorStore`:

```python
from vectorstore.base import VectorStore
from models.movie import Movie

class MyVectorStore(VectorStore):
    def upsert(self, movies: list[Movie], vectors: list[list[float]]) -> None:
        """Insert or update movie vectors. Called once per batch."""
        ...

    def collection_exists(self) -> bool:
        """Return True if the target collection already exists."""
        ...
```

2. **Wire up** in `src/main.py` alongside existing implementations.

3. **Add env vars** to `.env.example` for any connection config.

---

## Testing strategy

**Rule: no real API calls or file I/O** in unit tests.

```bash
# Run all tests
uv sync --group test && PYTHONPATH=src uv run pytest tests/ -v

# With coverage
PYTHONPATH=src uv run pytest tests/ --cov=src --cov-report=term-missing
```

Mock patterns to follow:
- Kaggle download: `mocker.patch("dataset.dataset.kagglehub.dataset_download")`
- OpenAI embeddings: `mocker.patch("embeddings.openai_provider.OpenAI")`
- Qdrant client: `mocker.patch("vectorstore.qdrant_vectorstore.QdrantClient")`

---

## Triggering ingestion via Jenkins

The `Ingest` stage in `Jenkinsfile` is **manual-only** — it does not run on every PR.

**To trigger a new ingestion run:**

1. Open the `movie-finder-rag` pipeline in Jenkins
2. Click **Build with Parameters**
3. Set the parameters:
   - `RUN_INGESTION` = **true**
   - `COLLECTION_NAME` = name for the new collection (e.g. `movies-v2`)
   - `VECTOR_STORE` = `qdrant` (or `chromadb` for a local test run)
   - `FORCE_DOWNLOAD` = `true` if you want to re-download the dataset
4. Click **Build**

After a successful run, Jenkins archives an `ingestion-outputs.env` artifact containing the collection name and model used.

---

## Sharing outputs with the chain team

After a successful ingestion run, share these values with the chain team **via Jenkins credentials store** — never via Slack, email, or plain text:

```
QDRANT_ENDPOINT=<cloud-endpoint>       ← rarely changes
QDRANT_API_KEY=<api-key>               ← rarely changes
QDRANT_COLLECTION=<new-collection>     ← changes with each ingest run
EMBEDDING_MODEL=text-embedding-3-large ← confirm the model used
EMBEDDING_DIMENSION=3072
```

**Handoff steps:**
1. Download the `ingestion-outputs.env` artifact from the Jenkins build
2. Update the `qdrant-collection` Jenkins credential in the chain repo pipeline
3. Notify the chain team (Slack / ticket) that new vectors are available
4. The chain team's next build automatically picks up the new collection
