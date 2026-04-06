# JetBrains AI (Junie) — rag_ingestion submodule guidelines

This is **`movie-finder-rag`** (`backend/rag_ingestion/`) — Offline embedding ingestion pipeline.
GitHub repo: `aharbii/movie-finder-rag` · Parent: `aharbii/movie-finder`

---

## What this submodule does

One-shot offline pipeline: downloads the movie dataset from Kaggle, generates embeddings, and
upserts vectors into Qdrant Cloud. Not part of the live API.

- **Data source:** Kaggle dataset via `kagglehub`
- **Embedding model:** OpenAI `text-embedding-3-large` (3072-dim) — must match `chain/` at query time
- **Vector store:** Qdrant Cloud — always external, never a local container
- **Standalone:** intentionally outside the `uv` workspace (separate lifecycle)

### Key layout

```
src/rag/
├── config.py              Pydantic BaseSettings
├── main.py                Pipeline entry point
├── ingestion/             CSV loader, pipeline orchestration
├── embeddings/            Provider implementations (OpenAI, Gemini)
├── vectorstore/           Qdrant + ChromaDB adapters
└── utils/                 Logger, helpers
scripts/                   Dev utilities (retrieve.py, backup_vectorstore.py)
```

---

## Quality commands (Docker-only)

```bash
make pre-commit   # lint + typecheck + format
make test         # pytest + pytest-cov
make lint         # ruff check
make typecheck    # mypy --strict
make ingest       # run the ingestion pipeline
```

---

## Design patterns

| Pattern      | Where              | Rule                                                              |
| ------------ | ------------------ | ----------------------------------------------------------------- |
| Strategy     | Embedding providers| New provider = new class implementing the provider interface      |
| Strategy     | Vector stores      | `qdrant` and `chromadb` are strategies; add new stores similarly  |
| Config object| `config.py`        | All env vars loaded once — no `os.getenv()` in pipeline functions |
| Factory      | Entry point        | Provider objects created once in `main.py`, not inside pipeline   |

---

## Python standards

- `mypy --strict` must pass; no `type: ignore` without comment
- No bare `except:`
- Docstrings (Google style) on all public classes and functions
- No `print()` in production — use `logging`
- Line length: 100

---

## Environment variables

```
QDRANT_URL, QDRANT_API_KEY_RW, QDRANT_COLLECTION_NAME
OPENAI_API_KEY
KAGGLE_API_TOKEN
```

---

## Embedding model coordination

The embedding model and dimension here **must match** what `chain/` uses at query time.
If you change `text-embedding-3-large` or the dimension, coordinate with `chain/` and update
`.env.example` in both repos. The `ingestion-outputs.env` Jenkins artifact carries the
collection name and dimension for cross-team verification.

---

## Workflow

- Branches: `feature/<kebab>`, `fix/<kebab>`, `chore/<kebab>`
- Commits: `feat(rag): add Gemini embedding provider`
- Pre-commit: `make pre-commit` (Docker)
- After merge: bump pointer in `backend/`, then in root `movie-finder`
