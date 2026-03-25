# GitHub Copilot — movie-finder-rag

Offline embedding ingestion pipeline for Movie Finder. Reads raw movie data, generates
OpenAI embeddings, and upserts vectors into Qdrant Cloud. Runs as a standalone script —
not part of the live request path.

Parent project: `aharbii/movie-finder` — all issues created there first, then linked here.

---

## Package role

- Reads movie dataset (CSV / JSON source)
- Chunks and preprocesses text
- Calls OpenAI `text-embedding-3-large` (3072-dim) to generate embeddings
- Upserts vectors + payload into a Qdrant Cloud collection
- **Standalone `uv` project** — has its own `.venv` (not a workspace member)

Qdrant is always external (Qdrant Cloud) — no local Qdrant container ever.

---

## Python standards

- Python 3.13, standalone `uv` project (`rag_ingestion/.venv`), `ruff` + `mypy --strict`, line length **100**
- Type annotations required on all public functions
- Async all the way — no blocking I/O in async context
- Docstrings on all public classes and functions (Google style)
- Tests: `pytest` with `pytest-mock`. No real OpenAI or Qdrant calls — mock at HTTP boundary.

---

## Design patterns — follow these

| Pattern | Rule |
|---|---|
| **Strategy** | New embedding provider = new class implementing the provider interface. No `if provider == "openai"` in pipeline logic. |
| **Configuration object** | All env vars loaded once in `config.py` via Pydantic `BaseSettings`. |

---

## Pre-commit hooks

```bash
# From rag_ingestion/ directory (standalone uv project)
uv run pre-commit install
uv run pre-commit run --all-files
```

Hooks: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-merge-conflict`,
`detect-private-key`, `detect-secrets`, `pretty-format-json`, `sort-simple-yaml`,
`mypy --strict`, `ruff-check --fix`, `ruff-format`.

---

## Known issues most relevant to this package

| # | Title |
|---|---|
| #14 | Shared production Qdrant cluster across all environments |
| #19 | No batch embedding — single calls per document |

---

## Cross-cutting — check for every change

1. GitHub issue in `aharbii/movie-finder` + this repo (linked)
2. Branch: `feature/`, `fix/`, `chore/` (kebab-case)
3. ADR if embedding model, vector dimensions, or Qdrant schema changes
4. `.env.example` updated in rag_ingestion + backend + root
5. `Dockerfile` updated if new deps or env vars
6. `backend/chain/` assessed — embedding model change affects search quality
7. PlantUML diagrams updated for pipeline changes
