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

## Workflow invariants

- This repo is the gitlink path `rag_ingestion` inside `aharbii/movie-finder-backend`. Parent
  workflow/path filters must use `rag_ingestion`, not `rag_ingestion/**`.
- Cross-repo tracker issues originate in `aharbii/movie-finder`. Create the linked child issue in
  this repo only if this repo will actually change.
- Inspect `.github/ISSUE_TEMPLATE/*.yml`, `.github/PULL_REQUEST_TEMPLATE.md` when present, and a
  recent example before creating or editing issues/PRs. Do not improvise titles or bodies.
- For child issues in this repo, use `.github/ISSUE_TEMPLATE/linked_task.yml` and keep the
  description, file references, and acceptance criteria repo-specific.
- If CI, required checks, or merge policy changes affect this repo, update contributor-facing docs
  here and in `aharbii/movie-finder-backend` and/or `aharbii/movie-finder` where relevant.
- If a new standalone issue appears mid-session, branch from `main` unless stacking is explicitly
  requested.
- PR descriptions must disclose the AI authoring tool + model. Any AI-assisted review comment or
  approval must also disclose the review tool + model.

---

## Cross-cutting — check for every change

1. GitHub issue in `aharbii/movie-finder` + linked child issue here only if this repo changes, using the current templates and recent examples
2. Branch: `feature/`, `fix/`, `chore/` (kebab-case) from `main` unless stacking is explicitly requested
3. ADR if embedding model, vector dimensions, or Qdrant schema changes
4. `.env.example` updated in rag_ingestion + backend + root
5. `Dockerfile` updated if new deps or env vars
6. `backend/chain/` assessed — embedding model change affects search quality
7. PlantUML diagrams updated for pipeline changes
