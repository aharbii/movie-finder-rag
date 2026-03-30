# GitHub Copilot — movie-finder-rag

Offline embedding ingestion pipeline for Movie Finder. Reads Kaggle data, generates OpenAI
embeddings, and upserts vectors into Qdrant Cloud. This repo uses a Docker-only local-development
contract and is not part of the live request path.

Parent project: `aharbii/movie-finder` — all cross-repo tracker issues start there.

---

## Package role

- Downloads the Wikipedia Movie Plots dataset via `kagglehub`
- Generates `text-embedding-3-large` vectors
- Writes to Qdrant Cloud with the write-capable API key used only by this repo
- Exposes repo-local `make ...` targets backed by Docker Compose

Qdrant is always external. Do not reintroduce a local Qdrant container or `localhost` endpoint.

---

## Python standards

- Python 3.13
- Docker-only local workflow; use the attached `rag` container for editor/debug flows
- `ruff` + `mypy --strict`, line length **100**
- Type annotations required on public functions
- Tests must stub OpenAI, Qdrant, and Kaggle interactions

---

## Design patterns — follow these

| Pattern | Rule |
|---|---|
| **Strategy** | New embedding provider = new class implementing the provider interface. No `if provider == "openai"` in pipeline logic. |
| **Configuration object** | All env vars loaded once in `config.py` via Pydantic `BaseSettings`. |

---

## Developer workflow

```bash
make init
make up
make lint
make typecheck
make test
make test-coverage
make pre-commit
make ingest
```

Canonical env vars:

- `QDRANT_URL`
- `QDRANT_API_KEY_RW`
- `QDRANT_COLLECTION_NAME`
- `OPENAI_API_KEY`
- `KAGGLE_API_TOKEN`

---

## Known issues most relevant to this package

| # | Title |
|---|---|
| #2 | Shared production Qdrant cluster across all environments |
| #13 | Standardize Docker-only local development workflow and repo tooling |
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
5. `Dockerfile`, `docker-compose.yml`, and `Makefile` updated if the local contract changes
6. `backend/chain/` assessed — embedding model or collection changes affect search quality
7. PlantUML diagrams updated for pipeline changes
