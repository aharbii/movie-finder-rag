# Gemini CLI — rag_ingestion submodule

Foundational mandate for `movie-finder-rag` (`backend/rag_ingestion/`).

---

## What this submodule does

Offline RAG ingestion pipeline. Downloads Kaggle data, generates embeddings, and writes to
Qdrant Cloud. Local development follows the Docker-only repo contract.

---

## Technology stack

- Python 3.13, `kagglehub`, `pandas`
- OpenAI `text-embedding-3-large`
- Qdrant Cloud
- Docker Compose + Make for local workflows

---

## Design patterns

- **Strategy pattern** for embedding providers and vector stores.
- **Configuration:** Pydantic `BaseSettings`.

---

## Common tasks

- `make init`
- `make up`
- `make test-coverage`
- `make pre-commit`
- `make ingest`

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

## VSCode setup

`backend/rag_ingestion/.vscode/` — full workspace configuration for rag_ingestion only.

- Tasks call `make ...` from this directory
- Interpreter: `/opt/venv/bin/python` inside the attached `rag` container
- `launch.json`: ingestion pipeline + pytest profiles for the attached container
- `tasks.json`: init, up, down, logs, shell, lint, format, typecheck, test, coverage, pre-commit,
  ingest
- Modifying configs: `options.cwd` must point to this directory when called from parent
  workspaces. Update `CLAUDE.md`, `GEMINI.md`, `AGENTS.md`, and the repo's
  `.github/copilot-instructions.md` after.
