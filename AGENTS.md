# OpenAI Codex CLI — rag_ingestion submodule

Foundational mandate for `movie-finder-rag` (`backend/rag_ingestion/`).

---

## What this submodule does
Offline RAG ingestion pipeline. Downloads Kaggle data → Embeddings → Qdrant.
Runs as a standalone `uv` project.

---

## Technology stack
- Python 3.13, `kagglehub`, `pandas`
- OpenAI `text-embedding-3-large`
- Qdrant Cloud

---

## Design patterns
- **Strategy pattern** for embedding providers and vector stores.
- **Configuration:** Pydantic `BaseSettings`.

---

## Common tasks
- `uv sync` to create local `.venv`.
- `uv run pre-commit run --all-files`.
- `pytest --cov`.

---

## VSCode setup

`backend/rag_ingestion/.vscode/` — full workspace configuration for rag_ingestion only.
- Interpreter: `rag_ingestion/.venv/bin/python` (standalone — run `uv sync` from this directory)
- `launch.json`: ingestion pipeline runner + pytest all/current file
- `tasks.json`: lint, test, pre-commit, dry run
- This is a **standalone uv project** (not a workspace member). Interpreter path differs from other backend packages.
- Modifying configs: `options.cwd` must point to this directory when called from parent workspaces.
  Update `CLAUDE.md`, `GEMINI.md`, and `AGENTS.md` after.
