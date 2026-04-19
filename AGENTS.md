# OpenAI Codex CLI — rag submodule

This is **`movie-finder-rag`** (`rag/`) — part of the Movie Finder project.
GitHub repo: `aharbii/movie-finder-rag` · Parent repo: `aharbii/movie-finder`

> See root AGENTS.md for: full submodule map, GitHub issue/PR hygiene, coding standards, branching strategy, session start protocol.

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

```bash
make init / make up / make test-coverage / make pre-commit / make ingest
```

---

## VS Code setup

`rag/.vscode/` — full workspace configuration for rag only.

- Tasks call `make ...` from this directory
- Interpreter: `/opt/venv/bin/python` inside the attached `rag` container
- `launch.json`: ingestion pipeline + pytest profiles for the attached container
- `tasks.json`: init, up, down, logs, shell, lint, format, typecheck, test, coverage, pre-commit, ingest
- `options.cwd` must point to this directory when called from parent workspaces

---

## Workflow invariants (rag-specific)

- Gitlink path is `rag` inside `aharbii/movie-finder` (direct root submodule, `update = none`). Initialise explicitly: `git submodule update --init rag`.
- Parent path filters must use `rag`, not `rag/**`.
- Embedding model changes here require coordinating with `chain/` — query-time and ingestion-time embeddings must match.

### Submodule pointer bump

```bash
# in root movie-finder
git add rag && git commit -m "chore(rag): bump to latest main"
```
