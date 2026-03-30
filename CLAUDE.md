# Claude Code — rag_ingestion submodule

This is **`movie-finder-rag`** (`backend/rag_ingestion/`) — part of the Movie Finder project.
GitHub repo: `aharbii/movie-finder-rag` · Parent repo: `aharbii/movie-finder`

---

## What this submodule does

Offline RAG ingestion pipeline. Downloads the movie dataset from Kaggle, generates embeddings,
and upserts vectors into Qdrant Cloud. Runs as a one-shot script, not part of the live API.

- **Data source:** Kaggle dataset via `kagglehub`
- **Embedding model:** OpenAI `text-embedding-3-large` (3072-dim)
- **Vector store:** Qdrant Cloud — always external, never a local container
- **Intentionally outside** the `uv` workspace (separate lifecycle from `backend/`)
- **Jenkins trigger:** Manual (`RUN_INGESTION=true` parameter) or on `main`/tags
- **Output artifact:** `ingestion-outputs.env` — archived by Jenkins, used by `chain/` team to verify collection name and dimension

---

## Full project context

### Submodule map

| Path | GitHub repo | Role |
|---|---|---|
| `.` (root) | `aharbii/movie-finder` | Parent — all cross-repo issues |
| `backend/` | `aharbii/movie-finder-backend` | FastAPI + uv workspace root |
| `backend/app/` | (nested in backend) | FastAPI application layer |
| `backend/chain/` | `aharbii/movie-finder-chain` | LangGraph 8-node AI pipeline |
| `backend/imdbapi/` | `aharbii/imdbapi-client` | Async IMDb REST client |
| `backend/rag_ingestion/` | `aharbii/movie-finder-rag` | **← you are here** |
| `frontend/` | `aharbii/movie-finder-frontend` | Angular 21 SPA |
| `docs/` | `aharbii/movie-finder-docs` | MkDocs documentation |
| `infrastructure/` | `aharbii/movie-finder-infrastructure` | IaC / Azure provisioning |

### Technology stack

| Layer | Stack |
|---|---|
| Language | Python 3.13 (standalone `uv` project, not workspace member) |
| Data | `kagglehub`, `pandas`, `pydantic` |
| Embeddings | `openai` SDK — `text-embedding-3-large`, 3072-dim |
| Vector store | `qdrant-client` (Qdrant Cloud only) |
| Linting | `ruff` (line-length 100, rules: E/W/F/I/B/UP) |
| Type checking | `mypy --strict` (Python 3.13) |
| Tests | `pytest`, `pytest-cov` |
| CI | Jenkins Multibranch → Azure Container Registry |

### Environment variables (`.env.example`)

```
QDRANT_ENDPOINT, QDRANT_API_KEY, QDRANT_COLLECTION
OPENAI_API_KEY
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072
KAGGLE_USERNAME, KAGGLE_KEY
VECTOR_STORE=qdrant            # or chromadb (experimental)
```

---

## Design patterns to follow

| Pattern | Where | Rule |
|---|---|---|
| **Strategy** | Embedding providers | New provider = new class implementing the embedding interface. Never add `if provider == "openai":` in the ingestion loop. |
| **Strategy** | Vector store backends | `qdrant` and `chromadb` are strategies behind a common interface. Add new stores the same way. |
| **Configuration object** | `config.py` / Pydantic `BaseSettings` | All env vars loaded once. Never call `os.getenv()` inside pipeline functions. |
| **Factory** | Provider instantiation | Provider objects are created in one place (entrypoint / factory function), not scattered throughout pipeline steps. |

---

## Coding standards

- `mypy --strict` must pass — no `type: ignore` without an explanatory comment
- No bare `except:` — catch specific exceptions
- Docstrings on all public classes and functions (Google style)
- No `print()` in production code — use `logging`
- Line length: 100 (`ruff`)
- `ruff` rules: E, W, F, I, B, UP (ignore E501, B008)
- Tests are not optional — every new provider or pipeline step needs coverage

---

## Pre-commit hooks

`backend/rag_ingestion/.pre-commit-config.yaml` — install and run from this directory.

```bash
uv run pre-commit install    # once per clone
uv run pre-commit run --all-files
```

| Hook | Notes |
|---|---|
| `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-case-conflict`, `check-merge-conflict` | File health |
| `check-added-large-files`, `check-illegal-windows-names`, `detect-private-key` | Safety |
| `pretty-format-json` | JSON files auto-formatted |
| `sort-simple-yaml` | YAML keys sorted |
| `detect-secrets` | No API keys or tokens |
| `mypy` (strict, Python 3.13, extra dep: `pydantic`) | Type checking |
| `ruff-check --fix`, `ruff-format` | Linting and formatting |

**Never `--no-verify`.** False-positive → `# pragma: allowlist secret` + `detect-secrets scan > .secrets.baseline`.

---

## VSCode setup

`backend/rag_ingestion/.vscode/` is committed with a full workspace configuration:
- `settings.json` — Python interpreter (`rag_ingestion/.venv`, standalone project), Ruff, mypy strict, pytest
- `extensions.json` — Python, debugpy, Ruff, mypy, TOML, GitLens
- `launch.json` — ingestion pipeline runner + pytest all / current file
- `tasks.json` — lint, format, test, test with coverage, pre-commit, ingestion dry run

**Interpreter:** Run `uv sync` from `backend/rag_ingestion/` — creates its own `.venv/` (standalone)

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

## Session start protocol

1. `gh issue list --repo aharbii/movie-finder --state open` — check existing issues
2. Inspect `.github/ISSUE_TEMPLATE/*.yml`, `.github/PULL_REQUEST_TEMPLATE.md` when present, and a
   recent example of the same type
3. Create the parent issue in `aharbii/movie-finder`, then the linked child issue in
   `aharbii/movie-finder-rag` only if this repo will actually change
4. Create a branch from `main`: `feature/`, `fix/`, `chore/`, or `docs/` (kebab-case)
5. Work through the cross-cutting checklist below

---

## Branching and commits

```
feature/<kebab>  fix/<kebab>  chore/<kebab>  docs/<kebab>
```

Conventional Commits: `feat(rag): add Gemini embedding provider`

---

## Cross-cutting change checklist

Full detail in `ai-context/issue-agent-briefing-template.md`.

| # | Category | Key gate |
|---|---|---|
| 1 | **Issues** | Parent `aharbii/movie-finder` + child here only if this repo changes; templates inspected |
| 2 | **Branch** | `feature/fix/chore/docs` in this repo + pointer-bump `chore/` in `backend/` and root |
| 3 | **ADR** | New embedding provider, new vector store, or new external dep → ADR in `docs/` |
| 4 | **Implementation** | Strategy pattern for new providers; `ruff`+`mypy --strict` pass; pre-commit pass |
| 5 | **Tests** | `pytest --cov` passes; coverage doesn't regress |
| 6 | **Env & secrets** | `.env.example` updated here + `backend/` + `chain/` if embedding model shared + root; new keys → Key Vault + Jenkins |
| 7 | **Docker** | `Dockerfile` + `docker-compose.yml` updated for dep/env changes |
| 8 | **CI** | `Jenkinsfile` reviewed; `ingestion-outputs.env` artifact format still valid |
| 9 | **Diagrams** | `02-system-architecture.puml` or `03-backend-architecture.puml` if provider changes; `workspace.dsl` if new external system; commit to `docs/` first; **never `.mdj`** |
| 9a | **Docs** | `docs/` pages (ingestion guide, embedding config); `README.md` + `CHANGELOG.md` updated |

### 10. Sibling submodules likely affected
| Submodule | Why |
|---|---|
| `backend/chain/` | Embedding model at query time must match ingestion — coordinate changes |
| `backend/` | New env vars may need to pass through the backend workspace |
| `infrastructure/` | New Azure AI service or new secret → infra and Key Vault update |
| `docs/` | Architecture diagrams, ingestion runbook |

### 11. Submodule pointer bump
```bash
# in backend/ repo
git add rag_ingestion && git commit -m "chore(rag): bump to latest main"
# in root movie-finder
git add backend && git commit -m "chore(backend): bump to latest main"
```

### 12. Pull request
- [ ] PR in `aharbii/movie-finder-rag` discloses the AI authoring tool + model
- [ ] PR in `aharbii/movie-finder-backend` (pointer bump)
- [ ] PR in `aharbii/movie-finder` (pointer bump)
- [ ] Any AI-assisted review comment or approval discloses the review tool + model
