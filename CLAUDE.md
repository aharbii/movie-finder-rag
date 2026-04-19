# Claude Code — rag submodule

This is **`movie-finder-rag`** (`rag/`) — part of the Movie Finder project.
GitHub repo: `aharbii/movie-finder-rag` · Parent repo: `aharbii/movie-finder`

> See root `CLAUDE.md` for: full submodule map, GitHub issue/PR hygiene, cross-cutting checklist, coding standards, branching strategy, session start protocol.

---

## What this submodule does

Offline RAG ingestion pipeline. Downloads the movie dataset from Kaggle, generates embeddings,
and upserts vectors into Qdrant Cloud. Runs as a one-shot script, not part of the live API.

- **Data source:** Kaggle dataset via `kagglehub`
- **Embedding model:** OpenAI `text-embedding-3-large` (3072-dim) — must match `chain/` query-time embedding
- **Vector store:** Qdrant Cloud — always external, never a local container
- **Local dev contract:** Docker-only via repo-local `make ...` targets
- **Intentionally standalone:** not a `uv` workspace member — separate lifecycle from `backend/`
- **Jenkins trigger:** Manual (`RUN_INGESTION=true` parameter) or on `main`/tags
- **Output artifact:** `ingestion-outputs.env` — archived by Jenkins, used by `chain/` team to verify collection name and dimension

---

## Technology stack (rag-specific)

| Layer         | Stack                                                     |
| ------------- | --------------------------------------------------------- |
| Language      | Python 3.13, standalone `uv` project, Docker-only local   |
| Data          | `kagglehub`, `pandas`, `pydantic`                         |
| Embeddings    | `openai` SDK — `text-embedding-3-large`, 3072-dim         |
| Vector store  | `qdrant-client` (Qdrant Cloud only)                       |
| Tests         | `pytest`, `pytest-cov`                                    |
| CI            | Jenkins Multibranch → Azure Container Registry            |

---

## Environment variables (`.env.example`)

```
QDRANT_URL, QDRANT_API_KEY_RW, QDRANT_COLLECTION_NAME
OPENAI_API_KEY
KAGGLE_API_TOKEN
```

Note: `qdrant-api-key-rw` is used exclusively by this CI pipeline and lives in the Jenkins
credentials store only — never in Azure Key Vault (this is an offline job, not a deployed service).

---

## Design patterns (rag-specific)

| Pattern                  | Where                                 | Rule                                                                                                                       |
| ------------------------ | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Strategy**             | Embedding providers                   | New provider = new class implementing the embedding interface. Never add `if provider == "openai":` in the ingestion loop. |
| **Strategy**             | Vector store backends                 | `qdrant` and `chromadb` are strategies behind a common interface. Add new stores the same way.                             |
| **Configuration object** | `config.py` / Pydantic `BaseSettings` | All env vars loaded once. Never call `os.getenv()` inside pipeline functions.                                              |
| **Factory**              | Provider instantiation                | Provider objects created in one place (entrypoint / factory function), not scattered through pipeline steps.               |

---

## Coding standards (additions to root CLAUDE.md)

- `ruff` rules: E, W, F, I, B, UP (ignore E501, B008)
- Every new embedding provider or pipeline step needs test coverage

---

## Pre-commit hooks

```bash
make pre-commit
```

Hooks: whitespace/YAML/safety checks, `detect-secrets`, `mypy --strict`, `ruff-check --fix`, `ruff-format`. **Never `--no-verify`.**
False positive → `# pragma: allowlist secret` + `detect-secrets scan > .secrets.baseline`.

---

## VSCode setup

- `settings.json` — attached-container interpreter (`/opt/venv/bin/python`), Ruff, mypy strict, pytest
- `launch.json` — ingestion pipeline runner + pytest profiles for the attached `rag` container
- `tasks.json` — `make ...` targets for init/up/down/logs/shell/lint/format/typecheck/test/coverage/pre-commit/ingest

**Workflow:** `make up` from `rag/`, then attach VS Code to the `rag` container using `/opt/venv/bin/python`.

---

## Workflow invariants (rag-specific)

- Gitlink path is `rag` inside `aharbii/movie-finder` (direct root submodule, `update = none`). Initialise explicitly: `git submodule update --init rag`.
- Parent path filters must use `rag`, not `rag/**`.
- Embedding model changes here require coordinating with `chain/` — query-time and ingestion-time embeddings must match.

Run `/session-start` in root workspace.

---

## Branching and commits

```
feature/<kebab>  fix/<kebab>  chore/<kebab>  docs/<kebab>
```

Conventional Commits: `feat(rag): add Gemini embedding provider`

---

## Cross-cutting change checklist (rag-specific rows)

| #   | Category           | Key gate                                                                                                                                                               |
| --- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Branch**         | `feature/fix/chore/docs` in this repo + pointer-bump `chore/` in root `movie-finder`                                                                                   |
| 2   | **ADR**            | New embedding provider, new vector store, or new external dep → ADR in `docs/`                                                                                         |
| 3   | **Env & secrets**  | `.env.example` updated here + `chain/` if embedding model shared + root; new keys → Jenkins credentials store (not Key Vault — this is an offline job)                 |
| 4   | **Docker**         | `Dockerfile` + `docker-compose.yml` updated for dep/env changes                                                                                                        |
| 5   | **CI**             | `Jenkinsfile` reviewed; `ingestion-outputs.env` artifact format still valid for `chain/` team                                                                          |
| 6   | **Diagrams**       | `02-system-architecture.puml` or `03-backend-architecture.puml` if provider changes; `workspace.dsl` if new external system; commit to `docs/` first; **never `.mdj`** |

### Sibling submodules likely affected

| Submodule         | Why                                                                     |
| ----------------- | ----------------------------------------------------------------------- |
| `backend/chain/`  | Embedding model at query time must match ingestion — coordinate changes |
| `infrastructure/` | New Azure AI service or new secret → infra and Key Vault update         |
| `docs/`           | Architecture diagrams, ingestion runbook                                |

### Submodule pointer bump

```bash
# in root movie-finder
git add rag && git commit -m "chore(rag): bump to latest main"
```

### Pull request

- [ ] PR in `aharbii/movie-finder-rag` discloses the AI authoring tool + model
- [ ] PR in `aharbii/movie-finder` (pointer bump)
- [ ] Any AI-assisted review comment or approval discloses the review tool + model
