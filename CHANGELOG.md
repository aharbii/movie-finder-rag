# Changelog — movie-finder-rag

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

**Textual TUI — retrieval evaluation app (issue #23)**

- `tui/` — new top-level package containing the full Textual TUI for retrieval evaluation:
  - `tui/app.py` — `RetrievalApp`: slash-command driven, non-blocking search via `@work(thread=True)`,
    inline cosine-similarity scores sourced directly from `store.client.query_points()`, connection
    health indicators for each provider and store, automatic ADR-0008 collection name resolution.
  - `tui/widgets.py` — `MovieCard` (result card with title, year, id, director, genre, cast, plot
    preview, and optional `sim` score), `StatusBar` (always-visible active config line),
    `CommandOverlay` (keyboard-navigable slash-command dropdown).
  - `tui/constants.py` — provider / model / store / top-k option lists, slash-command registry,
    shared `OverlayItem` type, and widget ID constants.
- `scripts/launch_tui.py` — thin entry point that launches `RetrievalApp`; called by `make tui`.
- `tests/tui/test_widgets.py` — unit tests for `MovieCard` (score display, cast/plot truncation,
  id visibility, movie property accessor).
- `Makefile` — added `tui` target (`python scripts/launch_tui.py`).
- `.vscode/launch.json` — added `TUI: launch Retrieval TUI (rag container)` debugpy configuration.
- `.vscode/tasks.json` — added `rag: tui` task.
- `pyproject.toml` — added `mypy_path = "src:scripts:."`, `pythonpath = ["src", "."]`,
  and a `[[tool.mypy.overrides]]` entry to exempt `tui.*` from strict mypy (Textual framework).

**Ingestion cost reporting (issue #6)**

- `scripts/generate_cost_report.py` — reads `ingestion-outputs.env` and writes
  `outputs/reports/cost-report.json` with provider, model, dimension, token counts, and estimated
  USD cost. Called by `make cost-report` and by both Jenkins and GitHub Actions after ingestion.
- `scripts/backup_vectorstore.py` — added `backup_format` field to `BackupConfig` (default
  `"chromadb"`); renamed the Qdrant-specific executor to `backup_chromadb_artifact`; added a
  `backup_vector_store` dispatcher so the Makefile target and CI remain backend-agnostic.
- `Jenkinsfile` — added `BACKUP_FORMAT` parameter; `make cost-report` called in Ingest stage;
  `make qdrant-live-eval` called in Post-Ingest Validate stage (gated on `VECTOR_STORE=qdrant`);
  `VECTOR_COLLECTION_PREFIX` defaults to `movies_<git sha8>` to prevent cross-run collection
  collisions; `BACKUP_FORMAT` wired through `configureRuntimeEnv()`.
- `docs/devops-setup.md` — added `BACKUP_FORMAT` parameter row and expanded Archived Artifacts
  section to list all outputs: `ingestion-outputs.env`, `cost-report.json`, `skipped-movies.json`,
  `validation-report.json`, `qdrant-live-eval/**`, `outputs/backups/**`.

**Qdrant live retrieval evaluation (issue #7)**

- `scripts/evaluate_qdrant_collections.py` — post-ingest evaluation harness that runs a fixed
  query set against configured Qdrant collections, computes hit@k and mean cosine similarity per
  collection, and writes per-collection HTML reports and a `summary.json` under
  `outputs/reports/qdrant-live-eval/`. Supports `--collection-name`, `--provider`, `--model`
  CLI flags for targeted evaluation.

**CI — artifact contract alignment (issues #6, #7)**

- `.github/workflows/ci.yml` — `ingest` job now calls `make cost-report` after `make ingest`;
  `validate` job calls `make qdrant-live-eval` when `vector_store=qdrant`; `backup` job uploads
  `outputs/backups/`, `outputs/reports/`, and `ingestion-outputs.env` as a consolidated
  `vector-store-backup` artifact; `ingest` job uploads `ingestion-outputs.env`,
  `cost-report.json`, and `skipped-movies.json` as `ingestion-outputs`.

**Agent tooling**

- `.mcp.json` — rag-workspace MCP server config (github, qdrant-evaluator, kaggle, jenkins-local).
- `.gemini/settings.json` — Gemini CLI MCP server configuration for the rag workspace.
- `.codex/config.toml` — updated: jenkins-local server added, github tools set to approval mode,
  qdrant-evaluator cwd corrected to `../mcp/qdrant-explorer`.

### Changed

- `scripts/retrieve.py` — restored as a simple interactive CLI for ad-hoc retrieval smoke tests;
  no longer the TUI entry point (that moved to `tui/` and `scripts/launch_tui.py`).
- `tui/constants.py` — Ollama model list updated to use `:latest`-suffixed identifiers
  (`nomic-embed-text:latest`, `mxbai-embed-large:latest`, `all-minilm:latest`) matching the
  actual collection names ingested under ADR 0008.

### Removed

- `Makefile` — removed `migrate-legacy-qdrant-collection`; collection migration is no longer
  needed under the ADR 0008 provider/vector naming contract.

### Fixed

- `tui/app.py` — `_resolve_collection_name` now calls `infer_embedding_dimension` and
  `resolve_collection_name` from `rag.vectorstore.naming` directly, matching the exact naming
  path used during ingestion; previously it attempted to instantiate a full provider + store
  object and could silently produce wrong names if the provider SDK was absent.
- `tui/app.py` — Ollama connection probe now reads `OLLAMA_BASE_URL` and
  attaches `Authorization: Bearer <OLLAMA_API_KEY>` when the env var is set.
- `.vscode/launch.json` — TUI launch configuration corrected to reference
  `scripts/launch_tui.py` (was `scripts/tui.py`).
- `.github/workflows/ci.yml` — `ingest` job now calls `make cost-report` so
  `outputs/reports/cost-report.json` is actually present when the artifact is uploaded.

---

## [0.1.0] — 2026-03-22

### Added

- Dataset download via `kagglehub` — Wikipedia Movie Plots (Kaggle dataset)
- `EmbeddingProvider` ABC with two implementations:
    - `OpenAIEmbeddingProvider` — `text-embedding-3-large` (3072 dimensions)
    - `GeminiEmbeddingProvider` — Google Gemini embeddings
- `VectorStore` ABC with two implementations:
    - `QdrantVectorStore` — Qdrant Cloud (production)
    - `ChromaDBVectorStore` — local ChromaDB (development / backup)
- `Movie` Pydantic model for typed CSV row representation
- CSV loader (`csv_loader.py`) — pandas-based, handles deduplication
- Ingestion pipeline (`pipeline.py`) — orchestrates load → embed → upsert
- Entry point `src/main.py` — runs the full pipeline end-to-end
- `scripts/backup_vectorstore.py` — Qdrant → ChromaDB migration utility
- Docker multi-stage batch job image (`ENTRYPOINT python -m src.main`)
- `docker-compose.yml` — local Qdrant + manual ingestion service (profile: `run`)
- `Jenkinsfile` — lint → test → build → manual `Ingest` stage with parameterized input
  (`RUN_INGESTION`, `COLLECTION_NAME`, `VECTOR_STORE`, `FORCE_DOWNLOAD`)
- Post-ingest Jenkins artifact: `ingestion-outputs.env` for chain team handoff
- Structured logging via `get_logger` factory
