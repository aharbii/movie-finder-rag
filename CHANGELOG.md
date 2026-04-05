# Changelog — movie-finder-rag

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added

- `Makefile` — Docker-only repo contract for `init`, `up`, `down`, `logs`, `shell`, `lint`,
  `format`, `typecheck`, `test`, `test-coverage`, `pre-commit`, and `ingest`
- Comprehensive test suite for `movie-finder-rag` module, increasing coverage from 31% to 96%:
    - `tests/test_config.py`: Configuration and embedding dimension logic tests.
    - `tests/test_dataset.py`: Dataset download and filesystem operation tests.
    - `tests/test_embeddings.py`: OpenAI and Gemini provider tests with usage tracking.
    - `tests/test_ingestion.py`: CSV loading and ingestion pipeline orchestration tests.
    - `tests/test_main.py`: Entry point and provider resolution tests.
    - `tests/test_vectorstore.py`: ChromaDB and Qdrant vector store operation tests.

### Changed

- `Dockerfile`, `docker-compose.yml`, `.env.example`, `Jenkinsfile`, and `.vscode/*` now align
  with the shared Docker-only local-development contract from `movie-finder#35`
- Qdrant configuration now uses the canonical `QDRANT_URL`, `QDRANT_API_KEY_RW`, and
  `QDRANT_COLLECTION_NAME` contract, with temporary legacy fallback in code
- Tests now use stubbed Qdrant clients instead of real external API calls so CI can run without
  live Qdrant/OpenAI credentials
- `src/rag/ingestion/csv_loader.py`: Improved parsing of `genre` and `cast` fields into lists.
- All test outputs (`junit.xml`, `coverage.xml`, `htmlcov/`) now written to a `reports/`
  subdirectory; `Makefile` paths updated accordingly; `.gitignore` updated to a single
  `reports/` entry
- GitHub Actions CI workflow updated: added `EnricoMi/publish-unit-test-result-action@v2`,
  `irongut/CodeCoverageSummary@v1.3.0`, and `marocchino/sticky-pull-request-comment@v2`
  reporting plugins mirroring Jenkins plugin behaviour; removed Build App Image stage

### Fixed

- Bug in `csv_loader.py` where `genre` and `cast` were being passed as strings instead of lists to the `Movie` model.

### Removed

- Local Qdrant compose workflow and contributor-facing docs that depended on host `uv`

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
