# Changelog — movie-finder-rag

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

<!-- Add new changes here under the appropriate subsection. -->
<!-- Subsections: Added, Changed, Deprecated, Removed, Fixed, Security -->

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
