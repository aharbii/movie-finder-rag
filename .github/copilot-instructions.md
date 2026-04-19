# GitHub Copilot — movie-finder-rag

Offline embedding ingestion pipeline — downloads Kaggle movie data, generates OpenAI embeddings, and upserts into Qdrant Cloud.

> For full project context, persona prompts, and architecture reference: see root `.github/copilot-instructions.md`.

---

## Python standards

- `ruff` rules in scope: E, W, F, I, B, UP (E501 and B008 are ignored)
- Tests must stub OpenAI, Qdrant, and Kaggle interactions — no real API calls in tests
- Every new embedding provider or pipeline step needs test coverage
- Run `make help` for all available targets

**Critical constraint:** The embedding model (`text-embedding-3-large`, 3072-dim) must match the query-time embedding in `chain/`. Changing the model here requires re-ingestion of the entire dataset and coordination with the `chain/` team before merging.

---

## Design patterns

| Pattern                  | Rule                                                                                                                          |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| **Strategy**             | New embedding provider = new class implementing the embedding interface. Never add `if provider == "openai":` in the ingestion loop. |
| **Strategy**             | `qdrant` and `chromadb` are strategies behind a common vector store interface. Add new stores the same way.                   |
| **Factory**              | Provider objects are created in one place (entrypoint/factory function), not scattered through pipeline steps.                |
| **Configuration object** | All env vars loaded once in `config.py` via Pydantic `BaseSettings`. Never call `os.getenv()` inside pipeline functions.     |

---

## Key files

| Path                | Description                                                               |
| ------------------- | ------------------------------------------------------------------------- |
| `config.py`         | Pydantic `BaseSettings` — single source for all env vars                  |
| `Makefile`          | Docker-only dev contract — run `make help` for all targets                |
| `docker-compose.yml` | Local `rag` container (Qdrant is always external — never local)          |
| `Jenkinsfile`       | CI pipeline — triggered manually or on `main`/tags via `RUN_INGESTION`   |
| `.env.example`      | Required vars: `QDRANT_URL`, `QDRANT_API_KEY_RW`, `QDRANT_COLLECTION_NAME`, `OPENAI_API_KEY`, `KAGGLE_API_TOKEN` |
