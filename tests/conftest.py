import os

# Keep pytest isolated from developer and CI runtime configuration.
# Pydantic reads process env before env files, so these must override values
# injected by Docker Compose from a local `.env`.
os.environ["RAG_ENV_FILE"] = ".env.pytest-do-not-load"
os.environ["VECTOR_STORE"] = "qdrant"
os.environ["QDRANT_URL"] = "https://qdrant.test"
os.environ["QDRANT_API_KEY_RW"] = "test-qdrant-key"
os.environ["EMBEDDING_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "test-openai-key"
