"""Compile-time constants and widget IDs for the Retrieval TUI."""

from __future__ import annotations

from rag.config import EmbeddingProviderName, VectorStoreName

PROVIDERS: list[tuple[str, EmbeddingProviderName]] = [
    ("OpenAI", "openai"),
    ("Ollama", "ollama"),
    ("HuggingFace", "huggingface"),
    ("Sentence Transformers", "sentence-transformers"),
    ("Google", "google"),
]

VECTOR_STORES: list[tuple[str, VectorStoreName]] = [
    ("Qdrant", "qdrant"),
    ("ChromaDB", "chromadb"),
    ("Pinecone", "pinecone"),
    ("PGVector", "pgvector"),
]

TOP_K_OPTIONS: list[tuple[str, int]] = [
    ("3", 3),
    ("5", 5),
    ("10", 10),
    ("15", 15),
    ("20", 20),
]

PROVIDER_SELECT_ID = "provider-select"
MODEL_SELECT_ID = "model-select"
VECTOR_STORE_SELECT_ID = "vector-store-select"
TOP_K_SELECT_ID = "top-k-select"
SEARCH_INPUT_ID = "search-input"
SEARCH_BUTTON_ID = "search-button"
BACKUP_BUTTON_ID = "backup-button"
RESULTS_LIST_ID = "results-list"
STATUS_LOG_ID = "status-log"
