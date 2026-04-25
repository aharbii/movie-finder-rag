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

PROVIDER_NAMES: set[str] = {v for _, v in PROVIDERS}
VECTOR_STORE_NAMES: set[str] = {v for _, v in VECTOR_STORES}
TOP_K_VALUES: set[int] = {v for _, v in TOP_K_OPTIONS}

# Widget IDs
SEARCH_INPUT_ID = "search-input"
SEARCH_BUTTON_ID = "search-button"
BACKUP_BUTTON_ID = "backup-button"
RESULTS_LIST_ID = "results-list"
COMPLETION_LIST_ID = "completion-list"
CONFIG_BAR_ID = "config-bar"
MESSAGE_BAR_ID = "message-bar"

# Registry: (command_id, display_name, description)
# command_id format: "type:value"
COMMAND_REGISTRY: list[tuple[str, str, str]] = [
    # --- providers ---
    *[(f"provider:{v}", f"/provider {v}", f"Set embedding provider → {v}") for _, v in PROVIDERS],
    # --- vector stores ---
    *[
        (f"store:{v}", f"/store {v}", f"Use {label} as the vector store backend")
        for label, v in VECTOR_STORES
    ],
    # --- top-k ---
    *[(f"topk:{v}", f"/topk {v}", f"Return {v} results per search") for _, v in TOP_K_OPTIONS],
    # --- actions ---
    ("action:clear", "/clear", "Clear the results panel"),
    ("action:help", "/help", "Show command reference"),
    ("action:quit", "/quit", "Exit the app"),
]
