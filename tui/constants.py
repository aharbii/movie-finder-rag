"""Constants, command registry, and provider/model data for the Retrieval TUI."""

from __future__ import annotations

from typing import NamedTuple

from rag.config import EmbeddingProviderName, VectorStoreName

# ---------------------------------------------------------------------------
# Provider / store / top-k option lists
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Models available per provider: (model_id, description)
# ---------------------------------------------------------------------------

PROVIDER_MODELS: dict[EmbeddingProviderName, list[tuple[str, str]]] = {
    "openai": [
        ("text-embedding-3-large", "3072-dim · recommended"),
        ("text-embedding-3-small", "1536-dim · lighter"),
        ("text-embedding-ada-002", "1536-dim · older OpenAI model"),
    ],
    "ollama": [
        ("nomic-embed-text:latest", "768-dim"),
        ("mxbai-embed-large:latest", "1024-dim"),
        ("all-minilm:latest", "384-dim · fast"),
    ],
    "huggingface": [
        ("sentence-transformers/all-MiniLM-L6-v2", "384-dim"),
        ("sentence-transformers/all-mpnet-base-v2", "768-dim"),
        ("BAAI/bge-large-en-v1.5", "1024-dim"),
    ],
    "sentence-transformers": [
        ("all-MiniLM-L6-v2", "384-dim · fast"),
        ("all-mpnet-base-v2", "768-dim"),
        ("paraphrase-multilingual-MiniLM-L12-v2", "384-dim · multilingual"),
    ],
    "google": [
        ("text-embedding-004", "768-dim · recommended"),
    ],
}

# ---------------------------------------------------------------------------
# Overlay item — shared data type for the command overlay list
# ---------------------------------------------------------------------------


class OverlayItem(NamedTuple):
    """One row in the command overlay list."""

    item_id: str  # unique ID used to identify the selection
    name: str  # left column (command / option name)
    description: str  # right column (dim)
    indicator: str = ""  # ●/○/▶ appended after description


# ---------------------------------------------------------------------------
# Top-level slash commands shown when user types /
# ---------------------------------------------------------------------------

TOP_COMMANDS: list[OverlayItem] = [
    OverlayItem("cmd_provider", "/provider", "Set embedding provider and model", "▶"),
    OverlayItem("cmd_store", "/store", "Set vector store backend", "▶"),
    OverlayItem("cmd_topk", "/topk", "Set number of results per search", "▶"),
    OverlayItem("cmd_backup", "/backup", "Backup the active collection"),
    OverlayItem("cmd_help", "/help", "Show key bindings and commands"),
    OverlayItem("cmd_quit", "/quit", "Exit the application"),
]

# ---------------------------------------------------------------------------
# Widget IDs
# ---------------------------------------------------------------------------

SEARCH_INPUT_ID = "search-input"
SEARCH_BUTTON_ID = "search-button"
BACKUP_BUTTON_ID = "backup-button"
RESULTS_LIST_ID = "results-list"
OVERLAY_ID = "command-overlay"
OVERLAY_LIST_ID = "overlay-list"
STATUS_BAR_ID = "status-bar"
MESSAGE_BAR_ID = "message-bar"
