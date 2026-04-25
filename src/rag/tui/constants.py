"""Compile-time constants, widget IDs, and slash-command help for the Retrieval TUI."""

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
SETTINGS_BAR_ID = "settings-bar"
MESSAGE_BAR_ID = "message-bar"

COMMAND_HELP = """\
[bold]Slash commands[/bold]

  [cyan]/provider[/cyan] <name>   Set embedding provider
                  [dim]openai · ollama · huggingface · sentence-transformers · google[/dim]
  [cyan]/model[/cyan] <name>      Set embedding model (must match provider)
  [cyan]/store[/cyan] <name>      Set vector store backend
                  [dim]qdrant · chromadb · pinecone · pgvector[/dim]
  [cyan]/topk[/cyan] <n>          Set number of results  [dim](3 · 5 · 10 · 15 · 20)[/dim]
  [cyan]/clear[/cyan]             Clear the results panel
  [cyan]/help[/cyan]              Show this message

[dim]Any other text runs a semantic search.[/dim]\
"""
