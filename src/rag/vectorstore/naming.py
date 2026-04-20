import re

from rag.embeddings.base import EmbeddingModelMetadata

_SANITIZE_PATTERN = re.compile(r"[\/.\-\s]+")
_INVALID_PATTERN = re.compile(r"[^a-z0-9_]+")
_MULTI_UNDERSCORE_PATTERN = re.compile(r"_+")


def sanitize_collection_token(value: str) -> str:
    """Return the ADR 0008-compliant token for model and collection naming."""
    normalized = _SANITIZE_PATTERN.sub("_", value.strip().lower())
    normalized = _INVALID_PATTERN.sub("_", normalized)
    normalized = _MULTI_UNDERSCORE_PATTERN.sub("_", normalized)
    return normalized.strip("_")


def resolve_collection_name(prefix: str, embedding_model: EmbeddingModelMetadata) -> str:
    """Resolve the zero-collision collection name shared with the chain repo."""
    sanitized_prefix = sanitize_collection_token(prefix)
    sanitized_model = sanitize_collection_token(embedding_model.name)
    return f"{sanitized_prefix}_{sanitized_model}_{embedding_model.dimension}"
