"""Tests for rag.tui.constants."""

from __future__ import annotations

from rag.tui.constants import PROVIDERS, TOP_K_OPTIONS, VECTOR_STORES


def test_providers_non_empty() -> None:
    """PROVIDERS must contain at least one entry."""
    assert len(PROVIDERS) > 0


def test_vector_stores_covers_all_backends() -> None:
    """VECTOR_STORES covers all four supported backends."""
    store_values = {value for _, value in VECTOR_STORES}
    assert store_values == {"qdrant", "chromadb", "pinecone", "pgvector"}


def test_top_k_options_include_default_five() -> None:
    """TOP_K_OPTIONS must include a 5-result option."""
    values = [value for _, value in TOP_K_OPTIONS]
    assert 5 in values


def test_providers_values_are_valid_provider_names() -> None:
    """Every PROVIDERS value must be a known EmbeddingProviderName."""
    from rag.config import EmbeddingProviderName

    valid: set[str] = set(EmbeddingProviderName.__args__)  # type: ignore[attr-defined]
    for _, value in PROVIDERS:
        assert value in valid
