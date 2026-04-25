"""Tests for rag.tui.constants."""

from __future__ import annotations

from rag.tui.constants import (
    COMMAND_REGISTRY,
    PROVIDER_NAMES,
    PROVIDERS,
    TOP_K_OPTIONS,
    TOP_K_VALUES,
    VECTOR_STORE_NAMES,
    VECTOR_STORES,
)


def test_providers_non_empty() -> None:
    assert len(PROVIDERS) > 0


def test_vector_stores_covers_all_backends() -> None:
    assert {v for _, v in VECTOR_STORES} == {"qdrant", "chromadb", "pinecone", "pgvector"}


def test_top_k_options_include_default_five() -> None:
    assert 5 in {v for _, v in TOP_K_OPTIONS}


def test_provider_names_matches_providers_list() -> None:
    assert {v for _, v in PROVIDERS} == PROVIDER_NAMES


def test_vector_store_names_matches_vector_stores_list() -> None:
    assert {v for _, v in VECTOR_STORES} == VECTOR_STORE_NAMES


def test_top_k_values_matches_top_k_options_list() -> None:
    assert {v for _, v in TOP_K_OPTIONS} == TOP_K_VALUES


def test_providers_values_are_valid_provider_names() -> None:
    from rag.config import EmbeddingProviderName

    valid: set[str] = set(EmbeddingProviderName.__args__)  # type: ignore[attr-defined]
    for _, value in PROVIDERS:
        assert value in valid


def test_command_registry_has_all_providers() -> None:
    registry_names = {name for _, name, _ in COMMAND_REGISTRY}
    for _, provider in PROVIDERS:
        assert f"/provider {provider}" in registry_names


def test_command_registry_has_all_stores() -> None:
    registry_names = {name for _, name, _ in COMMAND_REGISTRY}
    for _, store in VECTOR_STORES:
        assert f"/store {store}" in registry_names


def test_command_registry_has_quit_and_help() -> None:
    registry_names = {name for _, name, _ in COMMAND_REGISTRY}
    assert "/quit" in registry_names
    assert "/help" in registry_names


def test_command_registry_ids_are_unique() -> None:
    ids = [cmd_id for cmd_id, _, _ in COMMAND_REGISTRY]
    assert len(ids) == len(set(ids))
