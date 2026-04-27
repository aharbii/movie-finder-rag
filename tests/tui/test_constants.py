"""Tests for tui.constants."""

from __future__ import annotations

from tui.constants import (
    PROVIDER_MODELS,
    PROVIDER_NAMES,
    PROVIDERS,
    TOP_COMMANDS,
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


def test_provider_models_covers_all_providers() -> None:
    for _, provider in PROVIDERS:
        assert provider in PROVIDER_MODELS
        assert len(PROVIDER_MODELS[provider]) >= 1


def test_top_commands_has_all_expected_commands() -> None:
    ids = {item.item_id for item in TOP_COMMANDS}
    assert "cmd_provider" in ids
    assert "cmd_store" in ids
    assert "cmd_topk" in ids
    assert "cmd_backup" in ids
    assert "cmd_help" in ids
    assert "cmd_quit" in ids


def test_top_commands_ids_are_unique() -> None:
    ids = [item.item_id for item in TOP_COMMANDS]
    assert len(ids) == len(set(ids))
