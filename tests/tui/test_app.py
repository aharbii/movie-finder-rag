"""Tests for rag.tui.app.RetrievalApp."""

from __future__ import annotations

from rag.tui.app import RetrievalApp
from rag.tui.constants import PROVIDERS, VECTOR_STORES


def test_model_options_for_returns_default_model() -> None:
    """_model_options_for returns a single-entry list with the provider's default model."""
    options = RetrievalApp._model_options_for("openai")
    assert len(options) == 1
    label, value = options[0]
    assert value == "text-embedding-3-large"
    assert label == value


def test_model_options_for_google() -> None:
    """_model_options_for resolves the correct default for the Google provider."""
    options = RetrievalApp._model_options_for("google")
    assert options[0][1] == "text-embedding-004"


def test_model_options_for_all_providers() -> None:
    """_model_options_for returns a non-empty list for every configured provider."""
    for _, provider in PROVIDERS:
        options = RetrievalApp._model_options_for(provider)
        assert len(options) >= 1, f"No model options for provider: {provider}"


def test_retrieval_app_default_reactives() -> None:
    """RetrievalApp initialises with sensible reactive defaults."""
    app = RetrievalApp()
    assert app._provider == "openai"
    assert app._vector_store == "qdrant"
    assert app._top_k == 5


def test_vector_stores_constant_accessible_from_app_module() -> None:
    """VECTOR_STORES constant is importable from the app module."""
    from rag.tui import app as tui_app

    assert hasattr(tui_app, "VECTOR_STORES")
    assert tui_app.VECTOR_STORES == VECTOR_STORES
