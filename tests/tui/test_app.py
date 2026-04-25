"""Tests for rag.tui.app.RetrievalApp."""

from __future__ import annotations

from rag.tui.app import RetrievalApp
from rag.tui.constants import PROVIDERS, VECTOR_STORES


def test_model_options_for_openai() -> None:
    options = RetrievalApp._model_options_for("openai")
    assert len(options) == 1
    label, value = options[0]
    assert value == "text-embedding-3-large"
    assert label == value


def test_model_options_for_google() -> None:
    assert RetrievalApp._model_options_for("google")[0][1] == "text-embedding-004"


def test_model_options_for_all_providers() -> None:
    for _, provider in PROVIDERS:
        assert len(RetrievalApp._model_options_for(provider)) >= 1


def test_retrieval_app_default_reactives() -> None:
    app = RetrievalApp()
    assert app._provider == "openai"
    assert app._vector_store == "qdrant"
    assert app._top_k == 5


def test_vector_stores_constant_accessible_from_app_module() -> None:
    from rag.tui import app as tui_app

    assert hasattr(tui_app, "VECTOR_STORES")
    assert tui_app.VECTOR_STORES == VECTOR_STORES
