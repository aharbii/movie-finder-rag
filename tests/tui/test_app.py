"""Tests for tui.app.RetrievalApp."""

from __future__ import annotations

from tui.app import RetrievalApp
from tui.constants import PROVIDERS, VECTOR_STORES


def test_model_options_for_openai() -> None:
    options = RetrievalApp._model_options_for("openai")
    assert len(options) == 3
    assert options[0][1] == "text-embedding-3-large"


def test_model_options_for_google() -> None:
    assert RetrievalApp._model_options_for("google")[0][1] == "text-embedding-004"


def test_model_options_for_all_providers() -> None:
    for _, provider in PROVIDERS:
        assert len(RetrievalApp._model_options_for(provider)) >= 1


def test_retrieval_app_default_reactives() -> None:
    import asyncio

    async def _check() -> None:
        app = RetrievalApp()
        assert app._provider == "openai"
        assert app._vector_store == "qdrant"
        assert app._top_k == 5

    asyncio.run(_check())


def test_vector_stores_constant_accessible_from_app_module() -> None:
    from tui import app as tui_app

    assert hasattr(tui_app, "VECTOR_STORES")
    assert tui_app.VECTOR_STORES == VECTOR_STORES
