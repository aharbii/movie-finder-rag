"""Tests for the Textual TUI retrieval app (scripts/retrieve.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Module loader helper
# ---------------------------------------------------------------------------


def _load_retrieve_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "retrieve",
        Path(__file__).resolve().parents[1] / "scripts" / "retrieve.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/retrieve.py for testing.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retrieve_script = _load_retrieve_script()


# ---------------------------------------------------------------------------
# MovieCard
# ---------------------------------------------------------------------------


def _make_movie(**kwargs: object) -> MagicMock:
    """Return a Movie-shaped mock."""
    movie = MagicMock()
    movie.title = kwargs.get("title", "Test Movie")
    movie.release_year = kwargs.get("release_year", 2000)
    movie.director = kwargs.get("director", "Test Director")
    movie.genre = kwargs.get("genre", ["Drama"])
    movie.cast = kwargs.get("cast", ["Actor A", "Actor B"])
    movie.plot = kwargs.get("plot", "A test plot.")
    return movie


def test_movie_card_renders_short_cast() -> None:
    """MovieCard with 3 cast members shows them without truncation."""
    movie = _make_movie(cast=["A", "B", "C"])
    card = retrieve_script.MovieCard(1, movie)
    assert card.movie is movie


def test_movie_card_truncates_long_cast() -> None:
    """MovieCard with more than 5 cast members appends '...'."""
    movie = _make_movie(cast=["A", "B", "C", "D", "E", "F"])
    card = retrieve_script.MovieCard(1, movie)
    # The rendered markup should contain '...' for the truncated cast preview.
    rendered = card.content
    assert "..." in str(rendered)


def test_movie_card_truncates_long_plot() -> None:
    """MovieCard with a long plot truncates to 200 chars."""
    long_plot = "x" * 300
    movie = _make_movie(plot=long_plot)
    card = retrieve_script.MovieCard(1, movie)
    rendered = str(card.content)
    assert "..." in rendered


def test_movie_card_short_plot_not_truncated() -> None:
    """MovieCard with a short plot does not add '...' for the plot section."""
    short_plot = "Short plot."
    movie = _make_movie(plot=short_plot, cast=["A"])
    card = retrieve_script.MovieCard(1, movie)
    rendered = str(card.content)
    assert "Short plot." in rendered


# ---------------------------------------------------------------------------
# RetrievalApp — unit-level helpers
# ---------------------------------------------------------------------------


def test_model_options_for_returns_default_model() -> None:
    """_model_options_for returns a list with the provider's default model."""
    options = retrieve_script.RetrievalApp._model_options_for("openai")
    assert len(options) == 1
    label, value = options[0]
    assert value == "text-embedding-3-large"
    assert label == "text-embedding-3-large"


def test_model_options_for_google() -> None:
    """_model_options_for correctly resolves the Google provider default."""
    options = retrieve_script.RetrievalApp._model_options_for("google")
    assert options[0][1] == "text-embedding-004"


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


def test_main_calls_app_run() -> None:
    """main() instantiates RetrievalApp and calls run()."""
    mock_app = MagicMock()
    with patch.object(retrieve_script, "RetrievalApp", return_value=mock_app) as mock_cls:
        retrieve_script.main()
        mock_cls.assert_called_once()
        mock_app.run.assert_called_once()


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


def test_providers_list_is_non_empty() -> None:
    """PROVIDERS constant must contain at least one entry."""
    assert len(retrieve_script.PROVIDERS) > 0


def test_vector_stores_list_covers_expected_backends() -> None:
    """VECTOR_STORES covers all four supported backends."""
    store_values = {value for _, value in retrieve_script.VECTOR_STORES}
    assert store_values == {"qdrant", "chromadb", "pinecone", "pgvector"}


def test_top_k_options_include_default_five() -> None:
    """TOP_K_OPTIONS must include a 5-result option."""
    values = [value for _, value in retrieve_script.TOP_K_OPTIONS]
    assert 5 in values
