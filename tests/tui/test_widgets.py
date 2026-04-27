"""Tests for tui.widgets.MovieCard."""

from __future__ import annotations

from unittest.mock import MagicMock

from tui.widgets import MovieCard


def _make_movie(**kwargs: object) -> MagicMock:
    movie = MagicMock()
    movie.id = kwargs.get("id", 0)
    movie.title = kwargs.get("title", "Test Movie")
    movie.release_year = kwargs.get("release_year", 2000)
    movie.director = kwargs.get("director", "Test Director")
    movie.genre = kwargs.get("genre", ["Drama"])
    movie.cast = kwargs.get("cast", ["Actor A", "Actor B"])
    movie.plot = kwargs.get("plot", "A test plot.")
    return movie


def test_movie_card_exposes_movie_property() -> None:
    """MovieCard.movie returns the underlying Movie instance."""
    movie = _make_movie()
    card = MovieCard(1, movie)
    assert card.movie is movie


def test_movie_card_short_cast_not_truncated() -> None:
    """Cast preview with ≤5 members contains no trailing ellipsis."""
    card = MovieCard(1, _make_movie(cast=["A", "B", "C"]))
    assert "…" not in str(card.card_content)


def test_movie_card_long_cast_truncated() -> None:
    """Cast preview with >5 members appends the ellipsis character."""
    card = MovieCard(1, _make_movie(cast=["A", "B", "C", "D", "E", "F"]))
    assert "…" in str(card.card_content)


def test_movie_card_long_plot_truncated() -> None:
    """Plot preview longer than 200 chars is truncated."""
    card = MovieCard(1, _make_movie(plot="x" * 300))
    assert "…" in str(card.card_content)


def test_movie_card_short_plot_preserved() -> None:
    """Short plot text appears verbatim in the rendered content."""
    card = MovieCard(1, _make_movie(plot="Short plot.", cast=["A"]))
    assert "Short plot." in str(card.card_content)


def test_movie_card_score_shown_when_provided() -> None:
    """Score string is included in card content when passed."""
    card = MovieCard(1, _make_movie(), score=0.9876)
    assert "0.9876" in str(card.card_content)


def test_movie_card_score_absent_when_none() -> None:
    """No sim label in card content when score is None."""
    card = MovieCard(1, _make_movie(), score=None)
    assert "sim " not in str(card.card_content)


def test_movie_card_shows_movie_id() -> None:
    """Movie id is always visible in the card metadata line."""
    card = MovieCard(1, _make_movie())
    assert "id:0" in str(card.card_content)
