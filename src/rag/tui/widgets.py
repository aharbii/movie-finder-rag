"""Custom Textual widgets for the Retrieval TUI."""

from __future__ import annotations

from textual.widgets import Static

from rag.models.movie import Movie


class MovieCard(Static):
    """A formatted card displaying one search result."""

    def __init__(self, index: int, movie: Movie, score: float | None = None) -> None:
        score_str = f" — score: {score:.4f}" if score is not None else ""
        cast_preview = ", ".join(movie.cast[:5])
        if len(movie.cast) > 5:
            cast_preview += "..."
        plot_preview = movie.plot[:200]
        if len(movie.plot) > 200:
            plot_preview += "..."
        content = (
            f"[bold cyan]{index}. {movie.title}[/bold cyan] "
            f"[dim]({movie.release_year}){score_str}[/dim]\n"
            f"[yellow]Director:[/yellow] {movie.director}   "
            f"[yellow]Genre:[/yellow] {', '.join(movie.genre)}\n"
            f"[yellow]Cast:[/yellow] {cast_preview}\n"
            f"[dim]{plot_preview}[/dim]"
        )
        super().__init__(content, classes="movie-card")
        self._movie = movie
        self._index = index
        self._card_content = content

    @property
    def movie(self) -> Movie:
        """Return the underlying Movie model."""
        return self._movie

    @property
    def card_content(self) -> str:
        """Return the rendered card markup as a plain string (for testing)."""
        return self._card_content
