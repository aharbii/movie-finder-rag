"""Custom Textual widgets for the Retrieval TUI."""

from __future__ import annotations

from textual.widgets import Static

from rag.config import DEFAULT_EMBEDDING_MODELS, EmbeddingProviderName, VectorStoreName
from rag.models.movie import Movie


class MovieCard(Static):
    """A formatted card displaying one search result."""

    def __init__(self, index: int, movie: Movie, score: float | None = None) -> None:
        score_str = f"  score {score:.4f}" if score is not None else ""
        cast_preview = ", ".join(movie.cast[:5])
        if len(movie.cast) > 5:
            cast_preview += "…"
        plot_preview = movie.plot[:200]
        if len(movie.plot) > 200:
            plot_preview += "…"
        content = (
            f"[bold]{index}. {movie.title}[/bold]"
            f"  [dim]{movie.release_year}{score_str}[/dim]\n"
            f"[dim]Director:[/dim] {movie.director}   "
            f"[dim]Genre:[/dim] {', '.join(movie.genre)}\n"
            f"[dim]Cast:[/dim] {cast_preview}\n"
            f"{plot_preview}"
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


class ConfigBar(Static):
    """One-line bar showing the active TUI configuration."""

    DEFAULT_CSS = """
    ConfigBar {
        height: 1;
        padding: 0 1;
        background: $panel;
        dock: bottom;
    }
    """

    def __init__(self) -> None:
        super().__init__("", id="config-bar")
        self._provider: EmbeddingProviderName = "openai"
        self._model: str = DEFAULT_EMBEDDING_MODELS["openai"]
        self._store: VectorStoreName = "qdrant"
        self._top_k: int = 5

    def refresh_config(
        self,
        provider: EmbeddingProviderName,
        model: str,
        store: VectorStoreName,
        top_k: int,
    ) -> None:
        """Redraw whenever a setting changes."""
        self._provider = provider
        self._model = model
        self._store = store
        self._top_k = top_k
        self.update(self._line())

    def _line(self) -> str:
        return (
            f"provider=[bold]{self._provider}[/bold]  "
            f"model=[bold]{self._model}[/bold]  "
            f"store=[bold]{self._store}[/bold]  "
            f"top-k=[bold]{self._top_k}[/bold]"
        )
