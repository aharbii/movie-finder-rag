"""Custom Textual widgets for the Retrieval TUI."""

from __future__ import annotations

import contextlib

from rich.text import Text
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import ListItem, ListView, Static

from rag.config import EmbeddingProviderName, VectorStoreName
from rag.models.movie import Movie
from tui.constants import OverlayItem


class MovieCard(Static):
    """A formatted card displaying one search result."""

    def __init__(self, index: int, movie: Movie, score: float | None = None) -> None:
        meta_parts = [str(movie.release_year), f"id:{movie.id}"]
        if score is not None:
            meta_parts.append(f"[bold green]sim {score:.4f}[/bold green]")
        meta_str = "  ".join(meta_parts)
        cast_preview = ", ".join(movie.cast[:5])
        if len(movie.cast) > 5:
            cast_preview += "…"
        plot_preview = movie.plot[:200]
        if len(movie.plot) > 200:
            plot_preview += "…"
        content = (
            f"[bold]{index}. {movie.title}[/bold]"
            f"  [dim]{meta_str}[/dim]\n"
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
        """Return the underlying Movie instance."""
        return self._movie

    @property
    def card_content(self) -> str:
        """Return the rendered markup string (for testing)."""
        return self._card_content


class StatusBar(Widget):
    """One-line bar always docked at the bottom showing the active configuration."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        padding: 0 1;
        background: $panel;
    }
    """

    def __init__(
        self,
        provider: EmbeddingProviderName = "openai",
        model: str = "",
        store: VectorStoreName = "qdrant",
        collection: str = "",
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self._cur_provider = provider
        self._cur_model = model
        self._cur_store = store
        self._cur_collection = collection
        self._cur_top_k = top_k
        self._connections: dict[str, bool | None] = {}

    def compose(self) -> ComposeResult:
        yield Static("", id=StatusBar.CONTENT_ID)

    def on_mount(self) -> None:
        self._redraw()

    CONTENT_ID = "status-bar-content"

    def refresh_status(
        self,
        provider: EmbeddingProviderName,
        model: str,
        store: VectorStoreName,
        collection: str,
        top_k: int,
        connections: dict[str, bool | None],
    ) -> None:
        """Redraw the status line whenever a setting or connection state changes."""
        self._cur_provider = provider
        self._cur_model = model
        self._cur_store = store
        self._cur_collection = collection
        self._cur_top_k = top_k
        self._connections = dict(connections)
        self._redraw()

    def _redraw(self) -> None:
        def dot(key: str) -> tuple[str, str]:
            val = self._connections.get(key)
            if val is True:
                return "●", "green"
            if val is False:
                return "○", "red"
            return "·", "dim"

        model_short = self._cur_model.split("/")[-1] if "/" in self._cur_model else self._cur_model
        col = self._cur_collection or "—"

        p_dot, p_style = dot(self._cur_provider)
        s_dot, s_style = dot(self._cur_store)

        line = Text(overflow="ellipsis", no_wrap=True)
        line.append(" provider: ", style="dim")
        line.append(p_dot, style=p_style)
        line.append(f" {self._cur_provider}", style="bold")
        if model_short:
            line.append(f"/{model_short}", style="dim")
        line.append("   store: ", style="dim")
        line.append(s_dot, style=s_style)
        line.append(f" {self._cur_store}", style="bold")
        line.append("   collection: ", style="dim")
        line.append(col, style="italic")
        line.append("   top_k: ", style="dim")
        line.append(str(self._cur_top_k), style="bold")

        with contextlib.suppress(Exception):
            self.query_one(f"#{StatusBar.CONTENT_ID}", Static).update(line)


class CommandOverlay(Widget):
    """Floating overlay with a keyboard-navigable list for slash command completions."""

    DEFAULT_CSS = """
    CommandOverlay {
        height: auto;
        max-height: 14;
        border: solid $primary;
        display: none;
    }
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._items: list[OverlayItem] = []

    def compose(self) -> ComposeResult:
        yield ListView(id="overlay-list")

    def populate(self, items: list[OverlayItem]) -> None:
        """Replace the overlay contents with the given items and show/hide accordingly.

        ListItems are NOT given widget IDs to avoid Textual's DuplicateIds error when
        populate() is called repeatedly before the async clear() completes. Items are
        tracked by index via self._items instead.
        """
        lv = self.query_one("#overlay-list", ListView)
        lv.clear()
        self._items = list(items)
        for item in items:
            row = Text()
            row.append(f"{item.name:<28}", style="bold")
            row.append(f" {item.description}", style="dim")
            if item.indicator:
                row.append(f"  {item.indicator}", style="bold")
            lv.append(ListItem(Static(row)))  # no id — avoid DuplicateIds on rapid repopulate
        self.display = len(items) > 0

    def item_at(self, index: int) -> OverlayItem | None:
        """Return the OverlayItem at the given list index, or None if out of range."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def focus_list(self) -> None:
        """Move keyboard focus into the list."""
        self.query_one("#overlay-list", ListView).focus()
