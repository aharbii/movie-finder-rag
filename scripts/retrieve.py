"""Textual TUI for interactive RAG retrieval evaluation.

Replaces the old linear CLI script with a rich terminal dashboard that allows
developers to dynamically select embedding providers, models, and vector stores,
then query and visualise search results with expanded metadata.

Launch via:
    make retrieve
    # or directly:
    python scripts/retrieve.py
"""

from __future__ import annotations

import logging
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    Log,
    Select,
    Static,
)

from rag.config import (
    DEFAULT_EMBEDDING_MODELS,
    EmbeddingProviderName,
    VectorStoreName,
)
from rag.embeddings.factory import create_embedding_provider
from rag.models.movie import Movie
from rag.vectorstore.factory import create_vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVIDERS: list[tuple[str, EmbeddingProviderName]] = [
    ("OpenAI", "openai"),
    ("Ollama", "ollama"),
    ("HuggingFace", "huggingface"),
    ("Sentence Transformers", "sentence-transformers"),
    ("Google", "google"),
]

VECTOR_STORES: list[tuple[str, VectorStoreName]] = [
    ("Qdrant", "qdrant"),
    ("ChromaDB", "chromadb"),
    ("Pinecone", "pinecone"),
    ("PGVector", "pgvector"),
]

TOP_K_OPTIONS: list[tuple[str, int]] = [
    ("3", 3),
    ("5", 5),
    ("10", 10),
    ("15", 15),
    ("20", 20),
]

_PROVIDER_SELECT_ID = "provider-select"
_MODEL_SELECT_ID = "model-select"
_VECTOR_STORE_SELECT_ID = "vector-store-select"
_TOP_K_SELECT_ID = "top-k-select"
_SEARCH_INPUT_ID = "search-input"
_SEARCH_BUTTON_ID = "search-button"
_BACKUP_BUTTON_ID = "backup-button"
_RESULTS_LIST_ID = "results-list"
_STATUS_LOG_ID = "status-log"
_DETAIL_PANEL_ID = "detail-panel"


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------


class MovieCard(Static):  # type: ignore[misc]
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

    @property
    def movie(self) -> Movie:
        """Return the underlying Movie model."""
        return self._movie


# ---------------------------------------------------------------------------
# Main TUI application
# ---------------------------------------------------------------------------


class RetrievalApp(App[None]):  # type: ignore[misc]
    """Movie Finder RAG — Interactive Retrieval TUI."""

    TITLE = "Movie Finder RAG — Retrieval TUI"
    CSS = """
    Screen {
        layout: horizontal;
    }

    #sidebar {
        width: 30;
        min-width: 28;
        height: 100%;
        background: $panel;
        border-right: solid $primary;
        padding: 1 2;
    }

    #sidebar Label {
        color: $text-muted;
        margin-top: 1;
    }

    #sidebar Select {
        margin-bottom: 1;
    }

    #main-area {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }

    #search-bar {
        height: 5;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $primary;
        layout: horizontal;
    }

    #search-input {
        width: 1fr;
        margin-right: 1;
    }

    #search-button {
        width: 14;
    }

    #content-area {
        height: 1fr;
        layout: horizontal;
    }

    #results-scroll {
        width: 1fr;
        height: 100%;
    }

    #results-list {
        height: auto;
        min-height: 100%;
        padding: 1 2;
    }

    .movie-card {
        border: round $primary-darken-2;
        padding: 0 1;
        margin-bottom: 1;
    }

    .movie-card:hover {
        border: round $accent;
        background: $surface-lighten-1;
    }

    #status-log {
        height: 7;
        border-top: solid $primary;
        padding: 0 1;
        background: $surface;
    }

    #backup-button {
        margin-top: 1;
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "search", "Search"),
        Binding("ctrl+b", "backup", "Backup"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    # Reactive configuration state — triggers re-init when changed
    _provider: reactive[EmbeddingProviderName] = reactive("openai")
    _model: reactive[str] = reactive(DEFAULT_EMBEDDING_MODELS["openai"])
    _vector_store: reactive[VectorStoreName] = reactive("qdrant")
    _top_k: reactive[int] = reactive(5)

    def compose(self) -> ComposeResult:
        """Build the TUI widget tree."""
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("Provider")
                yield Select(
                    options=PROVIDERS,
                    value="openai",
                    id=_PROVIDER_SELECT_ID,
                    allow_blank=False,
                )
                yield Label("Model")
                yield Select(
                    options=self._model_options_for("openai"),
                    value=DEFAULT_EMBEDDING_MODELS["openai"],
                    id=_MODEL_SELECT_ID,
                    allow_blank=False,
                )
                yield Label("Vector Store")
                yield Select(
                    options=VECTOR_STORES,
                    value="qdrant",
                    id=_VECTOR_STORE_SELECT_ID,
                    allow_blank=False,
                )
                yield Label("Top-K results")
                yield Select(
                    options=TOP_K_OPTIONS,
                    value=5,
                    id=_TOP_K_SELECT_ID,
                    allow_blank=False,
                )
                yield Button("Backup Store", id=_BACKUP_BUTTON_ID, variant="warning")

            with Vertical(id="main-area"):
                with Horizontal(id="search-bar"):
                    yield Input(
                        placeholder="Describe the movie you are looking for...",
                        id=_SEARCH_INPUT_ID,
                    )
                    yield Button("Search", id=_SEARCH_BUTTON_ID, variant="primary")

                with (
                    Horizontal(id="content-area"),
                    ScrollableContainer(id="results-scroll"),
                    Vertical(id=_RESULTS_LIST_ID),
                ):
                    yield Static(
                        "[dim]Enter a query and press Search.[/dim]",
                        id="placeholder-msg",
                    )

                yield Log(id=_STATUS_LOG_ID, highlight=True, markup=True)

        yield Footer()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_options_for(provider: EmbeddingProviderName) -> list[tuple[str, str]]:
        """Return (label, value) model options for the given provider."""
        default = DEFAULT_EMBEDDING_MODELS[provider]
        return [(default, default)]

    # ------------------------------------------------------------------
    # Reactive watchers
    # ------------------------------------------------------------------

    def watch__provider(self, value: EmbeddingProviderName) -> None:
        """Update the model select when the provider changes."""
        try:
            model_select = self.query_one(f"#{_MODEL_SELECT_ID}", Select)
            new_default = DEFAULT_EMBEDDING_MODELS[value]
            model_select.set_options(self._model_options_for(value))
            model_select.value = new_default
            self._model = new_default
        except Exception:  # widget may not be mounted yet
            pass

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    @on(Select.Changed, f"#{_PROVIDER_SELECT_ID}")  # type: ignore[untyped-decorator]
    def on_provider_changed(self, event: Select.Changed) -> None:
        """React to provider selection change."""
        if event.value is Select.BLANK:
            return
        self._provider = event.value

    @on(Select.Changed, f"#{_MODEL_SELECT_ID}")  # type: ignore[untyped-decorator]
    def on_model_changed(self, event: Select.Changed) -> None:
        """React to model selection change."""
        if event.value is Select.BLANK:
            return
        self._model = str(event.value)

    @on(Select.Changed, f"#{_VECTOR_STORE_SELECT_ID}")  # type: ignore[untyped-decorator]
    def on_vector_store_changed(self, event: Select.Changed) -> None:
        """React to vector store selection change."""
        if event.value is Select.BLANK:
            return
        self._vector_store = event.value

    @on(Select.Changed, f"#{_TOP_K_SELECT_ID}")  # type: ignore[untyped-decorator]
    def on_top_k_changed(self, event: Select.Changed) -> None:
        """React to top-k selection change."""
        if event.value is Select.BLANK:
            return
        self._top_k = int(event.value)

    @on(Button.Pressed, f"#{_SEARCH_BUTTON_ID}")  # type: ignore[untyped-decorator]
    def on_search_pressed(self) -> None:
        """Trigger the search worker."""
        self.action_search()

    @on(Input.Submitted, f"#{_SEARCH_INPUT_ID}")  # type: ignore[untyped-decorator]
    def on_search_submitted(self) -> None:
        """Submit search on Enter key in the search box."""
        self.action_search()

    @on(Button.Pressed, f"#{_BACKUP_BUTTON_ID}")  # type: ignore[untyped-decorator]
    def on_backup_pressed(self) -> None:
        """Trigger the backup worker."""
        self.action_backup()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_search(self) -> None:
        """Read the query and kick off the background search worker."""
        query_input = self.query_one(f"#{_SEARCH_INPUT_ID}", Input)
        query = query_input.value.strip()
        if not query:
            self._log_status("[yellow]Please enter a search query first.[/yellow]")
            return
        self._run_search(query)

    def action_backup(self) -> None:
        """Kick off the background backup worker."""
        self._run_backup()

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    @work(exclusive=True, thread=True)  # type: ignore[untyped-decorator]
    def _run_search(self, query: str) -> None:
        """Embed the query and search the vector store in a background thread."""
        self.call_from_thread(self._set_searching_state, True)
        try:
            self.call_from_thread(
                self._log_status,
                f"[cyan]Initialising provider=[bold]{self._provider}[/bold] "
                f"model=[bold]{self._model}[/bold] "
                f"store=[bold]{self._vector_store}[/bold][/cyan]",
            )
            provider = create_embedding_provider(
                provider_name=self._provider,
                model=self._model,
            )
            store = create_vector_store(vector_store_name=self._vector_store)
            model_info = provider.model_info

            self.call_from_thread(
                self._log_status,
                f"[cyan]Embedding query: [italic]{query[:80]}[/italic]...[/cyan]",
            )
            vector = provider.embed(query)
            if not vector:
                self.call_from_thread(
                    self._log_status,
                    "[red]Failed to generate embedding — check provider credentials.[/red]",
                )
                return

            self.call_from_thread(
                self._log_status,
                f"[cyan]Searching top-{self._top_k} in [bold]{store.target_name(model_info)}[/bold]...[/cyan]",
            )
            results = store.search(vector, top_k=self._top_k, embedding_model=model_info)
            self.call_from_thread(self._display_results, results, query)

        except Exception as exc:
            logger.exception("Search worker error")
            self.call_from_thread(self._log_status, f"[red]Error during search: {exc}[/red]")
        finally:
            self.call_from_thread(self._set_searching_state, False)

    @work(exclusive=True, thread=True)  # type: ignore[untyped-decorator]
    def _run_backup(self) -> None:
        """Run the vector store backup in a background thread."""
        self.call_from_thread(
            self._log_status,
            f"[yellow]Starting backup for store=[bold]{self._vector_store}[/bold]...[/yellow]",
        )
        try:
            from backup_vectorstore import BackupConfig, backup_vector_store
            from rag.config import settings

            provider = create_embedding_provider(provider_name=self._provider, model=self._model)
            store = create_vector_store(vector_store_name=self._vector_store)
            model_info = provider.model_info
            collection_name = store.target_name(model_info)

            config = BackupConfig(
                vector_store=self._vector_store,
                collection_name=collection_name,
                output_root=__import__("pathlib").Path("outputs/backups"),
                batch_size=settings.batch_size,
                embedding_provider=self._provider,
                embedding_model=self._model,
                embedding_dimension=model_info.dimension,
                vector_store_url=settings.vector_store_url or settings.qdrant_url,
                vector_store_api_key=settings.vector_store_api_key or settings.qdrant_api_key_rw,
                chromadb_persist_path=settings.chromadb_persist_path,
                pinecone_index_name=settings.pinecone_index_name,
                pinecone_index_host=settings.pinecone_index_host,
                pgvector_dsn=settings.pgvector_dsn,
                pgvector_schema=settings.pgvector_schema,
            )
            stats = backup_vector_store(config)
            self.call_from_thread(
                self._log_status,
                f"[green]Backup complete: {stats.local_points} points → {stats.output_path}[/green]",
            )
        except Exception as exc:
            logger.exception("Backup worker error")
            self.call_from_thread(self._log_status, f"[red]Backup failed: {exc}[/red]")

    # ------------------------------------------------------------------
    # UI state helpers (must be called from the main thread)
    # ------------------------------------------------------------------

    def _log_status(self, message: str) -> None:
        """Append a message to the status log panel."""
        log_widget = self.query_one(f"#{_STATUS_LOG_ID}", Log)
        log_widget.write_line(message)

    def _set_searching_state(self, searching: bool) -> None:
        """Toggle button states while a search or backup is in progress."""
        search_btn = self.query_one(f"#{_SEARCH_BUTTON_ID}", Button)
        backup_btn = self.query_one(f"#{_BACKUP_BUTTON_ID}", Button)
        search_btn.disabled = searching
        backup_btn.disabled = searching

    def _display_results(self, results: list[Movie], query: str) -> None:
        """Replace the results panel contents with fresh movie cards."""
        results_panel = self.query_one(f"#{_RESULTS_LIST_ID}", Vertical)

        # Remove all existing children
        for child in list(results_panel.children):
            child.remove()

        if not results:
            results_panel.mount(Static("[dim]No matching movies found for that query.[/dim]"))
            self._log_status(f"[yellow]No results for: [italic]{query[:80]}[/italic][/yellow]")
            return

        cards: list[Any] = []
        for index, movie in enumerate(results, start=1):
            cards.append(MovieCard(index, movie))

        results_panel.mount(*cards)
        self._log_status(
            f"[green]Found {len(results)} result(s) for: [italic]{query[:80]}[/italic][/green]"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launch the Retrieval TUI application."""
    app = RetrievalApp()
    app.run()


if __name__ == "__main__":
    main()
