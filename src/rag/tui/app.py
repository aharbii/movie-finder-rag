"""Main Textual application for interactive RAG retrieval evaluation."""

from __future__ import annotations

import logging
from typing import Any, cast

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Input, Label, Log, Select, Static

from rag.config import DEFAULT_EMBEDDING_MODELS, EmbeddingProviderName, VectorStoreName
from rag.embeddings.factory import create_embedding_provider
from rag.models.movie import Movie
from rag.tui.constants import (
    BACKUP_BUTTON_ID,
    MODEL_SELECT_ID,
    PROVIDER_SELECT_ID,
    PROVIDERS,
    RESULTS_LIST_ID,
    SEARCH_BUTTON_ID,
    SEARCH_INPUT_ID,
    STATUS_LOG_ID,
    TOP_K_OPTIONS,
    TOP_K_SELECT_ID,
    VECTOR_STORE_SELECT_ID,
    VECTOR_STORES,
)
from rag.tui.widgets import MovieCard
from rag.vectorstore.factory import create_vector_store

logger = logging.getLogger(__name__)

_CSS = """
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


class RetrievalApp(App[None]):
    """Movie Finder RAG — Interactive Retrieval TUI."""

    TITLE = "Movie Finder RAG — Retrieval TUI"
    CSS = _CSS

    BINDINGS = [
        Binding("ctrl+s", "search", "Search"),
        Binding("ctrl+b", "backup", "Backup"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

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
                    id=PROVIDER_SELECT_ID,
                    allow_blank=False,
                )
                yield Label("Model")
                yield Select(
                    options=self._model_options_for("openai"),
                    value=DEFAULT_EMBEDDING_MODELS["openai"],
                    id=MODEL_SELECT_ID,
                    allow_blank=False,
                )
                yield Label("Vector Store")
                yield Select(
                    options=VECTOR_STORES,
                    value="qdrant",
                    id=VECTOR_STORE_SELECT_ID,
                    allow_blank=False,
                )
                yield Label("Top-K results")
                yield Select(
                    options=TOP_K_OPTIONS,
                    value=5,
                    id=TOP_K_SELECT_ID,
                    allow_blank=False,
                )
                yield Button("Backup Store", id=BACKUP_BUTTON_ID, variant="warning")

            with Vertical(id="main-area"):
                with Horizontal(id="search-bar"):
                    yield Input(
                        placeholder="Describe the movie you are looking for...",
                        id=SEARCH_INPUT_ID,
                    )
                    yield Button("Search", id=SEARCH_BUTTON_ID, variant="primary")

                with (
                    Horizontal(id="content-area"),
                    ScrollableContainer(id="results-scroll"),
                    Vertical(id=RESULTS_LIST_ID),
                ):
                    yield Static(
                        "[dim]Enter a query and press Search.[/dim]",
                        id="placeholder-msg",
                    )

                yield Log(id=STATUS_LOG_ID, highlight=True)

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
            model_select = self.query_one(f"#{MODEL_SELECT_ID}", Select)
            new_default = DEFAULT_EMBEDDING_MODELS[value]
            model_select.set_options(self._model_options_for(value))
            model_select.value = new_default
            self._model = new_default
        except Exception:  # widget may not be mounted yet
            pass

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    @on(Select.Changed, f"#{PROVIDER_SELECT_ID}")
    def on_provider_changed(self, event: Select.Changed) -> None:
        """React to provider selection change."""
        if event.value is Select.BLANK:
            return
        self._provider = cast(EmbeddingProviderName, event.value)

    @on(Select.Changed, f"#{MODEL_SELECT_ID}")
    def on_model_changed(self, event: Select.Changed) -> None:
        """React to model selection change."""
        if event.value is Select.BLANK:
            return
        self._model = str(event.value)

    @on(Select.Changed, f"#{VECTOR_STORE_SELECT_ID}")
    def on_vector_store_changed(self, event: Select.Changed) -> None:
        """React to vector store selection change."""
        if event.value is Select.BLANK:
            return
        self._vector_store = cast(VectorStoreName, event.value)

    @on(Select.Changed, f"#{TOP_K_SELECT_ID}")
    def on_top_k_changed(self, event: Select.Changed) -> None:
        """React to top-k selection change."""
        if event.value is Select.BLANK:
            return
        self._top_k = int(cast(int, event.value))

    @on(Button.Pressed, f"#{SEARCH_BUTTON_ID}")
    def on_search_pressed(self) -> None:
        """Trigger the search worker."""
        self.action_search()

    @on(Input.Submitted, f"#{SEARCH_INPUT_ID}")
    def on_search_submitted(self) -> None:
        """Submit search on Enter key in the search box."""
        self.action_search()

    @on(Button.Pressed, f"#{BACKUP_BUTTON_ID}")
    def on_backup_pressed(self) -> None:
        """Trigger the backup worker."""
        self.action_backup()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_search(self) -> None:
        """Read the query and kick off the background search worker."""
        query_input = self.query_one(f"#{SEARCH_INPUT_ID}", Input)
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

    @work(exclusive=True, thread=True)
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
                f"[cyan]Searching top-{self._top_k} in "
                f"[bold]{store.target_name(model_info)}[/bold]...[/cyan]",
            )
            results = store.search(vector, top_k=self._top_k, embedding_model=model_info)
            self.call_from_thread(self._display_results, results, query)

        except Exception as exc:
            logger.exception("Search worker error")
            self.call_from_thread(self._log_status, f"[red]Error during search: {exc}[/red]")
        finally:
            self.call_from_thread(self._set_searching_state, False)

    @work(exclusive=True, thread=True)
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
                f"[green]Backup complete: {stats.local_points} points "
                f"→ {stats.output_path}[/green]",
            )
        except Exception as exc:
            logger.exception("Backup worker error")
            self.call_from_thread(self._log_status, f"[red]Backup failed: {exc}[/red]")

    # ------------------------------------------------------------------
    # UI state helpers (must be called from the main thread)
    # ------------------------------------------------------------------

    def _log_status(self, message: str) -> None:
        """Append a message to the status log panel."""
        log_widget = self.query_one(f"#{STATUS_LOG_ID}", Log)
        log_widget.write_line(message)

    def _set_searching_state(self, searching: bool) -> None:
        """Toggle button states while a search or backup is in progress."""
        search_btn = self.query_one(f"#{SEARCH_BUTTON_ID}", Button)
        backup_btn = self.query_one(f"#{BACKUP_BUTTON_ID}", Button)
        search_btn.disabled = searching
        backup_btn.disabled = searching

    def _display_results(self, results: list[Movie], query: str) -> None:
        """Replace the results panel contents with fresh movie cards."""
        results_panel = self.query_one(f"#{RESULTS_LIST_ID}", Vertical)

        for child in list(results_panel.children):
            child.remove()

        if not results:
            results_panel.mount(Static("[dim]No matching movies found for that query.[/dim]"))
            self._log_status(
                f"[yellow]No results for: [italic]{query[:80]}[/italic][/yellow]"
            )
            return

        cards: list[Any] = [MovieCard(i, movie) for i, movie in enumerate(results, start=1)]
        results_panel.mount(*cards)
        self._log_status(
            f"[green]Found {len(results)} result(s) for: [italic]{query[:80]}[/italic][/green]"
        )
