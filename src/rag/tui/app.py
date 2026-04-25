"""Main Textual application for interactive RAG retrieval evaluation."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Label, Static, TextArea

from rag.config import DEFAULT_EMBEDDING_MODELS, EmbeddingProviderName, VectorStoreName
from rag.embeddings.factory import create_embedding_provider
from rag.models.movie import Movie
from rag.tui.constants import (
    BACKUP_BUTTON_ID,
    COMMAND_HELP,
    MESSAGE_BAR_ID,
    PROVIDER_NAMES,
    RESULTS_LIST_ID,
    SEARCH_BUTTON_ID,
    SEARCH_INPUT_ID,
    TOP_K_VALUES,
    VECTOR_STORE_NAMES,
    VECTOR_STORES,
)
from rag.tui.widgets import MovieCard, SettingsBar
from rag.vectorstore.factory import create_vector_store

logger = logging.getLogger(__name__)

_CSS = """
Screen {
    layout: vertical;
}

#results-scroll {
    height: 1fr;
    border-bottom: solid $primary-darken-3;
}

#results-list {
    height: auto;
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

#input-area {
    height: auto;
    max-height: 10;
    padding: 0 1;
    background: $surface;
    border-top: solid $primary;
    layout: horizontal;
    align: left middle;
}

#search-input {
    height: auto;
    max-height: 8;
    width: 1fr;
    margin-right: 1;
}

#button-col {
    width: 14;
    height: auto;
    layout: vertical;
}

#search-button {
    width: 100%;
    margin-bottom: 1;
}

#backup-button {
    width: 100%;
}

#settings-bar {
    height: 1;
    padding: 0 2;
    background: $panel;
    border-top: solid $primary-darken-3;
}

#message-bar {
    height: 1;
    padding: 0 2;
    background: $surface;
    color: $text-muted;
}
"""


class RetrievalApp(App[None]):
    """Movie Finder RAG — Interactive Retrieval TUI."""

    TITLE = "Movie Finder RAG — Retrieval TUI"
    CSS = _CSS

    BINDINGS = [
        Binding("ctrl+s", "search", "Search", show=True),
        Binding("ctrl+b", "backup", "Backup", show=True),
        Binding("ctrl+l", "clear_results", "Clear", show=True),
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    _provider: reactive[EmbeddingProviderName] = reactive("openai")
    _model: reactive[str] = reactive(DEFAULT_EMBEDDING_MODELS["openai"])
    _vector_store: reactive[VectorStoreName] = reactive("qdrant")
    _top_k: reactive[int] = reactive(5)

    def compose(self) -> ComposeResult:
        """Build the TUI widget tree."""
        yield Header()
        with ScrollableContainer(id="results-scroll"), Vertical(id=RESULTS_LIST_ID):
            yield Static(
                COMMAND_HELP,
                id="placeholder-msg",
            )
        with Vertical(id="input-area"):
            yield TextArea(
                id=SEARCH_INPUT_ID,
                language=None,
                show_line_numbers=False,
                tab_behavior="focus",
                soft_wrap=True,
            )
            with Vertical(id="button-col"):
                yield Button("Search", id=SEARCH_BUTTON_ID, variant="primary")
                yield Button("Backup", id=BACKUP_BUTTON_ID, variant="warning")
        yield SettingsBar()
        yield Label("", id=MESSAGE_BAR_ID)
        yield Footer()

    # ------------------------------------------------------------------
    # Reactive watchers — keep settings bar in sync
    # ------------------------------------------------------------------

    def watch__provider(self, value: EmbeddingProviderName) -> None:
        """Sync model default and refresh settings bar when provider changes."""
        self._model = DEFAULT_EMBEDDING_MODELS[value]
        self._refresh_settings_bar()

    def watch__model(self, _: str) -> None:
        self._refresh_settings_bar()

    def watch__vector_store(self, _: VectorStoreName) -> None:
        self._refresh_settings_bar()

    def watch__top_k(self, _: int) -> None:
        self._refresh_settings_bar()

    def _refresh_settings_bar(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one(SettingsBar).update_settings(
                self._provider, self._model, self._vector_store, self._top_k
            )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _model_options_for(provider: EmbeddingProviderName) -> list[tuple[str, str]]:
        """Return (label, value) model options for the given provider."""
        default = DEFAULT_EMBEDDING_MODELS[provider]
        return [(default, default)]

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    @on(Button.Pressed, f"#{SEARCH_BUTTON_ID}")
    def on_search_pressed(self) -> None:
        self.action_search()

    @on(Button.Pressed, f"#{BACKUP_BUTTON_ID}")
    def on_backup_pressed(self) -> None:
        self.action_backup()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_search(self) -> None:
        """Read the query, dispatch slash commands or kick off search."""
        text_area = self.query_one(f"#{SEARCH_INPUT_ID}", TextArea)
        query = text_area.text.strip()
        if not query:
            self._set_message("[yellow]Enter a query or /help for commands.[/yellow]")
            return
        if query.startswith("/"):
            self._handle_command(query)
            text_area.clear()
        else:
            self._run_search(query)

    def action_backup(self) -> None:
        self._run_backup()

    def action_clear_results(self) -> None:
        """Clear the results panel."""
        self._clear_results()
        self._set_message("[dim]Results cleared.[/dim]")

    # ------------------------------------------------------------------
    # Slash command dispatcher
    # ------------------------------------------------------------------

    def _handle_command(self, raw: str) -> None:
        """Parse and execute a slash command entered in the search field."""
        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        match cmd:
            case "/help" | "/?":
                self._show_help()
            case "/clear":
                self.action_clear_results()
            case "/provider" | "/p":
                self._cmd_provider(arg)
            case "/model" | "/m":
                self._cmd_model(arg)
            case "/store" | "/s":
                self._cmd_store(arg)
            case "/topk" | "/k":
                self._cmd_topk(arg)
            case _:
                self._set_message(
                    f"[red]Unknown command: {cmd}  — type /help for available commands[/red]"
                )

    def _cmd_provider(self, arg: str) -> None:
        if arg not in PROVIDER_NAMES:
            self._set_message(
                f"[red]Unknown provider: {arg!r}  Valid: {', '.join(sorted(PROVIDER_NAMES))}[/red]"
            )
            return
        self._provider = arg  # type: ignore[assignment]
        self._set_message(f"[green]Provider → {arg}[/green]")

    def _cmd_model(self, arg: str) -> None:
        if not arg:
            self._set_message("[red]/model requires a model name[/red]")
            return
        self._model = arg
        self._set_message(f"[green]Model → {arg}[/green]")

    def _cmd_store(self, arg: str) -> None:
        if arg not in VECTOR_STORE_NAMES:
            self._set_message(
                f"[red]Unknown store: {arg!r}  "
                f"Valid: {', '.join(v for _, v in VECTOR_STORES)}[/red]"
            )
            return
        self._vector_store = arg  # type: ignore[assignment]
        self._set_message(f"[green]Store → {arg}[/green]")

    def _cmd_topk(self, arg: str) -> None:
        try:
            n = int(arg)
        except ValueError:
            self._set_message(f"[red]/topk requires an integer, got: {arg!r}[/red]")
            return
        if n not in TOP_K_VALUES:
            self._set_message(f"[red]top-k must be one of {sorted(TOP_K_VALUES)}, got {n}[/red]")
            return
        self._top_k = n
        self._set_message(f"[green]top-k → {n}[/green]")

    def _show_help(self) -> None:
        results_panel = self.query_one(f"#{RESULTS_LIST_ID}", Vertical)
        for child in list(results_panel.children):
            child.remove()
        results_panel.mount(Static(COMMAND_HELP, id="help-msg"))
        self._set_message("[dim]/help — command reference[/dim]")

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    @work(exclusive=True, thread=True)
    def _run_search(self, query: str) -> None:
        """Embed the query and search the vector store in a background thread."""
        self.call_from_thread(self._set_searching_state, True)
        try:
            self.call_from_thread(
                self._set_message,
                f"[cyan]Searching  provider={self._provider}  "
                f"store={self._vector_store}  k={self._top_k}…[/cyan]",
            )
            provider = create_embedding_provider(
                provider_name=self._provider,
                model=self._model,
            )
            store = create_vector_store(vector_store_name=self._vector_store)
            model_info = provider.model_info
            vector = provider.embed(query)
            if not vector:
                self.call_from_thread(
                    self._set_message,
                    "[red]Embedding failed — check provider credentials.[/red]",
                )
                return
            results = store.search(vector, top_k=self._top_k, embedding_model=model_info)
            self.call_from_thread(self._display_results, results, query)
        except Exception as exc:
            logger.exception("Search worker error")
            self.call_from_thread(self._set_message, f"[red]Error: {exc}[/red]")
        finally:
            self.call_from_thread(self._set_searching_state, False)

    @work(exclusive=True, thread=True)
    def _run_backup(self) -> None:
        """Run the vector store backup in a background thread."""
        self.call_from_thread(
            self._set_message,
            f"[yellow]Starting backup  store={self._vector_store}…[/yellow]",
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
                self._set_message,
                f"[green]Backup complete: {stats.local_points} points → {stats.output_path}[/green]",
            )
        except Exception as exc:
            logger.exception("Backup worker error")
            self.call_from_thread(self._set_message, f"[red]Backup failed: {exc}[/red]")

    # ------------------------------------------------------------------
    # UI state helpers (main thread only)
    # ------------------------------------------------------------------

    def _set_message(self, message: str) -> None:
        """Update the single-line message bar."""
        self.query_one(f"#{MESSAGE_BAR_ID}", Label).update(message)

    def _set_searching_state(self, searching: bool) -> None:
        search_btn = self.query_one(f"#{SEARCH_BUTTON_ID}", Button)
        backup_btn = self.query_one(f"#{BACKUP_BUTTON_ID}", Button)
        search_btn.disabled = searching
        backup_btn.disabled = searching

    def _clear_results(self) -> None:
        results_panel = self.query_one(f"#{RESULTS_LIST_ID}", Vertical)
        for child in list(results_panel.children):
            child.remove()

    def _display_results(self, results: list[Movie], query: str) -> None:
        """Replace the results panel contents with fresh movie cards."""
        self._clear_results()
        results_panel = self.query_one(f"#{RESULTS_LIST_ID}", Vertical)

        if not results:
            results_panel.mount(Static("[dim]No results found.[/dim]"))
            self._set_message(f"[yellow]No results for: [italic]{query[:80]}[/italic][/yellow]")
            return

        cards: list[Any] = [MovieCard(i, movie) for i, movie in enumerate(results, start=1)]
        results_panel.mount(*cards)
        self._set_message(
            f"[green]{len(results)} result(s) for: [italic]{query[:80]}[/italic][/green]"
        )
