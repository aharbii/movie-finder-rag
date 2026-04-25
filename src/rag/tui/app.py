"""Main Textual application for interactive RAG retrieval evaluation."""

from __future__ import annotations

import contextlib
import logging
from functools import partial
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.containers import ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Label, OptionList, Static, TextArea

from rag.config import DEFAULT_EMBEDDING_MODELS, EmbeddingProviderName, VectorStoreName
from rag.embeddings.factory import create_embedding_provider
from rag.models.movie import Movie
from rag.tui.constants import (
    BACKUP_BUTTON_ID,
    COMMAND_REGISTRY,
    COMPLETION_LIST_ID,
    MESSAGE_BAR_ID,
    PROVIDER_NAMES,
    RESULTS_LIST_ID,
    SEARCH_BUTTON_ID,
    SEARCH_INPUT_ID,
    TOP_K_VALUES,
    VECTOR_STORE_NAMES,
    VECTOR_STORES,
)
from rag.tui.widgets import ConfigBar, MovieCard
from rag.vectorstore.factory import create_vector_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Textual CommandPalette provider (Ctrl+P)
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
[bold]Movie Finder RAG — Retrieval TUI[/bold]

[dim]Search:[/dim]  Type a query and press [bold]Ctrl+S[/bold] or the Search button.

[dim]Settings:[/dim]
  Type [bold]/[/bold] in the search box → completion list appears
  Press [bold]↓[/bold] to enter the list, [bold]↑↓[/bold] to navigate, [bold]Enter[/bold] to apply
  Press [bold]Esc[/bold] to dismiss  ·  [bold]Ctrl+P[/bold] for the full command palette

  Available settings and actions are listed below as you type.
  You never need to type a value manually — just select from the list.

[dim]Keys:[/dim]  Ctrl+S search  ·  Ctrl+P palette  ·  Ctrl+B backup  ·  Ctrl+L clear  ·  Ctrl+Q quit\
"""


class RAGCommandProvider(Provider):
    """Supplies all setting-change commands to the Textual CommandPalette."""

    async def discover(self) -> Hits:
        app: RetrievalApp = self.app  # type: ignore[assignment]
        for cmd_id, name, description in COMMAND_REGISTRY:
            yield self._make_hit(app, cmd_id, name, description)

    async def search(self, query: str) -> Hits:
        app: RetrievalApp = self.app  # type: ignore[assignment]
        matcher = self.matcher(query)
        for cmd_id, name, description in COMMAND_REGISTRY:
            score = matcher.match(f"{name} {description}")
            if score > 0:
                yield Hit(
                    score=score,
                    match_display=matcher.highlight(name),
                    command=partial(app._dispatch_command_id, cmd_id),
                    text=name,
                    help=description,
                )

    @staticmethod
    def _make_hit(app: RetrievalApp, cmd_id: str, name: str, description: str) -> Hit:
        return Hit(
            score=0,
            match_display=name,
            command=partial(app._dispatch_command_id, cmd_id),
            text=name,
            help=description,
        )


# ---------------------------------------------------------------------------
# App CSS — layout only, no color overrides; uses Textual's default theme
# ---------------------------------------------------------------------------

_CSS = """
#results-scroll {
    height: 1fr;
}

#results-list {
    height: auto;
    padding: 1 2;
}

.movie-card {
    border: round $primary;
    padding: 0 1;
    margin-bottom: 1;
}

#completion-list {
    height: auto;
    max-height: 12;
    display: none;
}

#input-row {
    height: auto;
    max-height: 10;
    padding: 0 1;
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

#btn-col {
    width: 12;
    height: auto;
    layout: vertical;
    align: center top;
}

#search-button { width: 100%; margin-bottom: 1; }
#backup-button { width: 100%; }

#message-bar {
    height: 1;
    padding: 0 1;
    color: $text-muted;
}
"""

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class RetrievalApp(App[None]):
    """Movie Finder RAG — Interactive Retrieval TUI."""

    TITLE = "Movie Finder RAG — Retrieval TUI"
    CSS = _CSS
    COMMANDS = {RAGCommandProvider}

    BINDINGS = [
        Binding("ctrl+s", "search", "Search"),
        Binding("ctrl+p", "command_palette", "Commands"),
        Binding("ctrl+b", "backup", "Backup"),
        Binding("ctrl+l", "clear_results", "Clear"),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("escape", "dismiss_completion", "Dismiss", show=False),
        Binding("down", "focus_completion", "↓ complete", show=False),
    ]

    _provider: reactive[EmbeddingProviderName] = reactive("openai")
    _model: reactive[str] = reactive(DEFAULT_EMBEDDING_MODELS["openai"])
    _vector_store: reactive[VectorStoreName] = reactive("qdrant")
    _top_k: reactive[int] = reactive(5)

    # Tracks command IDs for the visible completion list by index
    _completion_ids: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="results-scroll"), Vertical(id=RESULTS_LIST_ID):
            yield Static(_HELP_TEXT, id="help-msg")
        yield OptionList(id=COMPLETION_LIST_ID)
        with Vertical(id="input-row"):
            yield TextArea(
                id=SEARCH_INPUT_ID,
                language=None,
                show_line_numbers=False,
                tab_behavior="focus",
                soft_wrap=True,
            )
            with Vertical(id="btn-col"):
                yield Button("Search", id=SEARCH_BUTTON_ID, variant="primary")
                yield Button("Backup", id=BACKUP_BUTTON_ID, variant="warning")
        yield ConfigBar()
        yield Label("", id=MESSAGE_BAR_ID)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(f"#{SEARCH_INPUT_ID}", TextArea).focus()

    # ------------------------------------------------------------------
    # Reactive watchers
    # ------------------------------------------------------------------

    def watch__provider(self, value: EmbeddingProviderName) -> None:
        self._model = DEFAULT_EMBEDDING_MODELS[value]
        self._sync_config_bar()

    def watch__model(self, _: str) -> None:
        self._sync_config_bar()

    def watch__vector_store(self, _: VectorStoreName) -> None:
        self._sync_config_bar()

    def watch__top_k(self, _: int) -> None:
        self._sync_config_bar()

    def _sync_config_bar(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one(ConfigBar).refresh_config(
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
    # Inline completion — triggered by TextArea.Changed
    # ------------------------------------------------------------------

    @on(TextArea.Changed, f"#{SEARCH_INPUT_ID}")
    def on_search_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        completion = self.query_one(f"#{COMPLETION_LIST_ID}", OptionList)
        if text.startswith("/"):
            needle = text.lower()
            completion.clear_options()
            self._completion_ids = []
            for cmd_id, name, description in COMMAND_REGISTRY:
                if needle in name or needle in description.lower():
                    line = Text()
                    line.append(f"{name:<35}", style="bold")
                    line.append(description, style="dim")
                    completion.add_option(line)
                    self._completion_ids.append(cmd_id)
            completion.display = len(self._completion_ids) > 0
        else:
            completion.display = False

    @on(OptionList.OptionSelected, f"#{COMPLETION_LIST_ID}")
    def on_completion_selected(self, event: OptionList.OptionSelected) -> None:
        idx = event.option_index
        if idx < len(self._completion_ids):
            self._dispatch_command_id(self._completion_ids[idx])
        self.query_one(f"#{SEARCH_INPUT_ID}", TextArea).clear()
        self.query_one(f"#{COMPLETION_LIST_ID}", OptionList).display = False
        self.query_one(f"#{SEARCH_INPUT_ID}", TextArea).focus()

    # ------------------------------------------------------------------
    # Keyboard actions
    # ------------------------------------------------------------------

    def action_focus_completion(self) -> None:
        """Move focus into the completion list when it is visible (Down key)."""
        completion = self.query_one(f"#{COMPLETION_LIST_ID}", OptionList)
        if completion.display:
            completion.focus()

    def action_dismiss_completion(self) -> None:
        """Hide the completion list and return focus to the search input."""
        self.query_one(f"#{COMPLETION_LIST_ID}", OptionList).display = False
        self.query_one(f"#{SEARCH_INPUT_ID}", TextArea).focus()

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
        """Execute search or dispatch slash command from the input box."""
        text_area = self.query_one(f"#{SEARCH_INPUT_ID}", TextArea)
        query = text_area.text.strip()
        if not query:
            self._set_message("Enter a query — or / for commands, Ctrl+P for palette")
            return
        if query.startswith("/"):
            self._dispatch_slash(query)
            text_area.clear()
        else:
            self._run_search(query)

    def action_backup(self) -> None:
        self._run_backup()

    def action_clear_results(self) -> None:
        self._clear_results()
        self._set_message("Results cleared.")

    # ------------------------------------------------------------------
    # Slash / command-id dispatch
    # ------------------------------------------------------------------

    def _dispatch_slash(self, raw: str) -> None:
        """Dispatch a slash command — exact registry match, then special cases."""
        raw = raw.strip()
        # Exact match in registry (e.g. "/provider openai", "/store qdrant")
        for cmd_id, name, _ in COMMAND_REGISTRY:
            if raw == name:
                self._dispatch_command_id(cmd_id)
                return
        # Special cases not in the registry
        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        match cmd:
            case "/model" | "/m":
                self._do_model(arg)
            case "/quit" | "/q" | "/exit":
                self.exit()
            case _:
                self._set_message(f"Unknown: {raw!r}  — type / and press ↓ to browse, or Ctrl+P")

    def _dispatch_command_id(self, cmd_id: str) -> None:
        """Execute a command by its registry ID (used by CommandPalette and completion)."""
        kind, _, value = cmd_id.partition(":")
        match kind:
            case "provider":
                self._do_provider(value)
            case "store":
                self._do_store(value)
            case "topk":
                self._do_topk(value)
            case "action":
                match value:
                    case "clear":
                        self.action_clear_results()
                    case "help":
                        self._show_help()
                    case "quit":
                        self.exit()

    def _do_provider(self, arg: str) -> None:
        if arg not in PROVIDER_NAMES:
            self._set_message(
                f"Unknown provider: {arg!r}  Valid: {', '.join(sorted(PROVIDER_NAMES))}"
            )
            return
        self._provider = arg  # type: ignore[assignment]
        self._set_message(f"provider → {arg}")

    def _do_model(self, arg: str) -> None:
        if not arg:
            self._set_message("/model requires a model name")
            return
        self._model = arg
        self._set_message(f"model → {arg}")

    def _do_store(self, arg: str) -> None:
        if arg not in VECTOR_STORE_NAMES:
            self._set_message(
                f"Unknown store: {arg!r}  Valid: {', '.join(v for _, v in VECTOR_STORES)}"
            )
            return
        self._vector_store = arg  # type: ignore[assignment]
        self._set_message(f"store → {arg}")

    def _do_topk(self, arg: str) -> None:
        try:
            n = int(arg)
        except ValueError:
            self._set_message(f"/topk requires an integer, got: {arg!r}")
            return
        if n not in TOP_K_VALUES:
            self._set_message(f"top-k must be one of {sorted(TOP_K_VALUES)}, got {n}")
            return
        self._top_k = n
        self._set_message(f"top-k → {n}")

    def _show_help(self) -> None:
        self._clear_results()
        results_panel = self.query_one(f"#{RESULTS_LIST_ID}", Vertical)
        results_panel.mount(Static(_HELP_TEXT))

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    @work(exclusive=True, thread=True)
    def _run_search(self, query: str) -> None:
        self.call_from_thread(self._set_searching_state, True)
        try:
            self.call_from_thread(
                self._set_message,
                f"Searching  provider={self._provider}  store={self._vector_store}  k={self._top_k}…",
            )
            provider = create_embedding_provider(provider_name=self._provider, model=self._model)
            store = create_vector_store(vector_store_name=self._vector_store)
            model_info = provider.model_info
            vector = provider.embed(query)
            if not vector:
                self.call_from_thread(self._set_message, "Embedding failed — check credentials.")
                return
            results = store.search(vector, top_k=self._top_k, embedding_model=model_info)
            self.call_from_thread(self._display_results, results, query)
        except Exception as exc:
            logger.exception("Search worker error")
            self.call_from_thread(self._set_message, f"Error: {exc}")
        finally:
            self.call_from_thread(self._set_searching_state, False)

    @work(exclusive=True, thread=True)
    def _run_backup(self) -> None:
        self.call_from_thread(self._set_message, f"Starting backup  store={self._vector_store}…")
        try:
            from backup_vectorstore import BackupConfig, backup_vector_store
            from rag.config import settings

            provider = create_embedding_provider(provider_name=self._provider, model=self._model)
            store = create_vector_store(vector_store_name=self._vector_store)
            model_info = provider.model_info
            config = BackupConfig(
                vector_store=self._vector_store,
                collection_name=store.target_name(model_info),
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
                f"Backup complete: {stats.local_points} points → {stats.output_path}",
            )
        except Exception as exc:
            logger.exception("Backup worker error")
            self.call_from_thread(self._set_message, f"Backup failed: {exc}")

    # ------------------------------------------------------------------
    # UI helpers (main thread only)
    # ------------------------------------------------------------------

    def _set_message(self, message: str) -> None:
        self.query_one(f"#{MESSAGE_BAR_ID}", Label).update(message)

    def _set_searching_state(self, searching: bool) -> None:
        self.query_one(f"#{SEARCH_BUTTON_ID}", Button).disabled = searching
        self.query_one(f"#{BACKUP_BUTTON_ID}", Button).disabled = searching

    def _clear_results(self) -> None:
        results_panel = self.query_one(f"#{RESULTS_LIST_ID}", Vertical)
        for child in list(results_panel.children):
            child.remove()

    def _display_results(self, results: list[Movie], query: str) -> None:
        self._clear_results()
        results_panel = self.query_one(f"#{RESULTS_LIST_ID}", Vertical)
        if not results:
            results_panel.mount(Static("No results found."))
            self._set_message(f"No results for: {query[:80]}")
            return
        cards: list[Any] = [MovieCard(i, movie) for i, movie in enumerate(results, start=1)]
        results_panel.mount(*cards)
        self._set_message(f"{len(results)} result(s) for: {query[:80]}")
