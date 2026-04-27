"""Main Textual application for interactive RAG retrieval evaluation."""

from __future__ import annotations

import contextlib
import logging
import os
import urllib.request

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, Label, ListView, Static

from rag.config import (
    DEFAULT_EMBEDDING_MODELS,
    EmbeddingProviderName,
    VectorStoreName,
    infer_embedding_dimension,
)
from rag.embeddings.base import EmbeddingModelMetadata
from rag.embeddings.factory import create_embedding_provider
from rag.models.movie import Movie
from rag.vectorstore.factory import create_vector_store
from rag.vectorstore.naming import resolve_collection_name
from rag.vectorstore.qdrant_vectorstore import QdrantVectorStore
from tui.constants import (
    MESSAGE_BAR_ID,
    OVERLAY_ID,
    OVERLAY_LIST_ID,
    PROVIDER_MODELS,
    PROVIDERS,
    RESULTS_LIST_ID,
    SEARCH_INPUT_ID,
    TOP_COMMANDS,
    TOP_K_OPTIONS,
    VECTOR_STORES,
    OverlayItem,
)
from tui.widgets import CommandOverlay, MovieCard, StatusBar

logger = logging.getLogger(__name__)


_HELP_TEXT = """\
[bold]Movie Finder RAG — Retrieval TUI[/bold]

[dim]Search:[/dim]  Type a query and press [bold]Enter[/bold].

[dim]Commands:[/dim]  Type [bold]/[/bold] to open the command overlay.
  [bold]↓[/bold] to enter the list · [bold]↑↓[/bold] to navigate · [bold]Enter[/bold] to select
  [bold]Esc[/bold] to dismiss at any level

[dim]Status line:[/dim]  Always visible — shows provider · model · store · collection · k

[dim]Keys:[/dim]  Ctrl+Q quit · Ctrl+L clear results\
"""


class RetrievalApp(App[None]):
    """Movie Finder RAG — Interactive Retrieval TUI."""

    TITLE = "Movie Finder RAG"
    COMMANDS = set()  # type: ignore[assignment]  # disable default command palette

    CSS = """
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

    #command-overlay {
        margin: 0 1;
    }

    #search-input {
        margin: 0 1 0 1;
    }

    #message-bar {
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_results", "Clear"),
        Binding("escape", "dismiss_overlay", "Dismiss", show=False),
        Binding("down", "focus_overlay", "↓ select", show=False),
        Binding("tab", "accept_first", "Complete", show=False),
    ]

    _provider: reactive[EmbeddingProviderName] = reactive("openai")
    _model: reactive[str] = reactive(DEFAULT_EMBEDDING_MODELS["openai"])
    _vector_store: reactive[VectorStoreName] = reactive("qdrant")
    _top_k: reactive[int] = reactive(5)

    # Overlay state machine: "hidden" | "commands" | "provider" | "model" | "store" | "topk"
    _overlay_mode: str = "hidden"
    _pending_provider: EmbeddingProviderName = "openai"

    def __init__(self) -> None:
        super().__init__()
        self._connections: dict[str, bool | None] = {}
        self._collection_name: str = ""

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    INITIAL_FOCUS = f"#{SEARCH_INPUT_ID}"

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="results-scroll"), Vertical(id=RESULTS_LIST_ID):
            yield Static(_HELP_TEXT, id="help-msg")
        yield CommandOverlay(id=OVERLAY_ID)
        yield Input(placeholder="Search movies… or type / for commands", id=SEARCH_INPUT_ID)
        yield Label("", id=MESSAGE_BAR_ID)
        yield StatusBar(
            provider=self._provider,
            model=self._model,
            store=self._vector_store,
            collection=self._collection_name,
            top_k=self._top_k,
        )
        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        # Prevent the results container from stealing focus on click
        self.query_one("#results-scroll").can_focus = False  # type: ignore[assignment]
        self.query_one(f"#{RESULTS_LIST_ID}").can_focus = False  # type: ignore[assignment]
        self.query_one(f"#{SEARCH_INPUT_ID}", Input).focus()
        self._check_connections()
        self._resolve_collection_name()

    # ------------------------------------------------------------------
    # Reactive watchers → keep StatusBar in sync
    # ------------------------------------------------------------------

    def watch__provider(self, value: EmbeddingProviderName) -> None:
        self._model = DEFAULT_EMBEDDING_MODELS[value]
        self._sync_status_bar()
        self._resolve_collection_name()

    def watch__model(self, _: str) -> None:
        self._sync_status_bar()
        self._resolve_collection_name()

    def watch__vector_store(self, _: VectorStoreName) -> None:
        self._sync_status_bar()
        self._resolve_collection_name()

    def watch__top_k(self, _: int) -> None:
        self._sync_status_bar()

    def _sync_status_bar(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one(StatusBar).refresh_status(
                self._provider,
                self._model,
                self._vector_store,
                self._collection_name,
                self._top_k,
                self._connections,
            )

    # ------------------------------------------------------------------
    # Static helper — used by tests
    # ------------------------------------------------------------------

    @staticmethod
    def _model_options_for(provider: EmbeddingProviderName) -> list[tuple[str, str]]:
        """Return (label, value) pairs for all models available under *provider*."""
        return [(m_id, m_id) for m_id, _ in PROVIDER_MODELS.get(provider, [])]

    # ------------------------------------------------------------------
    # Input → overlay state machine
    # ------------------------------------------------------------------

    @on(Input.Changed, f"#{SEARCH_INPUT_ID}")
    def on_search_changed(self, event: Input.Changed) -> None:
        text = event.value
        if text.startswith("/"):
            self._overlay_mode = "commands"
            needle = text.lower()
            filtered = [
                item
                for item in TOP_COMMANDS
                if needle in item.name.lower() or needle in item.description.lower()
            ]
            self.query_one(f"#{OVERLAY_ID}", CommandOverlay).populate(filtered)
        elif self._overlay_mode == "commands":
            # User deleted the slash — dismiss
            self._dismiss_overlay()

    @on(Input.Submitted, f"#{SEARCH_INPUT_ID}")
    def on_search_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        if query.startswith("/"):
            # Enter while overlay is visible: execute the first visible match
            overlay = self.query_one(f"#{OVERLAY_ID}", CommandOverlay)
            if overlay.display and overlay._items:
                self._handle_overlay_selection(overlay._items[0].item_id)
            return
        self._run_search(query)
        self.query_one(f"#{SEARCH_INPUT_ID}", Input).clear()

    # ------------------------------------------------------------------
    # Overlay list selection
    # ------------------------------------------------------------------

    @on(ListView.Selected, f"#{OVERLAY_LIST_ID}")
    def on_overlay_selected(self, event: ListView.Selected) -> None:
        """Dispatch the selected overlay item based on the current overlay mode."""
        overlay = self.query_one(f"#{OVERLAY_ID}", CommandOverlay)
        selected = overlay.item_at(event.list_view.index or 0)
        if selected is not None:
            self._handle_overlay_selection(selected.item_id)

    def _handle_overlay_selection(self, item_id: str) -> None:
        mode = self._overlay_mode

        if mode == "commands":
            self._dispatch_top_command(item_id)

        elif mode == "provider":
            provider = item_id  # item_id is the provider key (e.g. "openai")
            if provider in {p for _, p in PROVIDERS}:
                self._pending_provider = provider  # type: ignore[assignment]
                self._overlay_mode = "model"
                models = [
                    OverlayItem(m_id, m_id, desc)
                    for m_id, desc in PROVIDER_MODELS.get(provider, [])
                ]
                self.query_one(f"#{OVERLAY_ID}", CommandOverlay).populate(models)
                self.query_one(f"#{OVERLAY_ID}", CommandOverlay).focus_list()
            else:
                self._dismiss_overlay()

        elif mode == "model":
            model = item_id
            self._provider = self._pending_provider  # type: ignore[assignment]
            self._model = model
            self._set_message(f"provider → {self._provider}  model → {model}")
            self._dismiss_overlay()

        elif mode == "store":
            store = item_id
            if store in {s for _, s in VECTOR_STORES}:
                self._vector_store = store  # type: ignore[assignment]
                self._set_message(f"store → {store}")
            self._dismiss_overlay()

        elif mode == "topk":
            # item_id is "topk_N"
            with contextlib.suppress(ValueError):
                n = int(item_id.removeprefix("topk_"))
                self._top_k = n
                self._set_message(f"top-k → {n}")
            self._dismiss_overlay()

    def _dispatch_top_command(self, item_id: str) -> None:
        if item_id == "cmd_provider":
            self._overlay_mode = "provider"
            items = [
                OverlayItem(
                    p_id,
                    display,
                    PROVIDER_MODELS[p_id][0][0] if p_id in PROVIDER_MODELS else "",
                    "●" if self._connections.get(p_id) else "○",
                )
                for display, p_id in PROVIDERS
            ]
            self.query_one(f"#{OVERLAY_ID}", CommandOverlay).populate(items)
            self.query_one(f"#{OVERLAY_ID}", CommandOverlay).focus_list()

        elif item_id == "cmd_store":
            self._overlay_mode = "store"
            items = [
                OverlayItem(
                    s_id,
                    display,
                    "",
                    "●" if self._connections.get(s_id) else "○",
                )
                for display, s_id in VECTOR_STORES
            ]
            self.query_one(f"#{OVERLAY_ID}", CommandOverlay).populate(items)
            self.query_one(f"#{OVERLAY_ID}", CommandOverlay).focus_list()

        elif item_id == "cmd_topk":
            self._overlay_mode = "topk"
            items = [
                OverlayItem(f"topk_{n}", label, "results per search") for label, n in TOP_K_OPTIONS
            ]
            self.query_one(f"#{OVERLAY_ID}", CommandOverlay).populate(items)
            self.query_one(f"#{OVERLAY_ID}", CommandOverlay).focus_list()

        elif item_id == "cmd_backup":
            self._dismiss_overlay()
            self._run_backup()

        elif item_id == "cmd_help":
            self._dismiss_overlay()
            self._show_help()

        elif item_id == "cmd_quit":
            self.exit()

    # ------------------------------------------------------------------
    # Keyboard actions
    # ------------------------------------------------------------------

    def action_focus_overlay(self) -> None:
        """Move focus into the overlay list (Down key)."""
        overlay = self.query_one(f"#{OVERLAY_ID}", CommandOverlay)
        if overlay.display:
            overlay.focus_list()

    def action_accept_first(self) -> None:
        """Tab: execute the first visible overlay item, or move focus normally."""
        overlay = self.query_one(f"#{OVERLAY_ID}", CommandOverlay)
        if overlay.display and overlay._items:
            self._handle_overlay_selection(overlay._items[0].item_id)
        else:
            self.screen.focus_next()

    def action_dismiss_overlay(self) -> None:
        """Dismiss the overlay and return focus to the search input (Esc)."""
        self._dismiss_overlay()

    def action_clear_results(self) -> None:
        self._clear_results()
        self._set_message("Results cleared.")

    def _dismiss_overlay(self) -> None:
        self._overlay_mode = "hidden"
        overlay = self.query_one(f"#{OVERLAY_ID}", CommandOverlay)
        overlay.display = False
        with contextlib.suppress(Exception):
            inp = self.query_one(f"#{SEARCH_INPUT_ID}", Input)
            inp.clear()
            inp.focus()

    # ------------------------------------------------------------------
    # Background workers
    # ------------------------------------------------------------------

    @work(exclusive=True, thread=True)
    def _check_connections(self) -> None:
        """Probe each provider and store endpoint; update connection indicators."""
        results: dict[str, bool | None] = {}

        # Ollama — probe /api/tags; include Bearer token if OLLAMA_API_KEY is set
        ollama_base = (os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        ollama_key = os.getenv("OLLAMA_API_KEY", "")
        _ollama_req = urllib.request.Request(f"{ollama_base}/api/tags")  # noqa: S310
        if ollama_key:
            _ollama_req.add_header("Authorization", f"Bearer {ollama_key}")
        with contextlib.suppress(Exception):
            urllib.request.urlopen(_ollama_req, timeout=3)  # noqa: S310
            results["ollama"] = True
        if "ollama" not in results:
            results["ollama"] = False

        # Qdrant — cloud instances require an API key; local instances are probed via HTTP
        qdrant_url = os.getenv("QDRANT_URL", "")
        qdrant_key = os.getenv("QDRANT_API_KEY_RW") or os.getenv("QDRANT_API_KEY")
        _is_local = not qdrant_url or any(
            h in qdrant_url for h in ("localhost", "127.0.0.1", "::1")
        )
        if _is_local:
            probe = qdrant_url or "http://localhost:6333"
            with contextlib.suppress(Exception):
                urllib.request.urlopen(probe, timeout=2)  # noqa: S310
                results["qdrant"] = True
            if "qdrant" not in results:
                results["qdrant"] = False
        else:
            # Cloud Qdrant: reachability requires both a URL and an API key
            results["qdrant"] = bool(qdrant_url and qdrant_key)

        # API-key based providers — check env var presence
        results["openai"] = bool(os.getenv("OPENAI_API_KEY"))
        results["huggingface"] = bool(os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN"))
        results["google"] = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        results["sentence-transformers"] = True  # local, always available
        results["chromadb"] = True  # local by default
        results["pinecone"] = bool(os.getenv("PINECONE_API_KEY"))
        results["pgvector"] = bool(os.getenv("PGVECTOR_DSN") or os.getenv("DATABASE_URL"))

        self.call_from_thread(self._apply_connections, results)

    def _apply_connections(self, results: dict[str, bool | None]) -> None:
        self._connections = results
        self._sync_status_bar()

    def _resolve_collection_name(self) -> None:
        """Compute the target collection name using the same naming functions as the vectorstore.

        Uses resolve_collection_name() + sanitize_collection_token() directly — no provider
        or store instantiation required, so it never fails due to missing provider SDKs.
        """
        from rag.config import settings as _settings

        dim = infer_embedding_dimension(self._provider, self._model)
        model_info = EmbeddingModelMetadata(name=self._model, dimension=dim, cost_per_1k_tokens=0.0)
        name = resolve_collection_name(_settings.qdrant_collection_prefix, model_info)
        self._apply_collection_name(name)

    def _apply_collection_name(self, name: str) -> None:
        self._collection_name = name
        self._sync_status_bar()

    @work(exclusive=True, thread=True)
    def _run_search(self, query: str) -> None:
        self.call_from_thread(self._set_message, f"Searching  k={self._top_k}…")
        try:
            provider = create_embedding_provider(provider_name=self._provider, model=self._model)
            store = create_vector_store(vector_store_name=self._vector_store)
            model_info = provider.model_info
            vector = provider.embed(query)
            if not vector:
                self.call_from_thread(self._set_message, "Embedding failed — check credentials.")
                return

            # For Qdrant: use client.query_points() directly to get per-result scores,
            # the same way evaluate_qdrant_collections.py does it.
            if isinstance(store, QdrantVectorStore):
                collection_name = store.target_name(model_info)
                response = store.client.query_points(
                    collection_name=collection_name,
                    query=vector,
                    with_payload=True,
                    limit=self._top_k,
                )
                pairs: list[tuple[Movie, float | None]] = [
                    (Movie(**point.payload), point.score)
                    for point in response.points
                    if point.payload
                ]
            else:
                movies = store.search(vector, top_k=self._top_k, embedding_model=model_info)
                pairs = [(m, None) for m in movies]

            self.call_from_thread(self._display_results, pairs, query)
        except Exception as exc:
            logger.exception("Search worker error")
            self.call_from_thread(self._set_message, f"Error: {exc}")

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
        with contextlib.suppress(Exception):
            self.query_one(f"#{MESSAGE_BAR_ID}", Label).update(message)

    def _show_help(self) -> None:
        self._clear_results()
        results_panel = self.query_one(f"#{RESULTS_LIST_ID}", Vertical)
        results_panel.mount(Static(_HELP_TEXT))

    def _clear_results(self) -> None:
        results_panel = self.query_one(f"#{RESULTS_LIST_ID}", Vertical)
        for child in list(results_panel.children):
            child.remove()

    def _display_results(self, pairs: list[tuple[Movie, float | None]], query: str) -> None:
        self._clear_results()
        results_panel = self.query_one(f"#{RESULTS_LIST_ID}", Vertical)
        if not pairs:
            results_panel.mount(Static("No results found."))
            self._set_message(f"No results for: {query[:80]}")
            return
        cards = [
            MovieCard(i, movie, score=score) for i, (movie, score) in enumerate(pairs, start=1)
        ]
        results_panel.mount(*cards)
        self._set_message(f"{len(pairs)} result(s) for: {query[:80]}")
