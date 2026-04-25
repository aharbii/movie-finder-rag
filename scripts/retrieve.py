"""Entry point for the Retrieval TUI.

Launch via:
    make retrieve
    # or directly:
    python scripts/retrieve.py
"""

from rag.tui import RetrievalApp


def main() -> None:
    """Launch the Retrieval TUI application."""
    RetrievalApp().run()


if __name__ == "__main__":
    main()
