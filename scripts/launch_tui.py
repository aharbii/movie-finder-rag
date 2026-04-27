"""Entry point for the Retrieval TUI.

Launch via:
    make tui
    # or directly:
    python scripts/launch_tui.py
"""

import sys
from pathlib import Path

# tui/ lives at the rag root, one level above scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from tui.app import RetrievalApp


def main() -> None:
    """Launch the interactive Retrieval TUI application."""
    RetrievalApp().run()


if __name__ == "__main__":
    main()
