import os

# Keep pytest isolated from the developer's local runtime configuration.
# The application still defaults to `.env` outside tests.
os.environ.setdefault("RAG_ENV_FILE", ".env.pytest-do-not-load")
