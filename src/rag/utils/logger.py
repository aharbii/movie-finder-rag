"""Logging utilities for the rag_ingestion package.

Library code never configures the logging system — it only obtains loggers.
Configuration is the responsibility of the entry point that calls
``configure_logging()`` before running the ingestion pipeline.

Log level is controlled by the ``LOG_LEVEL`` environment variable at the
entry-point level — never per-logger.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime


def get_logger(name: str) -> logging.Logger:
    """Return a stdlib logger for the given name.

    Args:
        name: Logger name, typically ``__name__`` or ``self.__class__.__name__``.

    Returns:
        A ``logging.Logger`` instance.
    """
    return logging.getLogger(name)


class _JsonFormatter(logging.Formatter):
    """JSON formatter for structured log output."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record to a single-line JSON string.

        Args:
            record: The log record to format.

        Returns:
            A JSON-encoded log entry string.
        """
        data: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False)


def configure_logging() -> None:
    """Bootstrap logging for the rag_ingestion pipeline.

    Reads ``LOG_LEVEL`` (default ``INFO``) and ``LOG_FORMAT`` (default ``text``)
    from the environment.  Idempotent — safe to call multiple times.

    No file handlers are created — output goes to stdout only, which is
    appropriate for both interactive runs and containerised CI execution.
    """
    if logging.getLogger("rag").handlers:
        return  # already configured

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("LOG_FORMAT", "text").lower()
    level: int = getattr(logging, level_name, logging.INFO)

    if log_format == "json":
        formatter: logging.Formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    rag_logger = logging.getLogger("rag")
    rag_logger.setLevel(level)
    rag_logger.addHandler(handler)
    rag_logger.propagate = False

    _quiet_libs = ("httpx", "httpcore", "openai")
    quiet_level = logging.DEBUG if level == logging.DEBUG else logging.WARNING
    for lib in _quiet_libs:
        logging.getLogger(lib).setLevel(quiet_level)
