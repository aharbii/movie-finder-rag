import json
import logging

import pytest

from rag.utils.logger import _JsonFormatter, configure_logging, get_logger


def test_get_logger() -> None:
    logger = get_logger("test_name")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_name"


def test_json_formatter() -> None:
    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    result = formatter.format(record)
    data = json.loads(result)
    assert data["level"] == "INFO"
    assert data["logger"] == "test_logger"
    assert data["message"] == "Test message"
    assert "timestamp" in data
    assert "exception" not in data


def test_json_formatter_with_exception() -> None:
    formatter = _JsonFormatter()
    try:
        raise ValueError("Oops")
    except ValueError:
        import sys

        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="Error message",
        args=(),
        exc_info=exc_info,
    )
    result = formatter.format(record)
    data = json.loads(result)
    assert data["level"] == "ERROR"
    assert data["message"] == "Error message"
    assert "exception" in data
    assert "ValueError: Oops" in data["exception"]


def test_configure_logging_json(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = logging.getLogger("rag")
    # Clean up handlers for idempotency test
    logger.handlers.clear()

    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    configure_logging()

    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0].formatter, _JsonFormatter)
    assert logger.level == logging.DEBUG

    # Test idempotency
    configure_logging()
    assert len(logger.handlers) == 1


def test_configure_logging_text(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = logging.getLogger("rag")
    logger.handlers.clear()

    monkeypatch.setenv("LOG_FORMAT", "text")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    configure_logging()

    assert len(logger.handlers) == 1
    assert not isinstance(logger.handlers[0].formatter, _JsonFormatter)
    assert logger.level == logging.INFO
