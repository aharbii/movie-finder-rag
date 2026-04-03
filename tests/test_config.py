import pytest

from rag.config import RAGConfig


def test_embedding_dimension_openai_large(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dimension for OpenAI large model."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost")
    monkeypatch.setenv("QDRANT_API_KEY_RW", "key")
    monkeypatch.setenv("OPENAI_API_KEY", "okey")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    config = RAGConfig()
    assert config.embedding_dimension == 3072


def test_embedding_dimension_openai_small(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dimension for OpenAI small model."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost")
    monkeypatch.setenv("QDRANT_API_KEY_RW", "key")
    monkeypatch.setenv("OPENAI_API_KEY", "okey")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    config = RAGConfig()
    assert config.embedding_dimension == 1536


def test_embedding_dimension_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dimension for Gemini provider."""
    monkeypatch.setenv("QDRANT_URL", "http://localhost")
    monkeypatch.setenv("QDRANT_API_KEY_RW", "key")
    monkeypatch.setenv("OPENAI_API_KEY", "okey")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")

    config = RAGConfig()
    assert config.embedding_dimension == 768
