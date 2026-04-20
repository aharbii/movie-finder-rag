import pytest

from rag.config import DEFAULT_EMBEDDING_MODELS, RAGConfig


def apply_base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "chromadb")
    monkeypatch.setenv("CHROMADB_PERSIST_PATH", "outputs/chromadb-test")


def test_provider_default_model_is_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    config = RAGConfig()
    assert config.embedding_model == DEFAULT_EMBEDDING_MODELS["huggingface"]


def test_collection_prefix_normalizes_legacy_value(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.delenv("QDRANT_COLLECTION_PREFIX", raising=False)
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "text-embedding-3-large")

    config = RAGConfig()
    assert config.qdrant_collection_prefix == "text_embedding_3_large"


def test_embedding_dimension_uses_override(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")

    config = RAGConfig()
    assert config.embedding_dimension == 1024


def test_shape_validator_requires_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        RAGConfig()


def test_shape_validator_requires_qdrant_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("QDRANT_API_KEY_RW", "")

    with pytest.raises(ValueError, match="QDRANT_URL is required"):
        RAGConfig()


def test_shape_validator_rejects_google_dimension_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "256")

    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS is not supported"):
        RAGConfig()


def test_shape_validator_requires_ollama_url(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_URL", raising=False)

    with pytest.raises(ValueError, match="OLLAMA_URL is required"):
        RAGConfig()
