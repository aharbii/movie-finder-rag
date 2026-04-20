from unittest.mock import patch

from pytest import MonkeyPatch

from rag.embeddings.factory import create_embedding_provider, get_embedding_provider
from rag.vectorstore.factory import create_vector_store, get_vector_store


def test_embedding_factory_dispatch(monkeypatch: MonkeyPatch) -> None:
    from rag.embeddings.openai_provider import OpenAIEmbeddingProvider

    monkeypatch.setattr("rag.embeddings.factory.settings.embedding_provider", "openai")
    monkeypatch.setattr("rag.embeddings.factory.settings.embedding_model", "text-embedding-3-large")
    monkeypatch.setattr("rag.embeddings.openai_provider.settings.openai_api_key", "test-key")

    with patch("rag.embeddings.openai_provider.OpenAI"):
        get_embedding_provider.cache_clear()
        provider = create_embedding_provider("openai", "text-embedding-3-large")
        assert isinstance(provider, OpenAIEmbeddingProvider)


def test_embedding_factory_is_cached(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("rag.embeddings.factory.settings.embedding_provider", "openai")
    monkeypatch.setattr("rag.embeddings.factory.settings.embedding_model", "text-embedding-3-large")
    monkeypatch.setattr("rag.embeddings.openai_provider.settings.openai_api_key", "test-key")

    with patch("rag.embeddings.openai_provider.OpenAI"):
        get_embedding_provider.cache_clear()
        first = get_embedding_provider()
        second = get_embedding_provider()
        assert first is second


def test_vector_store_factory_dispatch(monkeypatch: MonkeyPatch) -> None:
    from rag.vectorstore.qdrant_vectorstore import QdrantVectorStore

    monkeypatch.setattr("rag.vectorstore.factory.settings.vector_store", "qdrant")
    monkeypatch.setattr("rag.vectorstore.qdrant_vectorstore.settings.qdrant_url", "http://test")
    monkeypatch.setattr("rag.vectorstore.qdrant_vectorstore.settings.qdrant_api_key_rw", "key")

    with patch("rag.vectorstore.qdrant_vectorstore.QdrantClient"):
        get_vector_store.cache_clear()
        store = create_vector_store("qdrant")
        assert isinstance(store, QdrantVectorStore)
