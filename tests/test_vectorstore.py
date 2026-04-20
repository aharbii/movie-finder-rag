from unittest.mock import MagicMock, patch

import pytest

from rag.config import settings
from rag.embeddings.base import EmbeddingModelMetadata
from rag.models.movie import Movie
from rag.vectorstore.chromadb_vectorstore import ChromaDBVectorStore
from rag.vectorstore.naming import resolve_collection_name, sanitize_collection_token
from rag.vectorstore.qdrant_vectorstore import QdrantVectorStore, QdrantVectorStoreError


@pytest.fixture
def sample_movie() -> Movie:
    return Movie(
        id=1,
        title="Test Movie",
        release_year=2024,
        director="Director",
        genre=["Action"],
        cast=["Actor"],
        plot="A test plot.",
    )


@pytest.fixture
def model_metadata() -> EmbeddingModelMetadata:
    return EmbeddingModelMetadata(name="BAAI/bge-m3", dimension=1024)


def test_sanitize_collection_token() -> None:
    assert sanitize_collection_token("BAAI/bge-m3") == "baai_bge_m3"


def test_resolve_collection_name(model_metadata: EmbeddingModelMetadata) -> None:
    assert resolve_collection_name("movies", model_metadata) == "movies_baai_bge_m3_1024"


def test_chromadb_uses_dynamic_target_name(
    sample_movie: Movie,
    model_metadata: EmbeddingModelMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "qdrant_collection_prefix", "movies")
    monkeypatch.setattr(settings, "chromadb_persist_path", "outputs/chromadb-test")

    with patch("chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        store = ChromaDBVectorStore()
        store.upsert(sample_movie, [0.1, 0.2, 0.3], model_metadata)

        mock_client.return_value.get_or_create_collection.assert_called_once_with(
            name="movies_baai_bge_m3_1024"
        )


def test_qdrant_validate_env_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "qdrant_api_key_rw", None)
    monkeypatch.setattr(settings, "qdrant_url", None)

    with pytest.raises(QdrantVectorStoreError, match="QDRANT_API_KEY_RW is not defined"):
        QdrantVectorStore()


def test_qdrant_uses_dynamic_collection_name(
    sample_movie: Movie,
    model_metadata: EmbeddingModelMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "qdrant_url", "http://test")
    monkeypatch.setattr(settings, "qdrant_api_key_rw", "key")
    monkeypatch.setattr(settings, "qdrant_collection_prefix", "movies")

    with patch("rag.vectorstore.qdrant_vectorstore.QdrantClient") as mock_client:
        mock_client.return_value.collection_exists.return_value = True
        store = QdrantVectorStore()
        store.upsert(sample_movie, [0.1, 0.2, 0.3], model_metadata)

        mock_client.return_value.upsert.assert_called_once()
        called_collection = mock_client.return_value.upsert.call_args.kwargs["collection_name"]
        assert called_collection == "movies_baai_bge_m3_1024"
