from unittest.mock import MagicMock, patch

import pytest

from rag.config import settings
from rag.embeddings.base import EmbeddingModelMetadata
from rag.models.movie import Movie
from rag.vectorstore.chromadb_vectorstore import ChromaDBVectorStore
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
    return EmbeddingModelMetadata(name="test-model", dimension=3)


def test_chromadb_upsert(sample_movie: Movie, model_metadata: EmbeddingModelMetadata) -> None:
    with patch("chromadb.Client") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = ChromaDBVectorStore()
        store.upsert(sample_movie, [0.1, 0.2, 0.3], model_metadata)

        mock_collection.add.assert_called_once()


def test_chromadb_upsert_batch(sample_movie: Movie, model_metadata: EmbeddingModelMetadata) -> None:
    with patch("chromadb.Client") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = ChromaDBVectorStore()
        store.upsert_batch([sample_movie], [[0.1, 0.2, 0.3]], model_metadata)

        mock_collection.add.assert_called_once()


def test_chromadb_search(model_metadata: EmbeddingModelMetadata) -> None:
    with patch("chromadb.Client") as mock_client:
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "metadatas": [
                [
                    {
                        "id": 1,
                        "title": "T",
                        "release_year": 2000,
                        "director": "D",
                        "genre": ["G"],
                        "cast": ["C"],
                        "plot": "P",
                    }
                ]
            ]
        }
        mock_client.return_value.get_or_create_collection.return_value = mock_collection

        store = ChromaDBVectorStore()
        results = store.search([0.1, 0.2, 0.3], top_k=1, embedding_model=model_metadata)

        assert len(results) == 1
        assert results[0].title == "T"


def test_qdrant_validate_env_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "qdrant_api_key_rw", "")
    with pytest.raises(QdrantVectorStoreError, match="QDRANT_API_KEY_RW is not defined"):
        QdrantVectorStore()


def test_qdrant_upsert(
    sample_movie: Movie, model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "qdrant_url", "http://test")
    monkeypatch.setattr(settings, "qdrant_api_key_rw", "key")

    with patch("rag.vectorstore.qdrant_vectorstore.QdrantClient") as mock_client:
        mock_client.return_value.collection_exists.return_value = True
        store = QdrantVectorStore()
        store.upsert(sample_movie, [0.1, 0.2, 0.3], model_metadata)
        mock_client.return_value.upsert.assert_called_once()


def test_qdrant_upsert_creates_collection(
    sample_movie: Movie, model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "qdrant_url", "http://test")
    monkeypatch.setattr(settings, "qdrant_api_key_rw", "key")

    with patch("rag.vectorstore.qdrant_vectorstore.QdrantClient") as mock_client:
        mock_client.return_value.collection_exists.return_value = False
        store = QdrantVectorStore()
        store.upsert(sample_movie, [0.1, 0.2, 0.3], model_metadata)
        mock_client.return_value.create_collection.assert_called_once()


def test_qdrant_search(
    model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "qdrant_url", "http://test")
    monkeypatch.setattr(settings, "qdrant_api_key_rw", "key")

    with patch("rag.vectorstore.qdrant_vectorstore.QdrantClient") as mock_client:
        mock_client.return_value.collection_exists.return_value = True
        mock_results = MagicMock()
        mock_point = MagicMock()
        mock_point.payload = {
            "id": 1,
            "title": "T",
            "release_year": 2000,
            "director": "D",
            "genre": ["G"],
            "cast": ["C"],
            "plot": "P",
        }
        mock_results.points = [mock_point]
        mock_client.return_value.query_points.return_value = mock_results

        store = QdrantVectorStore()
        results = store.search([0.1, 0.2, 0.3], top_k=1, embedding_model=model_metadata)
        assert len(results) == 1
