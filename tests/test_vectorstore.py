import sys
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from rag.config import settings
from rag.embeddings.base import EmbeddingModelMetadata
from rag.models.movie import Movie
from rag.vectorstore.chromadb_vectorstore import ChromaDBVectorStore
from rag.vectorstore.naming import resolve_collection_name, sanitize_collection_token
from rag.vectorstore.pgvector_vectorstore import PGVectorStore, cast_payload
from rag.vectorstore.pinecone_vectorstore import PineconeVectorStore
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


# --- Naming Tests ---
def test_sanitize_collection_token() -> None:
    assert sanitize_collection_token("BAAI/bge-m3") == "baai_bge_m3"


def test_resolve_collection_name(model_metadata: EmbeddingModelMetadata) -> None:
    assert resolve_collection_name("movies", model_metadata) == "movies_baai_bge_m3_1024"


# --- ChromaDB Tests ---
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
        mock_collection.upsert.assert_called_once()


def test_chromadb_count(
    model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "chromadb_persist_path", "test")
    with patch("chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        store = ChromaDBVectorStore()
        assert store.count(model_metadata) == 5


def test_chromadb_search_empty(
    model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "chromadb_persist_path", "test")
    with patch("chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_collection.query.return_value = {"metadatas": []}
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        store = ChromaDBVectorStore()
        assert store.search([0.1], 5, model_metadata) == []


def test_chromadb_search_populated(
    sample_movie: Movie, model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "chromadb_persist_path", "test")
    with patch("chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_collection.query.return_value = {"metadatas": [[sample_movie.model_dump()]]}
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        store = ChromaDBVectorStore()
        results = store.search([0.1], 5, model_metadata)
        assert len(results) == 1
        assert results[0].id == sample_movie.id


# --- Qdrant Tests ---
def test_qdrant_validate_env_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "qdrant_api_key_rw", None)
    monkeypatch.setattr(settings, "qdrant_url", None)
    with pytest.raises(QdrantVectorStoreError, match="QDRANT_API_KEY_RW is not defined"):
        QdrantVectorStore()


def test_qdrant_validate_env_error_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "qdrant_api_key_rw", "key")
    monkeypatch.setattr(settings, "qdrant_url", None)
    with pytest.raises(QdrantVectorStoreError, match="QDRANT_URL is not defined"):
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


def test_qdrant_count_exists(
    model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "qdrant_url", "http://test")
    monkeypatch.setattr(settings, "qdrant_api_key_rw", "key")
    with patch("rag.vectorstore.qdrant_vectorstore.QdrantClient") as mock_client:
        mock_client.return_value.collection_exists.return_value = True
        mock_client.return_value.count.return_value = MagicMock(count=10)
        store = QdrantVectorStore()
        assert store.count(model_metadata) == 10


def test_qdrant_count_not_exists(
    model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "qdrant_url", "http://test")
    monkeypatch.setattr(settings, "qdrant_api_key_rw", "key")
    with patch("rag.vectorstore.qdrant_vectorstore.QdrantClient") as mock_client:
        mock_client.return_value.collection_exists.return_value = False
        store = QdrantVectorStore()
        assert store.count(model_metadata) == 0


def test_qdrant_search(
    sample_movie: Movie, model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "qdrant_url", "http://test")
    monkeypatch.setattr(settings, "qdrant_api_key_rw", "key")
    with patch("rag.vectorstore.qdrant_vectorstore.QdrantClient") as mock_client:
        mock_client.return_value.collection_exists.return_value = False  # should create
        mock_client.return_value.query_points.return_value = MagicMock(
            points=[MagicMock(payload=sample_movie.model_dump())]
        )
        store = QdrantVectorStore()
        results = store.search([0.1], 5, model_metadata)
        assert len(results) == 1
        assert results[0].id == sample_movie.id
        mock_client.return_value.create_collection.assert_called_once()


# --- PGVector Tests ---
def test_pgvector_missing_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "pgvector_dsn", None)
    with pytest.raises(ValueError, match="PGVECTOR_DSN is not defined"):
        PGVectorStore()


def test_pgvector_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "pgvector_dsn", "postgres://test")
    with (
        patch.dict(sys.modules, {"psycopg": None}),
        pytest.raises(RuntimeError, match="PGVector support requires"),
    ):
        PGVectorStore()


def test_pgvector_count(
    model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "pgvector_dsn", "postgres://test")
    mock_psycopg = MagicMock()
    mock_register = MagicMock()
    with patch.dict(
        sys.modules,
        {"psycopg": mock_psycopg, "pgvector.psycopg": MagicMock(register_vector=mock_register)},
    ):
        store = PGVectorStore()
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_cursor.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = [42]
        store._psycopg.connect.return_value = mock_conn

        assert store.count(model_metadata) == 42
        mock_cursor.execute.assert_called_with(
            'SELECT COUNT(*) FROM "public_movies_baai_bge_m3_1024"'
        )


def test_pgvector_count_empty(
    model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "pgvector_dsn", "postgres://test")
    mock_psycopg = MagicMock()
    mock_register = MagicMock()
    with patch.dict(
        sys.modules,
        {"psycopg": mock_psycopg, "pgvector.psycopg": MagicMock(register_vector=mock_register)},
    ):
        store = PGVectorStore()
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_cursor.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        store._psycopg.connect.return_value = mock_conn

        assert store.count(model_metadata) == 0


def test_pgvector_upsert(
    sample_movie: Movie, model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "pgvector_dsn", "postgres://test")
    mock_psycopg = MagicMock()
    mock_register = MagicMock()
    with patch.dict(
        sys.modules,
        {"psycopg": mock_psycopg, "pgvector.psycopg": MagicMock(register_vector=mock_register)},
    ):
        store = PGVectorStore()
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_cursor.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value = mock_cursor
        store._psycopg.connect.return_value = mock_conn

        store.upsert(sample_movie, [0.1], model_metadata)
        assert mock_cursor.execute.call_count >= 3  # extension, table, insert
        mock_conn.commit.assert_called()


def test_pgvector_search(
    sample_movie: Movie, model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "pgvector_dsn", "postgres://test")
    mock_psycopg = MagicMock()
    mock_register = MagicMock()
    with patch.dict(
        sys.modules,
        {"psycopg": mock_psycopg, "pgvector.psycopg": MagicMock(register_vector=mock_register)},
    ):
        store = PGVectorStore()
        mock_conn = MagicMock()
        mock_conn.__enter__.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_cursor.__enter__.return_value = mock_cursor
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [[sample_movie.model_dump()]]
        store._psycopg.connect.return_value = mock_conn

        results = store.search([0.1], 5, model_metadata)
        assert len(results) == 1
        assert results[0].id == sample_movie.id


def test_cast_payload_str() -> None:

    assert cast_payload('{"a": 1}') == {"a": 1}


def test_cast_payload_dict() -> None:
    assert cast_payload({"a": 1}) == {"a": 1}


# --- Pinecone Tests ---
def test_pinecone_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "pinecone_api_key", None)
    with pytest.raises(ValueError, match="PINECONE_API_KEY is not defined"):
        PineconeVectorStore()


def test_pinecone_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "pinecone_api_key", "test")
    with (
        patch.dict(sys.modules, {"pinecone": None}),
        pytest.raises(RuntimeError, match="Pinecone support requires"),
    ):
        PineconeVectorStore()


def test_pinecone_count(
    model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "pinecone_api_key", "test")
    mock_pinecone = MagicMock()
    with patch.dict(
        sys.modules, {"pinecone": MagicMock(Pinecone=mock_pinecone, ServerlessSpec=MagicMock())}
    ):
        store = PineconeVectorStore()
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = {
            "namespaces": {"movies_baai_bge_m3_1024": {"vector_count": 99}}
        }
        store._index = mock_index
        assert store.count(model_metadata) == 99


def test_pinecone_count_no_namespace(
    model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "pinecone_api_key", "test")
    mock_pinecone = MagicMock()
    with patch.dict(
        sys.modules, {"pinecone": MagicMock(Pinecone=mock_pinecone, ServerlessSpec=MagicMock())}
    ):
        store = PineconeVectorStore()
        mock_index = MagicMock()
        mock_index.describe_index_stats.return_value = {"namespaces": {}}
        store._index = mock_index
        assert store.count(model_metadata) == 0


def test_pinecone_upsert(
    sample_movie: Movie, model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "pinecone_api_key", "test")
    mock_pinecone = MagicMock()
    with patch.dict(
        sys.modules, {"pinecone": MagicMock(Pinecone=mock_pinecone, ServerlessSpec=MagicMock())}
    ):
        store = PineconeVectorStore()
        mock_index = MagicMock()
        store._index = mock_index
        store.upsert(sample_movie, [0.1], model_metadata)
        mock_index.upsert.assert_called_once()


def test_pinecone_search(
    sample_movie: Movie, model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "pinecone_api_key", "test")
    mock_pinecone = MagicMock()
    with patch.dict(
        sys.modules, {"pinecone": MagicMock(Pinecone=mock_pinecone, ServerlessSpec=MagicMock())}
    ):
        store = PineconeVectorStore()
        mock_index = MagicMock()
        mock_index.query.return_value = {"matches": [{"metadata": sample_movie.model_dump()}]}
        store._index = mock_index
        results = store.search([0.1], 5, model_metadata)
        assert len(results) == 1
        assert results[0].id == sample_movie.id


def test_pinecone_get_index_no_host(
    model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "pinecone_api_key", "test")
    monkeypatch.setattr(settings, "pinecone_index_host", None)
    mock_pinecone = MagicMock()
    with patch.dict(
        sys.modules, {"pinecone": MagicMock(Pinecone=mock_pinecone, ServerlessSpec=MagicMock())}
    ):
        store = PineconeVectorStore()
        cast(MagicMock, store.client.has_index).return_value = False
        cast(MagicMock, store.client.describe_index).return_value = {"host": "my-host"}

        index = store._get_index(model_metadata)
        assert index is not None
        assert store.index_host == "my-host"


def test_pinecone_ensure_index_exists(
    model_metadata: EmbeddingModelMetadata, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "pinecone_api_key", "test")
    mock_pinecone = MagicMock()
    with patch.dict(
        sys.modules, {"pinecone": MagicMock(Pinecone=mock_pinecone, ServerlessSpec=MagicMock())}
    ):
        store = PineconeVectorStore()
        cast(MagicMock, store.client.has_index).return_value = True

        store._ensure_index(model_metadata)

        cast(MagicMock, store.client.create_index).assert_not_called()
