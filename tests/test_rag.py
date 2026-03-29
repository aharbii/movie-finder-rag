from collections.abc import Sized
from typing import cast

from pytest import MonkeyPatch

from rag.embeddings.base import EmbeddingModelMetadata
from rag.models.movie import Movie
from rag.vectorstore.qdrant_vectorstore import QdrantVectorStore


def test_qdrant_vectorstore_uses_pydantic_settings(monkeypatch: MonkeyPatch) -> None:
    """
    Ensure the Qdrant client is correctly initialized from Pydantic settings.
    """
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY_RW", "rw-test-key")

    captured: dict[str, str] = {}

    class DummyClient:
        def __init__(self, url: str, api_key: str) -> None:
            captured["url"] = url
            captured["api_key"] = api_key

    # Patch the client import within the rag package
    monkeypatch.setattr("rag.vectorstore.qdrant_vectorstore.QdrantClient", DummyClient)

    QdrantVectorStore()

    assert captured == {
        "url": "https://qdrant.example",
        "api_key": "rw-test-key",
    }


def test_qdrant_vectorstore_batch_upsert(monkeypatch: MonkeyPatch) -> None:
    """
    Verify that batch upsert logic correctly calls the Qdrant SDK.
    """
    monkeypatch.setenv("QDRANT_URL", "https://qdrant.example")
    monkeypatch.setenv("QDRANT_API_KEY_RW", "rw-test-key")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "movies-test")

    captured: dict[str, object] = {}

    class DummyClient:
        def __init__(self, url: str, api_key: str) -> None:
            pass

        def collection_exists(self, collection_name: str) -> bool:
            return True

        def upsert(self, collection_name: str, points: list[object]) -> None:
            captured["upsert_collection"] = collection_name
            captured["points"] = points

    monkeypatch.setattr("rag.vectorstore.qdrant_vectorstore.QdrantClient", DummyClient)

    store = QdrantVectorStore()
    model = EmbeddingModelMetadata(name="test-model", dimension=3072)

    store.upsert_batch(
        movies=[
            Movie(
                id=1,
                title="Test Movie",
                release_year=2024,
                director="",
                genre="",
                cast="",
                plot="",
            )
        ],
        vectors=[[0.1, 0.2, 0.3]],
        embedding_model=model,
    )

    assert captured["upsert_collection"] == "movies-test"
    assert len(cast(Sized, captured["points"])) == 1
