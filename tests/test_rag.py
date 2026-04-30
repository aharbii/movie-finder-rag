from pytest import MonkeyPatch

from rag.config import settings
from rag.embeddings.base import EmbeddingModelMetadata
from rag.vectorstore.qdrant_vectorstore import QdrantVectorStore


def test_qdrant_vectorstore_uses_pydantic_settings(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "qdrant_url", "https://qdrant.example")
    monkeypatch.setattr(settings, "qdrant_api_key_rw", "rw-test-key")
    monkeypatch.setattr(settings, "vector_collection_prefix", "movies")

    captured: dict[str, str] = {}

    class DummyClient:
        def __init__(self, url: str, api_key: str) -> None:
            captured["url"] = url
            captured["api_key"] = api_key

        def collection_exists(self, collection_name: str) -> bool:
            return True

        def count(self, collection_name: str, exact: bool) -> object:
            class Result:
                count = 1

            return Result()

    monkeypatch.setattr("rag.vectorstore.qdrant_vectorstore.QdrantClient", DummyClient)

    store = QdrantVectorStore()
    assert captured == {"url": "https://qdrant.example", "api_key": "rw-test-key"}

    model = EmbeddingModelMetadata(name="BAAI/bge-m3", dimension=1024)
    assert store.target_name(model) == "movies_baai_bge_m3_1024"
