import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag.config import settings
from rag.embeddings.gemini_provider import GeminiEmbeddingProvider
from rag.embeddings.ollama_provider import OllamaEmbeddingProvider
from rag.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag.embeddings.sentence_transformers_provider import SentenceTransformersEmbeddingProvider


def test_openai_provider_passes_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "embedding_dimensions", 1024)

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
    mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

    with patch("rag.embeddings.openai_provider.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.return_value = mock_response
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")
        assert provider.embed("test text") == [0.1, 0.2]
        mock_openai.return_value.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large",
            input=["test text"],
            dimensions=1024,
        )


def test_google_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "google_api_key", None)
    monkeypatch.setattr(settings, "embedding_dimensions", None)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY is not defined"):
        GeminiEmbeddingProvider(model="text-embedding-004")


def test_ollama_provider_uses_host_and_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimensions", 384)
    monkeypatch.setattr(settings, "ollama_url", "https://ollama.example")
    monkeypatch.setattr(settings, "ollama_api_key", "ollama-key")

    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": [[0.1, 0.2]], "prompt_eval_count": 5}

    fake_ollama = SimpleNamespace(Client=MagicMock(return_value=mock_client))
    with patch.dict(sys.modules, {"ollama": fake_ollama}):
        provider = OllamaEmbeddingProvider(model="embeddinggemma")
        assert provider.embed("test text") == [0.1, 0.2]
        fake_ollama.Client.assert_called_once_with(
            host="https://ollama.example",
            headers={"Authorization": "Bearer ollama-key"},
        )
        mock_client.embed.assert_called_once_with(
            model="embeddinggemma",
            input=["test text"],
            dimensions=384,
        )


def test_sentence_transformers_provider_uses_truncate_dim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embedding_dimensions", 256)
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 1024
    mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2]]

    fake_sentence_transformers = SimpleNamespace(
        SentenceTransformer=MagicMock(return_value=mock_model)
    )
    with patch.dict(sys.modules, {"sentence_transformers": fake_sentence_transformers}):
        provider = SentenceTransformersEmbeddingProvider(model="BAAI/bge-m3")
        assert provider.model_info.dimension == 256
        assert provider.embed("hello") == [0.1, 0.2]
        mock_model.encode.assert_called_once_with(
            ["hello"],
            convert_to_numpy=True,
            truncate_dim=256,
        )
