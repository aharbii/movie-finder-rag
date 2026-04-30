import sys
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from rag.config import settings
from rag.embeddings.gemini_provider import GeminiEmbeddingProvider
from rag.embeddings.ollama_provider import OllamaEmbeddingProvider
from rag.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag.embeddings.sentence_transformers_provider import SentenceTransformersEmbeddingProvider


# --- GeminiEmbeddingProvider Tests ---
def test_google_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "google_api_key", None)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY is not defined"):
        GeminiEmbeddingProvider(model="text-embedding-004")


def test_google_provider_rejects_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "google_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", 1024)
    with pytest.raises(ValueError, match="GOOGLE embeddings do not support EMBEDDING_DIMENSION"):
        GeminiEmbeddingProvider(model="text-embedding-004")


def test_google_provider_model_info_known(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "google_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with patch("rag.embeddings.gemini_provider.genai.Client"):
        provider = GeminiEmbeddingProvider(model="text-embedding-004")
        info = provider.model_info
        assert info.dimension == 768


def test_google_provider_model_info_infer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "google_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with (
        patch("rag.embeddings.gemini_provider.genai.Client"),
        patch("rag.embeddings.gemini_provider.infer_embedding_dimension", return_value=128),
    ):
        provider = GeminiEmbeddingProvider(model="unknown-model")
        info = provider.model_info
        assert info.dimension == 128


def test_google_provider_model_info_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "google_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with (
        patch("rag.embeddings.gemini_provider.genai.Client"),
        patch("rag.embeddings.gemini_provider.infer_embedding_dimension", return_value=0),
    ):
        provider = GeminiEmbeddingProvider(model="unknown-model")
        with pytest.raises(ValueError, match="Unknown Google embedding dimension"):
            _ = provider.model_info


def test_google_provider_embed_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "google_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with patch("rag.embeddings.gemini_provider.genai.Client"):
        provider = GeminiEmbeddingProvider(model="text-embedding-004")
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2]
        mock_response.embeddings = [mock_embedding]
        cast(MagicMock, provider.client.models.embed_content).return_value = mock_response

        assert provider.embed("test text") == [0.1, 0.2]
        usage = provider.get_model_usage()
        assert usage.prompt_tokens == 1
        assert usage.total_tokens == 1


def test_google_provider_embed_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "google_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with patch("rag.embeddings.gemini_provider.genai.Client"):
        provider = GeminiEmbeddingProvider(model="text-embedding-004")
        mock_response = MagicMock()
        mock_response.embeddings = []
        cast(MagicMock, provider.client.models.embed_content).return_value = mock_response
        assert provider.embed("test text") == []


def test_google_provider_embed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "google_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with patch("rag.embeddings.gemini_provider.genai.Client"):
        provider = GeminiEmbeddingProvider(model="text-embedding-004")
        cast(MagicMock, provider.client.models.embed_content).side_effect = Exception("API error")
        assert provider.embed("test text") == []


# --- OllamaEmbeddingProvider Tests ---
def test_ollama_provider_missing_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", 384)
    with (
        patch.dict(sys.modules, {"ollama": None}),
        pytest.raises(RuntimeError, match="Ollama support requires"),
    ):
        OllamaEmbeddingProvider()


def test_ollama_provider_uses_host_and_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", 384)
    monkeypatch.setattr(settings, "ollama_base_url", "https://ollama.example")
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
        usage = provider.get_model_usage()
        assert usage.total_tokens == 5


def test_ollama_provider_model_info_from_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    fake_ollama = SimpleNamespace(Client=MagicMock(return_value=mock_client))
    with (
        patch.dict(sys.modules, {"ollama": fake_ollama}),
        patch("rag.embeddings.ollama_provider.infer_embedding_dimension", return_value=0),
    ):
        provider = OllamaEmbeddingProvider(model="test-model")
        assert provider.model_info.dimension == 3


def test_ollama_provider_model_info_probe_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    mock_client = MagicMock()
    mock_client.embed.side_effect = Exception("Probe failed")
    fake_ollama = SimpleNamespace(Client=MagicMock(return_value=mock_client))
    with (
        patch.dict(sys.modules, {"ollama": fake_ollama}),
        patch("rag.embeddings.ollama_provider.infer_embedding_dimension", return_value=0),
    ):
        provider = OllamaEmbeddingProvider(model="test-model")
        with pytest.raises(ValueError, match="Unknown Ollama embedding dimension"):
            _ = provider.model_info


def test_ollama_provider_embed_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", 384)
    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": []}
    fake_ollama = SimpleNamespace(Client=MagicMock(return_value=mock_client))
    with patch.dict(sys.modules, {"ollama": fake_ollama}):
        provider = OllamaEmbeddingProvider()
        assert provider.embed("test text") == []


def test_ollama_provider_embed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", 384)
    mock_client = MagicMock()
    mock_client.embed.side_effect = Exception("API Error")
    fake_ollama = SimpleNamespace(Client=MagicMock(return_value=mock_client))
    with patch.dict(sys.modules, {"ollama": fake_ollama}):
        provider = OllamaEmbeddingProvider()
        assert provider.embed("test text") == []


# --- OpenAIEmbeddingProvider Tests ---
def test_ollama_provider_model_info_skip_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", 1024)
    fake_ollama = SimpleNamespace(Client=MagicMock())
    with patch.dict(sys.modules, {"ollama": fake_ollama}):
        provider = OllamaEmbeddingProvider(model="test-model")
        assert provider.model_info.dimension == 1024


def test_ollama_provider_model_info_probe_empty_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": []}
    fake_ollama = SimpleNamespace(Client=MagicMock(return_value=mock_client))
    with (
        patch.dict(sys.modules, {"ollama": fake_ollama}),
        patch("rag.embeddings.ollama_provider.infer_embedding_dimension", return_value=0),
    ):
        provider = OllamaEmbeddingProvider(model="test-model")
        with pytest.raises(ValueError, match="Unknown Ollama embedding dimension"):
            _ = provider.model_info


def test_ollama_provider_embed_batch_no_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    mock_client = MagicMock()
    mock_client.embed.return_value = {"embeddings": [[0.1]], "prompt_eval_count": 1}
    fake_ollama = SimpleNamespace(Client=MagicMock(return_value=mock_client))
    with patch.dict(sys.modules, {"ollama": fake_ollama}):
        provider = OllamaEmbeddingProvider(model="test-model")
        assert provider.embed("test") == [0.1]
        mock_client.embed.assert_called_once_with(
            model="test-model",
            input=["test"],
        )


def test_openai_provider_passes_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "embedding_dimension_override", 1024)
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


def test_openai_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", None)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not defined"):
        OpenAIEmbeddingProvider()


def test_openai_provider_model_info_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", 512)
    with patch("rag.embeddings.openai_provider.OpenAI"):
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
        assert provider.model_info.dimension == 512


def test_openai_provider_model_info_known_no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with patch("rag.embeddings.openai_provider.OpenAI"):
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
        assert provider.model_info.dimension == 1536


def test_openai_provider_model_info_infer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with (
        patch("rag.embeddings.openai_provider.OpenAI"),
        patch("rag.embeddings.openai_provider.infer_embedding_dimension", return_value=128),
    ):
        provider = OpenAIEmbeddingProvider(model="unknown-model")
        assert provider.model_info.dimension == 128


def test_openai_provider_model_info_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with (
        patch("rag.embeddings.openai_provider.OpenAI"),
        patch("rag.embeddings.openai_provider.infer_embedding_dimension", return_value=0),
    ):
        provider = OpenAIEmbeddingProvider(model="unknown-model")
        with pytest.raises(ValueError, match="Unknown OpenAI embedding dimension"):
            _ = provider.model_info


def test_openai_provider_embed_no_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with patch("rag.embeddings.openai_provider.OpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)
        mock_openai.return_value.embeddings.create.return_value = mock_response
        provider = OpenAIEmbeddingProvider(model="text-embedding-ada-002")
        assert provider.embed("test text") == [0.1]
        mock_openai.return_value.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002",
            input=["test text"],
        )
        usage = provider.get_model_usage()
        assert usage.prompt_tokens == 10


def test_openai_provider_embed_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with patch("rag.embeddings.openai_provider.OpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.data = []
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)
        mock_openai.return_value.embeddings.create.return_value = mock_response
        provider = OpenAIEmbeddingProvider(model="text-embedding-ada-002")
        assert provider.embed("test text") == []


def test_openai_provider_embed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with patch("rag.embeddings.openai_provider.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.side_effect = Exception("API error")
        provider = OpenAIEmbeddingProvider(model="text-embedding-ada-002")
        assert provider.embed("test text") == []


def test_openai_provider_update_usage_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "test")
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    with patch("rag.embeddings.openai_provider.OpenAI") as mock_openai:
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1])]
        mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)
        mock_openai.return_value.embeddings.create.return_value = mock_response
        provider = OpenAIEmbeddingProvider(model="unknown-model")
        provider.embed("test text")
        usage = provider.get_model_usage()
        assert usage.estimated_cost_usd == 0.0


# --- SentenceTransformersEmbeddingProvider Tests ---
def test_sentence_transformers_provider_missing_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", 256)
    with (
        patch.dict(sys.modules, {"sentence_transformers": None}),
        pytest.raises(RuntimeError, match="Sentence-Transformers support requires"),
    ):
        SentenceTransformersEmbeddingProvider()


def test_sentence_transformers_provider_uses_truncate_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", 256)
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 1024
    mock_model.encode.return_value.tolist.return_value = [[0.1, 0.2]]
    fake_st = SimpleNamespace(SentenceTransformer=MagicMock(return_value=mock_model))
    with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
        provider = SentenceTransformersEmbeddingProvider(model="BAAI/bge-m3")
        assert provider.model_info.dimension == 256
        assert provider.embed("hello") == [0.1, 0.2]
        mock_model.encode.assert_called_once_with(
            ["hello"],
            convert_to_numpy=True,
            truncate_dim=256,
        )
        usage = provider.get_model_usage()
        assert usage.total_tokens == 1


def test_sentence_transformers_provider_model_info_infer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = None
    fake_st = SimpleNamespace(SentenceTransformer=MagicMock(return_value=mock_model))
    with (
        patch.dict(sys.modules, {"sentence_transformers": fake_st}),
        patch(
            "rag.embeddings.sentence_transformers_provider.infer_embedding_dimension",
            return_value=128,
        ),
    ):
        provider = SentenceTransformersEmbeddingProvider(model="unknown")
        assert provider.model_info.dimension == 128


def test_sentence_transformers_provider_model_info_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 0
    fake_st = SimpleNamespace(SentenceTransformer=MagicMock(return_value=mock_model))
    with (
        patch.dict(sys.modules, {"sentence_transformers": fake_st}),
        patch(
            "rag.embeddings.sentence_transformers_provider.infer_embedding_dimension",
            return_value=0,
        ),
    ):
        provider = SentenceTransformersEmbeddingProvider(model="unknown")
        with pytest.raises(ValueError, match="Unable to resolve embedding dimension"):
            _ = provider.model_info


def test_sentence_transformers_provider_model_info_from_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", None)
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 768
    fake_st = SimpleNamespace(SentenceTransformer=MagicMock(return_value=mock_model))
    with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
        provider = SentenceTransformersEmbeddingProvider(model="some-model")
        assert provider.model_info.dimension == 768


def test_sentence_transformers_provider_embed_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", 256)
    mock_model = MagicMock()
    mock_model.encode.return_value.tolist.return_value = []
    fake_st = SimpleNamespace(SentenceTransformer=MagicMock(return_value=mock_model))
    with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
        provider = SentenceTransformersEmbeddingProvider()
        assert provider.embed("hello") == []


def test_sentence_transformers_provider_embed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_dimension_override", 256)
    mock_model = MagicMock()
    mock_model.encode.side_effect = Exception("encode error")
    fake_st = SimpleNamespace(SentenceTransformer=MagicMock(return_value=mock_model))
    with patch.dict(sys.modules, {"sentence_transformers": fake_st}):
        provider = SentenceTransformersEmbeddingProvider()
        assert provider.embed("hello") == []
