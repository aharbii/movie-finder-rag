from unittest.mock import MagicMock, patch

import pytest

from rag.config import settings
from rag.embeddings.gemini_provider import GeminiEmbeddingProvider
from rag.embeddings.openai_provider import OpenAIEmbeddingProvider


def test_openai_provider_init_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OpenAI provider initialization without API key."""
    monkeypatch.setattr(settings, "openai_api_key", "")
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not defined"):
        OpenAIEmbeddingProvider()


def test_openai_provider_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OpenAI provider with an unknown model name."""
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    with patch("rag.embeddings.openai_provider.OpenAI"):
        provider = OpenAIEmbeddingProvider(model="unknown-model")
        assert provider.model_info.name == "unknown-model"


def test_openai_provider_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OpenAI provider embedding and usage tracking."""
    monkeypatch.setattr(settings, "openai_api_key", "test-key")

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
    mock_response.usage = MagicMock(prompt_tokens=10, total_tokens=10)

    with patch("rag.embeddings.openai_provider.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.return_value = mock_response
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
        vector = provider.embed("test text")

        assert vector == [0.1, 0.2]
        usage = provider.get_model_usage()
        assert usage.total_tokens == 10
        assert usage.estimated_cost_usd > 0


def test_openai_provider_embed_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OpenAI provider batch embedding."""
    monkeypatch.setattr(settings, "openai_api_key", "test-key")

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2]), MagicMock(embedding=[0.3, 0.4])]
    mock_response.usage = MagicMock(prompt_tokens=20, total_tokens=20)

    with patch("rag.embeddings.openai_provider.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.return_value = mock_response
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
        vectors = provider.embed_batch(["text1", "text2"])

        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        assert provider.get_model_usage().total_tokens == 20


def test_openai_provider_embed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OpenAI provider error handling."""
    monkeypatch.setattr(settings, "openai_api_key", "test-key")

    with patch("rag.embeddings.openai_provider.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.side_effect = Exception("API Error")
        provider = OpenAIEmbeddingProvider()
        assert provider.embed("test") == []
        assert provider.embed_batch(["test"]) == []


def test_gemini_provider_init_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Gemini provider initialization without API key."""
    monkeypatch.setattr(settings, "google_api_key", None)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY is not defined"):
        GeminiEmbeddingProvider()


def test_gemini_provider_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Gemini provider with an unknown model name."""
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    with patch("rag.embeddings.gemini_provider.genai.Client"):
        provider = GeminiEmbeddingProvider(model="unknown-model")
        assert provider.model_info.name == "unknown-model"
        assert provider.model_info.dimension == 768


def test_gemini_provider_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Gemini provider embedding."""
    monkeypatch.setattr(settings, "google_api_key", "test-key")

    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1, 0.2])]

    with patch("rag.embeddings.gemini_provider.genai.Client") as mock_genai:
        mock_genai.return_value.models.embed_content.return_value = mock_response
        provider = GeminiEmbeddingProvider(model="text-embedding-004")
        vector = provider.embed("test text")

        assert vector == [0.1, 0.2]
        assert provider.model_info.name == "text-embedding-004"


def test_gemini_provider_embed_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Gemini provider batch embedding."""
    monkeypatch.setattr(settings, "google_api_key", "test-key")

    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1, 0.2]), MagicMock(values=[0.3, 0.4])]

    with patch("rag.embeddings.gemini_provider.genai.Client") as mock_genai:
        mock_genai.return_value.models.embed_content.return_value = mock_response
        provider = GeminiEmbeddingProvider()
        vectors = provider.embed_batch(["text1", "text2"])

        assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_gemini_provider_embed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test Gemini provider error handling."""
    monkeypatch.setattr(settings, "google_api_key", "test-key")

    with patch("rag.embeddings.gemini_provider.genai.Client") as mock_genai:
        mock_genai.return_value.models.embed_content.side_effect = Exception("API Error")
        provider = GeminiEmbeddingProvider()
        assert provider.embed("test") == []
        assert provider.embed_batch(["test"]) == []
