from unittest.mock import MagicMock, patch

import pytest

from rag.config import settings
from rag.main import main


def test_main_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "qdrant_url", "http://test")
    monkeypatch.setattr(settings, "qdrant_api_key_rw", "key")

    with (
        patch("rag.main.dataset.download_data"),
        patch("rag.main.OpenAIEmbeddingProvider"),
        patch("rag.main.QdrantVectorStore"),
        patch("rag.main.pipeline.ingest_csv") as mock_ingest,
    ):
        main()
        mock_ingest.assert_called_once()


def test_main_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_provider", "gemini")
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "qdrant_url", "http://test")
    monkeypatch.setattr(settings, "qdrant_api_key_rw", "key")

    with (
        patch("rag.main.dataset.download_data"),
        patch("rag.main.GeminiEmbeddingProvider"),
        patch("rag.main.QdrantVectorStore"),
        patch("rag.main.pipeline.ingest_csv") as mock_ingest,
    ):
        main()
        mock_ingest.assert_called_once()


def test_main_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embedding_provider", "unsupported")

    with patch("rag.main.dataset.download_data"), patch("rag.main.get_logger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        main()
        mock_logger.error.assert_called()
        # Find the call that contains the expected message
        found = False
        for call in mock_logger.error.call_args_list:
            if "Unsupported embedding provider" in str(call):
                found = True
                break
        assert found
