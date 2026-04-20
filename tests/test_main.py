from unittest.mock import patch

from rag.main import main


def test_main_uses_factories() -> None:
    with (
        patch("rag.main.dataset.download_data"),
        patch("rag.main.get_embedding_provider") as get_provider,
        patch("rag.main.get_vector_store") as get_store,
        patch("rag.main.pipeline.ingest_csv") as ingest_csv,
    ):
        provider = get_provider.return_value
        provider.model_info.name = "text-embedding-3-large"
        get_store.return_value = object()

        main()

        get_provider.assert_called_once()
        get_store.assert_called_once()
        ingest_csv.assert_called_once_with(provider, get_store.return_value)
