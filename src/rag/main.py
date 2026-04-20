from rag.dataset import dataset
from rag.embeddings.factory import get_embedding_provider
from rag.ingestion import pipeline
from rag.utils.logger import configure_logging, get_logger
from rag.vectorstore.factory import get_vector_store

configure_logging()
logger = get_logger(__name__)


def main() -> None:
    """Entrypoint for the offline RAG ingestion pipeline."""
    dataset.download_data()
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    logger.info(
        "Starting ingestion with provider=%s model=%s vector_store=%s",
        embedding_provider.__class__.__name__,
        embedding_provider.model_info.name,
        vector_store.__class__.__name__,
    )
    pipeline.ingest_csv(embedding_provider, vector_store)


if __name__ == "__main__":
    main()
