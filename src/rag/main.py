from rag.config import settings
from rag.dataset import dataset
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.gemini_provider import GeminiEmbeddingProvider
from rag.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag.ingestion import pipeline
from rag.utils.logger import configure_logging, get_logger
from rag.vectorstore.qdrant_vectorstore import QdrantVectorStore

# Bootstrap logging before any logger is obtained.
# Reads LOG_LEVEL and LOG_FORMAT from the environment.
configure_logging()

logger = get_logger(__name__)


def main() -> None:
    """Entrypoint for the offline RAG ingestion pipeline.

    Resolves configuration via Pydantic and triggers the ingestion flow.
    """
    # 1. Download data from Kaggle if not present
    dataset.download_data()

    # 2. Resolve embedding provider based on runtime settings
    embedding_provider: EmbeddingProvider

    if settings.embedding_provider == "openai":
        logger.info("Using OpenAI provider with model: %s", settings.openai_embedding_model)
        embedding_provider = OpenAIEmbeddingProvider()
    elif settings.embedding_provider == "gemini":
        logger.info("Using Gemini provider with model: %s", settings.gemini_embedding_model)
        embedding_provider = GeminiEmbeddingProvider()
    else:
        logger.error("Unsupported embedding provider: %s", settings.embedding_provider)
        return

    # 3. Initialize vector store (Qdrant Cloud)
    vector_store = QdrantVectorStore()

    # 4. Trigger the ingestion pipeline
    pipeline.ingest_csv(embedding_provider, vector_store)


if __name__ == "__main__":
    main()
