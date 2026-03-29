from rag.config import settings
from rag.dataset import dataset
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.gemini_provider import GeminiEmbeddingProvider
from rag.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag.ingestion import pipeline
from rag.utils.logger import get_logger
from rag.vectorstore.qdrant_vectorstore import QdrantVectorStore


def main() -> None:
    """
    Entrypoint for the offline RAG ingestion pipeline.
    Resolves configuration via Pydantic and triggers the ingestion flow.
    """
    logger = get_logger("rag_main", debug=True)

    # 1. Download data from Kaggle if not present
    dataset.download_data(debug=True)

    # 2. Resolve embedding provider based on runtime settings
    embedding_provider: EmbeddingProvider

    if settings.embedding_provider == "openai":
        logger.info(f"Using OpenAI provider with model: {settings.openai_embedding_model}")
        embedding_provider = OpenAIEmbeddingProvider()
    elif settings.embedding_provider == "gemini":
        logger.info(f"Using Gemini provider with model: {settings.gemini_embedding_model}")
        embedding_provider = GeminiEmbeddingProvider()
    else:
        logger.error(f"Unsupported embedding provider: {settings.embedding_provider}")
        return

    # 3. Initialize vector store (Qdrant Cloud)
    vector_store = QdrantVectorStore(debug=True)

    # 4. Trigger the ingestion pipeline
    pipeline.ingest_csv(embedding_provider, vector_store, debug=True)


if __name__ == "__main__":
    main()
