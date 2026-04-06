from tqdm import tqdm

from rag.config import settings
from rag.embeddings.base import EmbeddingProvider
from rag.ingestion import csv_loader
from rag.utils.logger import get_logger
from rag.vectorstore.base import VectorStore


def ingest_csv(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
) -> None:
    """
    Orchestrates the offline RAG ingestion pipeline.

    Workflow:
    1. Load and filter CSV
    2. Batch movies
    3. Call batch embedding
    4. Upsert to VectorStore

    Args:
        embedding_provider (EmbeddingProvider): The provider instance (e.g. OpenAI).
        vector_store (VectorStore): The vector store target (e.g. Qdrant).
    """
    logger = get_logger(__name__)

    logger.info("Initializing RAG ingestion pipeline...")
    movies = csv_loader.load_movies()

    if not movies:
        logger.warning("No movies found to ingest. Pipeline finished.")
        return

    logger.info(f"Loaded {len(movies)} movies. Starting batched embedding and ingestion...")

    batch_size = settings.batch_size
    model_info = embedding_provider.model_info

    # Progress bar and batched processing for efficiency
    for i in tqdm(range(0, len(movies), batch_size), desc="Ingesting Movies"):
        batch = movies[i : i + batch_size]
        batch_texts = [movie.plot for movie in batch]

        # Call batch embedding via the provider
        vectors = embedding_provider.embed_batch(batch_texts)

        if not vectors:
            logger.error(f"Failed to generate embeddings for batch {i // batch_size}. Skipping.")
            continue

        # Upsert the batch into the vector store
        vector_store.upsert_batch(batch, vectors, model_info)

    # Reporting final usage and estimated cost
    usage = embedding_provider.get_model_usage()
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"Total Movies Processed : {len(movies)}")
    logger.info(f"Total Prompt Tokens    : {usage.prompt_tokens}")
    logger.info(f"Estimated USD Cost     : ${usage.estimated_cost_usd:.5f}")
    logger.info("=" * 60)
