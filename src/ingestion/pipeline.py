from tqdm import tqdm

from embeddings.base import EmbeddingProvider
from ingestion import csv_loader
from utils.logger import get_logger
from vectorstore.base import VectorStore

logger = get_logger(__name__)


def ingest_csv(
    embedding_provider: EmbeddingProvider, vector_store: VectorStore
) -> None:
    movies_descriptions = csv_loader.parse_wiki_movie()
    logger.info(f"Embedding {len(movies_descriptions)} movies to the vector store")
    movies = []
    vectors = []
    with tqdm(total=len(movies_descriptions)) as pbar:
        for movie, description in movies_descriptions:
            movies.append(movie)
            embeddings = embedding_provider.embed(description)
            vectors.append(embeddings)
            if len(movies) >= 20:
                vector_store.upsert_batch(movies, vectors, embedding_provider.model)
                movies = []
                vectors = []
            pbar.update(1)
        if movies:
            vector_store.upsert_batch(movies, vectors, embedding_provider.model)
    model_usage = embedding_provider.get_model_usage()
    logger.info("Embedding is done")
    logger.info(f"{model_usage.model_dump()}")
