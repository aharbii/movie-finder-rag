import chromadb
import numpy as np

from rag.embeddings.base import EmbeddingModelMetadata
from rag.models.movie import Movie
from rag.utils.logger import get_logger
from rag.vectorstore.base import VectorStore


class ChromaDBVectorStore(VectorStore):
    """
    Local vector store implementation using ChromaDB for developer use and backups.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the local ChromaDB client.

        Args:
            debug (bool): Enable debug logging.
        """
        self.logger = get_logger(self.__class__.__name__, debug)
        self.client = chromadb.Client()

    def upsert(
        self, movie: Movie, vector: list[float], embedding_model: EmbeddingModelMetadata
    ) -> None:
        """Upsert a single movie into the local ChromaDB collection."""
        collection = self.client.get_or_create_collection(
            name=embedding_model.name.replace("/", "-")
        )
        collection.add(
            ids=[str(movie.id)],
            embeddings=np.asarray(vector),
            metadatas=[movie.model_dump()],
        )

    def upsert_batch(
        self,
        movies: list[Movie],
        vectors: list[list[float]],
        embedding_model: EmbeddingModelMetadata,
    ) -> None:
        """Upsert multiple movies into the local ChromaDB collection in a batch."""
        collection = self.client.get_or_create_collection(
            name=embedding_model.name.replace("/", "-")
        )
        collection.add(
            ids=[str(m.id) for m in movies],
            embeddings=np.asarray(vectors),
            metadatas=[m.model_dump() for m in movies],
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        embedding_model: EmbeddingModelMetadata,
    ) -> list[Movie]:
        """Perform a local vector search using ChromaDB."""
        collection = self.client.get_or_create_collection(
            name=embedding_model.name.replace("/", "-")
        )
        results = collection.query(query_embeddings=np.asarray(query_vector), n_results=top_k)

        movies = []
        if results["metadatas"]:
            for metadata in results["metadatas"][0]:
                movies.append(Movie(**metadata))
        return movies
