from abc import ABC, abstractmethod

from rag.embeddings.base import EmbeddingModelMetadata
from rag.models.movie import Movie


class VectorStore(ABC):
    """
    Abstract interface for vector database operations (e.g., Qdrant, ChromaDB).
    """

    @abstractmethod
    def upsert(
        self, movie: Movie, vector: list[float], embedding_model: EmbeddingModelMetadata
    ) -> None:
        """
        Upsert a single movie and its vector into the store.

        Args:
            movie (Movie): The Movie object to store.
            vector (list[float]): The embedding vector.
            embedding_model (EmbeddingModelMetadata): Metadata about the model used.
        """
        pass

    @abstractmethod
    def upsert_batch(
        self,
        movies: list[Movie],
        vectors: list[list[float]],
        embedding_model: EmbeddingModelMetadata,
    ) -> None:
        """
        Upsert multiple movies and their vectors into the store.

        Args:
            movies (list[Movie]): The list of Movie objects.
            vectors (list[list[float]]): The list of vectors.
            embedding_model (EmbeddingModelMetadata): Metadata about the model used.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        embedding_model: EmbeddingModelMetadata,
    ) -> list[Movie]:
        """
        Perform a vector similarity search.

        Args:
            query_vector (list[float]): The query vector.
            top_k (int): Number of top results to return.
            embedding_model (EmbeddingModelMetadata): Metadata about the model used.

        Returns:
            list[Movie]: The top-K similar movies.
        """
        pass
