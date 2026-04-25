from abc import ABC, abstractmethod

from rag.embeddings.base import EmbeddingModelMetadata
from rag.models.movie import Movie


class VectorStore(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    def target_name(self, embedding_model: EmbeddingModelMetadata) -> str:
        """Return the concrete target name for the provided embedding model."""

    @abstractmethod
    def count(self, embedding_model: EmbeddingModelMetadata) -> int:
        """Return the number of stored records for the provided embedding model."""

    @abstractmethod
    def upsert(
        self, movie: Movie, vector: list[float], embedding_model: EmbeddingModelMetadata
    ) -> None:
        """Upsert a single movie and its vector into the store."""

    @abstractmethod
    def upsert_batch(
        self,
        movies: list[Movie],
        vectors: list[list[float]],
        embedding_model: EmbeddingModelMetadata,
    ) -> None:
        """Upsert multiple movies and their vectors into the store."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        embedding_model: EmbeddingModelMetadata,
    ) -> list[Movie]:
        """Perform a vector similarity search."""
