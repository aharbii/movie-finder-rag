from abc import ABC, abstractmethod

from embeddings.base import EmbeddingModel
from models.movie import Movie


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, movie: Movie, vector: list[float], embedding_model: EmbeddingModel) -> None:
        pass

    @abstractmethod
    def upsert_batch(
        self,
        movies: list[Movie],
        vectors: list[list[float]],
        embedding_model: EmbeddingModel,
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
        embedding_model: EmbeddingModel,
    ) -> list[Movie]:
        pass
