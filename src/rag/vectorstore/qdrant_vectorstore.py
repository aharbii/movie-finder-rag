from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.config import settings
from rag.embeddings.base import EmbeddingModelMetadata
from rag.models.movie import Movie
from rag.utils.logger import get_logger
from rag.vectorstore.base import VectorStore
from rag.vectorstore.naming import resolve_collection_name


class QdrantVectorStoreError(Exception):
    """Custom exception for Qdrant vector store operations."""


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self._validate_env()
        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key_rw)

    def target_name(self, embedding_model: EmbeddingModelMetadata) -> str:
        return resolve_collection_name(settings.vector_collection_prefix, embedding_model)

    def count(self, embedding_model: EmbeddingModelMetadata) -> int:
        collection_name = self.target_name(embedding_model)
        if not self.client.collection_exists(collection_name):
            return 0
        return int(self.client.count(collection_name=collection_name, exact=True).count)

    def upsert(
        self, movie: Movie, vector: list[float], embedding_model: EmbeddingModelMetadata
    ) -> None:
        self.upsert_batch([movie], [vector], embedding_model)

    def upsert_batch(
        self,
        movies: list[Movie],
        vectors: list[list[float]],
        embedding_model: EmbeddingModelMetadata,
    ) -> None:
        collection_name = self.target_name(embedding_model)
        self._validate_collection(embedding_model, collection_name)
        points = [
            PointStruct(id=movie.id, vector=vector, payload=movie.model_dump())
            for movie, vector in zip(movies, vectors, strict=True)
        ]
        self.client.upsert(collection_name=collection_name, points=points)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        embedding_model: EmbeddingModelMetadata,
    ) -> list[Movie]:
        collection_name = self.target_name(embedding_model)
        self._validate_collection(embedding_model, collection_name)
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        return [Movie(**point.payload) for point in results.points if point.payload]

    def _validate_collection(
        self, embedding_model: EmbeddingModelMetadata, collection_name: str
    ) -> None:
        if not self.client.collection_exists(collection_name):
            self.logger.info("Creating Qdrant collection %s", collection_name)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_model.dimension,
                    distance=Distance.COSINE,
                ),
            )

    def _validate_env(self) -> None:
        if not settings.qdrant_api_key_rw:
            raise QdrantVectorStoreError("QDRANT_API_KEY_RW is not defined in settings")
        if not settings.qdrant_url:
            raise QdrantVectorStoreError("QDRANT_URL is not defined in settings")
