from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.config import settings
from rag.embeddings.base import EmbeddingModelMetadata
from rag.models.movie import Movie
from rag.utils.logger import get_logger
from rag.vectorstore.base import VectorStore


class QdrantVectorStoreError(Exception):
    """Custom exception for Qdrant VectorStore operations."""

    pass


class QdrantVectorStore(VectorStore):
    """
    Qdrant-specific implementation of the VectorStore interface.
    Integrates with Qdrant Cloud via Pydantic settings.
    """

    def __init__(self) -> None:
        """Initialize the Qdrant client using settings.

        Raises:
            QdrantVectorStoreError: If environment configuration is missing.
        """
        self.logger = get_logger(self.__class__.__name__)

        self._validate_env()

        self.client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key_rw)

    def upsert(
        self, movie: Movie, vector: list[float], embedding_model: EmbeddingModelMetadata
    ) -> None:
        """Upsert a single movie vector into Qdrant."""
        collection_name = self._collection_name(embedding_model)
        self._validate_collection(embedding_model, collection_name)

        point = PointStruct(
            id=movie.id,
            vector=vector,
            payload=movie.model_dump(),
        )
        self.client.upsert(collection_name=collection_name, points=[point])

    def upsert_batch(
        self,
        movies: list[Movie],
        vectors: list[list[float]],
        embedding_model: EmbeddingModelMetadata,
    ) -> None:
        """Upsert multiple movie vectors into Qdrant in a single call."""
        collection_name = self._collection_name(embedding_model)
        self._validate_collection(embedding_model, collection_name)

        points = []
        for movie, vector in zip(movies, vectors, strict=True):
            point = PointStruct(
                id=movie.id,
                vector=vector,
                payload=movie.model_dump(),
            )
            points.append(point)

        self.client.upsert(collection_name=collection_name, points=points)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        embedding_model: EmbeddingModelMetadata,
    ) -> list[Movie]:
        """Perform a vector search in Qdrant and return Movie objects."""
        collection_name = self._collection_name(embedding_model)
        self._validate_collection(embedding_model, collection_name)

        results = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )

        movies = []
        for result in results.points:
            if result.payload:
                movies.append(Movie(**result.payload))

        return movies

    def _validate_collection(
        self, embedding_model: EmbeddingModelMetadata, collection_name: str
    ) -> None:
        """Ensure the target collection exists with correct dimensions."""
        if not self.client.collection_exists(collection_name):
            self.logger.info(f"Creating collection for {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_model.dimension, distance=Distance.COSINE
                ),
            )

    def _collection_name(self, embedding_model: EmbeddingModelMetadata) -> str:
        """Resolve the collection name from settings or embedding model name."""
        return settings.qdrant_collection_name or embedding_model.name.replace("/", "-")

    def _validate_env(self) -> None:
        """Validate required settings exist."""
        if not settings.qdrant_api_key_rw:
            raise QdrantVectorStoreError("QDRANT_API_KEY_RW is not defined in settings")

        if not settings.qdrant_url:
            raise QdrantVectorStoreError("QDRANT_URL is not defined in settings")
