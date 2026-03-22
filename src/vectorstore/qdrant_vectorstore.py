import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from embeddings.base import EmbeddingModel
from models.movie import Movie
from utils.logger import get_logger
from vectorstore.base import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(self, debug: bool = False) -> None:
        self.logger = get_logger(self.__class__.__name__, debug)

        self._validate_env()

        self.client = QdrantClient(
            url=os.getenv("QDRANT_ENDPOINT"), api_key=os.getenv("QDRANT_API_KEY")
        )

    def upsert(
        self, movie: Movie, vector: list[float], embedding_model: EmbeddingModel
    ) -> None:
        self._validate_collection(embedding_model)
        point = PointStruct(
            id=movie.id,
            vector=vector,
            payload=movie.model_dump(),
        )
        self.client.upsert(
            collection_name=embedding_model.name.replace("/", "-"), points=[point]
        )

    def upsert_batch(
        self,
        movies: list[Movie],
        vectors: list[list[float]],
        embedding_model: EmbeddingModel,
    ) -> None:
        self._validate_collection(embedding_model)
        points = []
        for movie, vector in zip(movies, vectors, strict=True):
            point = PointStruct(
                id=movie.id,
                vector=vector,
                payload=movie.model_dump(),
            )
            points.append(point)
        self.client.upsert(
            collection_name=embedding_model.name.replace("/", "-"), points=points
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        embedding_model: EmbeddingModel,
    ) -> list[Movie]:
        self._validate_collection(embedding_model)

        results = self.client.query_points(
            collection_name=embedding_model.name.replace("/", "-"),
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )

        movies = []

        for result in results.points:
            if result.payload:
                movie = Movie(
                    title=result.payload.get("title"),
                    release_year=result.payload.get("release_year"),
                    director=result.payload.get("director"),
                    genre=result.payload.get("genre"),
                    cast=result.payload.get("cast"),
                    plot=result.payload.get("plot"),
                )
                movies.append(movie)

        return movies

    def _validate_collection(self, embedding_model: EmbeddingModel) -> None:
        if not self.client.collection_exists(embedding_model.name.replace("/", "-")):
            self.logger.info(
                f"Creating collection for {embedding_model.name.replace('/', '-')}"
            )
            self.client.create_collection(
                collection_name=embedding_model.name.replace("/", "-"),
                vectors_config=VectorParams(
                    size=embedding_model.embedding_dimension, distance=Distance.COSINE
                ),
            )

    def _validate_env(self) -> None:
        if not os.getenv("QDRANT_API_KEY"):
            self.logger.error("QDRANT_API_KEY is not defined")
            sys.exit(1)

        if not os.getenv("QDRANT_ENDPOINT"):
            self.logger.error("QDRANT_ENDPOINT is not defined")
            sys.exit(1)
