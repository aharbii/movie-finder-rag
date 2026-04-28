from typing import Any, cast

from rag.config import settings
from rag.embeddings.base import EmbeddingModelMetadata
from rag.models.movie import Movie
from rag.utils.logger import get_logger
from rag.vectorstore.base import VectorStore
from rag.vectorstore.naming import resolve_collection_name


class PineconeVectorStore(VectorStore):
    """Pinecone-backed vector store."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is not defined in settings")

        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise RuntimeError(
                "Pinecone support requires the optional 'pinecone' dependency."
            ) from exc

        self._serverless_spec_cls = ServerlessSpec
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self.index_host = settings.pinecone_index_host
        self._index: Any | None = None

    def target_name(self, embedding_model: EmbeddingModelMetadata) -> str:
        return resolve_collection_name(settings.vector_collection_prefix, embedding_model)

    def count(self, embedding_model: EmbeddingModelMetadata) -> int:
        index = self._get_index(embedding_model)
        namespace = self.target_name(embedding_model)
        stats = index.describe_index_stats()
        namespaces = getattr(stats, "namespaces", None) or stats.get("namespaces", {})
        namespace_stats = namespaces.get(namespace)
        if namespace_stats is None:
            return 0
        return int(
            getattr(namespace_stats, "vector_count", None) or namespace_stats.get("vector_count", 0)
        )

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
        index = self._get_index(embedding_model)
        namespace = self.target_name(embedding_model)
        records = [
            {
                "id": str(movie.id),
                "values": vector,
                "metadata": movie.model_dump(),
            }
            for movie, vector in zip(movies, vectors, strict=True)
        ]
        index.upsert(namespace=namespace, vectors=records)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        embedding_model: EmbeddingModelMetadata,
    ) -> list[Movie]:
        index = self._get_index(embedding_model)
        namespace = self.target_name(embedding_model)
        response = index.query(
            namespace=namespace,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )
        matches = getattr(response, "matches", None) or response.get("matches", [])
        results: list[Movie] = []
        for match in matches:
            metadata = getattr(match, "metadata", None) or match.get("metadata")
            if metadata:
                results.append(Movie(**cast(dict[str, Any], metadata)))
        return results

    def _get_index(self, embedding_model: EmbeddingModelMetadata) -> Any:
        if self._index is not None:
            return self._index

        if not self.index_host:
            self._ensure_index(embedding_model)
            description = self.client.describe_index(name=self.index_name)
            self.index_host = getattr(description, "host", None) or description.get("host")

        self._index = self.client.Index(host=self.index_host)
        return self._index

    def _ensure_index(self, embedding_model: EmbeddingModelMetadata) -> None:
        if self.client.has_index(self.index_name):
            return

        self.logger.info("Creating Pinecone index %s", self.index_name)
        self.client.create_index(
            name=self.index_name,
            vector_type="dense",
            dimension=embedding_model.dimension,
            metric="cosine",
            spec=self._serverless_spec_cls(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
