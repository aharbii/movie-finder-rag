import chromadb
import numpy as np

from rag.config import settings
from rag.embeddings.base import EmbeddingModelMetadata
from rag.models.movie import Movie
from rag.utils.logger import get_logger
from rag.vectorstore.base import VectorStore
from rag.vectorstore.naming import resolve_collection_name


class ChromaDBVectorStore(VectorStore):
    """Local ChromaDB vector store implementation."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.client = chromadb.PersistentClient(path=settings.chromadb_persist_path)

    def target_name(self, embedding_model: EmbeddingModelMetadata) -> str:
        return resolve_collection_name(settings.qdrant_collection_prefix, embedding_model)

    def count(self, embedding_model: EmbeddingModelMetadata) -> int:
        collection = self.client.get_or_create_collection(name=self.target_name(embedding_model))
        return int(collection.count())

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
        collection = self.client.get_or_create_collection(name=self.target_name(embedding_model))
        collection.upsert(
            ids=[str(movie.id) for movie in movies],
            embeddings=np.asarray(vectors, dtype=np.float32),
            metadatas=[movie.model_dump() for movie in movies],
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        embedding_model: EmbeddingModelMetadata,
    ) -> list[Movie]:
        collection = self.client.get_or_create_collection(name=self.target_name(embedding_model))
        results = collection.query(
            query_embeddings=np.asarray([query_vector], dtype=np.float32),
            n_results=top_k,
        )
        metadatas = results.get("metadatas", [])
        if not metadatas:
            return []
        return [Movie(**metadata) for metadata in metadatas[0] if metadata]
