from functools import lru_cache

from rag.config import VectorStoreName, settings
from rag.vectorstore.base import VectorStore
from rag.vectorstore.chromadb_vectorstore import ChromaDBVectorStore
from rag.vectorstore.pgvector_vectorstore import PGVectorStore
from rag.vectorstore.pinecone_vectorstore import PineconeVectorStore
from rag.vectorstore.qdrant_vectorstore import QdrantVectorStore

_VECTOR_STORES: dict[VectorStoreName, type[VectorStore]] = {
    "qdrant": QdrantVectorStore,
    "chromadb": ChromaDBVectorStore,
    "pinecone": PineconeVectorStore,
    "pgvector": PGVectorStore,
}


def create_vector_store(vector_store_name: VectorStoreName | None = None) -> VectorStore:
    """Instantiate the configured vector store without caching."""
    resolved_vector_store = vector_store_name or settings.vector_store
    store_cls = _VECTOR_STORES[resolved_vector_store]
    return store_cls()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return the configured vector store as a process singleton."""
    return create_vector_store()
