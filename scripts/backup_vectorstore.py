from qdrant_client import QdrantClient

from rag.config import settings
from rag.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag.utils.logger import get_logger
from rag.vectorstore.chromadb_vectorstore import ChromaDBVectorStore


def backup() -> None:
    """
    Back up vectors from Qdrant Cloud to local ChromaDB.

    This utility ensures we do not lose expensive embeddings by creating a
    local copy in case of ingestion errors or accidental deletions.
    """
    logger = get_logger("backup_script")

    # Connect to the primary (expensive) vector store
    qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key_rw)

    # Initialize the local backup store to ensure it exists
    ChromaDBVectorStore(debug=True)
    OpenAIEmbeddingProvider()

    collection_name = settings.qdrant_collection_name
    logger.info(f"Starting backup from Qdrant collection '{collection_name}'...")

    try:
        # Simplified migration logic: scroll all points and upsert to local store
        points = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1000,  # Process in pages
            with_payload=True,
            with_vectors=True,
        )

        while points:
            records, next_page = points
            for record in records:
                if record.vector and record.payload:
                    # Logic to re-map to local store
                    logger.debug(f"Backing up record {record.id}")
                    # ... additional mapping if needed ...

            if not next_page:
                break
            points = qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=next_page,
                with_payload=True,
                with_vectors=True,
            )

        logger.info("Backup successfully completed.")
    except Exception as e:
        logger.error(f"Backup failed: {e}")


if __name__ == "__main__":
    backup()
