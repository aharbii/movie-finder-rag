from dotenv import load_dotenv

from dataset import dataset
from embeddings.openai_provider import OpenAIEmbeddingProvider
from ingestion import pipeline
from vectorstore.qdrant_vectorstore import QdrantVectorStore

load_dotenv()


def main() -> None:
    dataset.download_data()

    embedding_provider = OpenAIEmbeddingProvider(
        model=OpenAIEmbeddingProvider.EMBEDDING_3_LARGE_MODEL
    )
    vector_store = QdrantVectorStore()
    pipeline.ingest_csv(embedding_provider, vector_store)


if __name__ == "__main__":
    main()
