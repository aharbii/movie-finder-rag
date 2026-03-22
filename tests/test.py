from dotenv import load_dotenv

from embeddings.openai_provider import OpenAIEmbeddingProvider
from vectorstore.qdrant_vectorstore import QdrantVectorStore

load_dotenv()

embedding_provider = OpenAIEmbeddingProvider(
    model=OpenAIEmbeddingProvider.EMBEDDING_3_LARGE_MODEL
)
vector_store = QdrantVectorStore()

while True:
    query = input("Asking for: ")
    query_vector = embedding_provider.embed(query)
    result = vector_store.search(query_vector, 5, embedding_provider.model)
    for movie in result:
        print(movie.title)
