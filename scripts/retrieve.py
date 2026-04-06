from rag.config import settings
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.gemini_provider import GeminiEmbeddingProvider
from rag.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag.utils.logger import get_logger
from rag.vectorstore.qdrant_vectorstore import QdrantVectorStore


def interactive_retrieve() -> None:
    """
    Interactive CLI for developers to evaluate RAG retrieval performance.

    Allows users to input natural language queries and see the top-K most
    similar movie results from the pre-embedded Qdrant collection.
    """
    logger = get_logger("interactive_eval")
    print("\n" + "=" * 60)
    print("✨  Movie Finder RAG — Interactive Evaluation CLI  ✨")
    print("=" * 60)
    logger.info("Starting interactive retrieval session...")

    # Initialize components
    provider: EmbeddingProvider

    if settings.embedding_provider == "openai":
        provider = OpenAIEmbeddingProvider()
    else:
        provider = GeminiEmbeddingProvider()

    store = QdrantVectorStore()
    model_info = provider.model_info

    print(f"\n🔗 Connected to Qdrant collection: '{settings.qdrant_collection_name}'")
    print(f"🤖 Using Embedding Provider: {settings.embedding_provider.upper()}")
    print(f"📦 Model: {model_info.name} ({model_info.dimension}d)")
    print("\n💡 Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            print("\n" + "-" * 40)
            query = input("🔍 Enter your movie search query: ").strip()

            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue

            print("⏳ Embedding query and searching...")

            # Embed the search query
            query_vector = provider.embed(query)
            if not query_vector:
                print("❌ Failed to generate embedding. Please check your API keys.")
                continue

            # Perform the search
            results = store.search(query_vector, top_k=5, embedding_model=model_info)

            # Display results
            if not results:
                print("🤷 No matching movies found in the vector store.")
            else:
                print(f"\n🍿 Top {len(results)} matching movies:")
                for idx, movie in enumerate(results, 1):
                    print(f"\n{idx}. 🎬 {movie.title} ({movie.release_year})")
                    print(f"   👤 Director: {movie.director}")
                    print(f"   🎭 Genre:    {', '.join(movie.genre)}")
                    print(f"   👥 Cast:     {', '.join(movie.cast[:5])}...")
                    print(f"   📝 Snippet:  {movie.plot[:250]}...")
                    print("   " + "." * 10)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            print(f"💥 An unexpected error occurred: {e}")

    print("\n" + "=" * 60)
    print("👋 Session ended. Happy finding! 🍿")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    interactive_retrieve()
