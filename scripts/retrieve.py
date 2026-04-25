from rag.config import settings
from rag.embeddings.factory import get_embedding_provider
from rag.utils.logger import get_logger
from rag.vectorstore.factory import get_vector_store


def interactive_retrieve() -> None:
    """Interactive CLI for developers to evaluate retrieval quality."""
    logger = get_logger("interactive_eval")
    print("\n" + "=" * 60)
    print("✨  Movie Finder RAG — Interactive Evaluation CLI  ✨")
    print("=" * 60)
    logger.info("Starting interactive retrieval session...")

    provider = get_embedding_provider()
    store = get_vector_store()
    model_info = provider.model_info
    target_name = store.target_name(model_info)

    print(f"\n🗂️  Vector Store: {settings.vector_store}")
    print(f"🔗 Connected target: '{target_name}'")
    print(f"🤖 Embedding Provider: {settings.embedding_provider.upper()}")
    print(f"📦 Model: {model_info.name} ({model_info.dimension}d)")
    print("\n💡 Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            print("\n" + "-" * 40)
            query = input("🔍 Enter your movie search query: ").strip()
            if query.lower() in {"exit", "quit"}:
                break
            if not query:
                continue

            print("⏳ Embedding query and searching...")
            query_vector = provider.embed(query)
            if not query_vector:
                print(
                    "❌ Failed to generate embedding. Check provider credentials or local runtime."
                )
                continue

            results = store.search(query_vector, top_k=5, embedding_model=model_info)
            if not results:
                print("🤷 No matching movies found.")
                continue

            print(f"\n🍿 Top {len(results)} matching movies:")
            for index, movie in enumerate(results, start=1):
                cast_preview = ", ".join(movie.cast[:5])
                if len(movie.cast) > 5:
                    cast_preview += "..."

                plot_preview = movie.plot[:250]
                if len(movie.plot) > 250:
                    plot_preview += "..."

                print(f"\n{index}. 🎬 {movie.title} ({movie.release_year})")
                print(f"   👤 Director: {movie.director}")
                print(f"   🎭 Genre:    {', '.join(movie.genre)}")
                print(f"   👥 Cast:     {cast_preview}")
                print(f"   📝 Snippet:  {plot_preview}")
                print("   " + "." * 10)

        except KeyboardInterrupt:
            break
        except Exception as exc:
            logger.error("Error during retrieval: %s", exc)
            print(f"💥 An unexpected error occurred: {exc}")

    print("\n" + "=" * 60)
    print("👋 Session ended. Happy finding! 🍿")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    interactive_retrieve()
