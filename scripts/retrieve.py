"""
Interactive RAG retrieval — run against cloud Qdrant.

Usage:
    PYTHONPATH=src uv run python scripts/retrieve.py
"""

from dotenv import load_dotenv


def print_movie(i: int, movie) -> None:  # type: ignore[no-untyped-def]
    genre = ", ".join(movie.genre) if isinstance(movie.genre, list) else movie.genre
    cast = ", ".join(movie.cast[:3]) if isinstance(movie.cast, list) else movie.cast

    print(f"\n  🎬 {i}. {movie.title}", end="")
    if movie.release_year:
        print(f" ({movie.release_year})", end="")
    print()
    if movie.director:
        print(f"     🎥 Director : {movie.director}")
    if genre:
        print(f"     🎭 Genre    : {genre}")
    if cast:
        print(f"     ⭐ Cast     : {cast}")
    if movie.plot:
        plot_preview = movie.plot[:200].rsplit(" ", 1)[0] + "..."
        print(f"     📖 Plot     : {plot_preview}")


def main() -> None:
    load_dotenv()

    # Import after dotenv so env vars are set before providers validate them
    from embeddings.openai_provider import OpenAIEmbeddingProvider
    from vectorstore.qdrant_vectorstore import QdrantVectorStore

    print("\n  🎬 Movie Finder — RAG Retrieval")
    print("  ================================")
    print("  🤖 Model  : text-embedding-3-large")
    print("  📊 Top-K  : 5")
    print("\n  ⏳ Initializing...")

    embedder = OpenAIEmbeddingProvider(model=OpenAIEmbeddingProvider.EMBEDDING_3_LARGE_MODEL)
    store = QdrantVectorStore()

    print("  ✅ Ready! Describe a movie you're looking for.")
    print("  💡 Commands: 'quit' or 'exit' to stop | 'top <n>' to change result count.\n")

    top_k = 5

    while True:
        try:
            query = input("  🔍 You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋 Goodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\n  👋 Goodbye!")
            break

        # Allow changing top-k on the fly: "top 8"
        if query.lower().startswith("top "):
            parts = query.split()
            if len(parts) == 2 and parts[1].isdigit():
                top_k = int(parts[1])
                print(f"\n  ✅ Result count set to {top_k}.\n")
            else:
                print("\n  💡 Usage: top <number>  (e.g. top 8)\n")
            continue

        print(f'\n  🔎 Searching for: "{query}"')
        print("  " + "─" * 50)

        try:
            query_vector = embedder.embed(query)
            results = store.search(
                query_vector=query_vector, top_k=top_k, embedding_model=embedder.model
            )
        except Exception as e:
            print(f"\n  ❌ Error: {e}\n")
            continue

        if not results:
            print("\n  🤷 No results found. Try a different description.\n")
            continue

        for i, movie in enumerate(results, 1):
            print_movie(i, movie)

        usage = embedder.get_model_usage()
        print(
            f"\n  📈 {len(results)} results | 🪙 tokens so far: {usage.total_tokens} | 💰 cost: ${usage.total_cost:.5f}\n"
        )


if __name__ == "__main__":
    main()
