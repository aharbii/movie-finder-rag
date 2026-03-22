from dotenv import load_dotenv

from embeddings.openai_provider import OpenAIEmbeddingProvider
from vectorstore.qdrant_vectorstore import QdrantVectorStore

load_dotenv()


def test_rag() -> None:
    embedding_provider = OpenAIEmbeddingProvider(
        model=OpenAIEmbeddingProvider.EMBEDDING_3_LARGE_MODEL
    )
    vector_store = QdrantVectorStore()

    test_set = {
        "The Inception": "A professional thief who steals secrets through use of dream-sharing technology is given the inverse task of planting an idea into the mind of a CEO.",
        "The Matrix": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
        "Parasite": "A poor family schemes to become employed by a wealthy family by infiltrating their household and posing as unrelated, highly qualified individuals.",
        "Interstellar": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival as Earth faces a global environmental collapse.",
        "The Lion King": "A young prince is cast out of his heritage by his cruel uncle and must find the courage to return and take back his place as the rightful king.",
        "Pulp Fiction": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
        "The Prestige": "After a tragic accident, two stage magicians in 1890s London engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other.",
        "Spirited Away": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, where humans are changed into beasts.",
        "The Truman Show": "An insurance salesman discovers that his entire life is actually a reality TV show telecast around the clock to the entire world.",
        "Arrival": "A linguist works with the military to communicate with alien newcomers who have landed giant spacecraft around the world before tensions lead to war.",
    }

    for title, query in test_set.items():
        query_vector = embedding_provider.embed(query)
        result = vector_store.search(query_vector, 5, embedding_provider.model)
        movies = [movie.title for movie in result]
        assert title in movies
