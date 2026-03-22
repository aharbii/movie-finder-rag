from chromadb import chromadb
from db.models import Movie

from embeddings.base import EmbeddingModel
from utils.logger import get_logger
from vectorstore.base import VectorStore


class ChromaDBVectorStore(VectorStore):
    def __init__(self, debug: bool = False) -> None:
        self.logger = get_logger(self.__class__.__name__, debug)

        self.client = chromadb.PersistentClient()

    def upsert(
        self, movie: Movie, vector: list[float], embedding_model: EmbeddingModel
    ) -> None:
        self._validate_collection(embedding_model)

        collection = self.client.get_collection(embedding_model.name.replace("/", "-"))

        movie.genre = "/".join(movie.genre)
        movie.cast = ", ".join(movie.cast)

        collection.add(
            ids=str(movie.id),
            embeddings=vector,
            metadatas=movie.model_dump(),
        )

    def upsert_batch(
        self,
        movies: list[Movie],
        vectors: list[list[float]],
        embedding_model: EmbeddingModel,
    ) -> None:
        self._validate_collection(embedding_model)

        collection = self.client.get_collection(embedding_model.name.replace("/", "-"))

        for movie in movies:
            movie.genre = "/".join(movie.genre)
            movie.cast = ", ".join(movie.cast)

        collection.add(
            ids=[str(movie.id) for movie in movies],
            embeddings=vectors,
            metadatas=[movie.model_dump() for movie in movies],
        )

    def search(
        self, query_vector: list[float], top_k: int, embedding_model: EmbeddingModel
    ) -> list[Movie]:
        self._validate_collection(embedding_model)

        collection = self.client.get_collection(embedding_model.name.replace("/", "-"))
        response = collection.query(query_embeddings=query_vector, n_results=top_k)

        movies = []

        if response["metadatas"]:
            for metadata in response["metadatas"]:
                for data in metadata:
                    movie = Movie(
                        title=data.get("title"),
                        release_year=data.get("release_year"),
                        director=data.get("director"),
                        genre=data.get("genre"),
                        cast=data.get("cast"),
                        plot=data.get("plot"),
                    )
                    movies.append(movie)
        return movies

    def _validate_collection(self, embedding_model: EmbeddingModel) -> None:
        try:
            self.client.get_collection(embedding_model.name.replace("/", "-"))
        except chromadb.errors.NotFoundError:
            self.logger.info(
                f"Creating collection for {embedding_model.name.replace('/', '-')}"
            )
            self.client.create_collection(name=embedding_model.name.replace("/", "-"))
