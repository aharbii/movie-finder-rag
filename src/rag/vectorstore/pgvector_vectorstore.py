import json
from typing import Any, cast

from rag.config import settings
from rag.embeddings.base import EmbeddingModelMetadata
from rag.models.movie import Movie
from rag.utils.logger import get_logger
from rag.vectorstore.base import VectorStore
from rag.vectorstore.naming import resolve_collection_name, sanitize_collection_token


class PGVectorStore(VectorStore):
    """PostgreSQL pgvector-backed vector store."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        if not settings.pgvector_dsn:
            raise ValueError("PGVECTOR_DSN is not defined in settings")

        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError(
                "PGVector support requires the optional 'psycopg' and 'pgvector' dependencies."
            ) from exc

        self._psycopg = psycopg
        self._register_vector = register_vector

    def target_name(self, embedding_model: EmbeddingModelMetadata) -> str:
        return resolve_collection_name(settings.qdrant_collection_prefix, embedding_model)

    def count(self, embedding_model: EmbeddingModelMetadata) -> int:
        table_name = self._table_name(embedding_model)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            row = cursor.fetchone()
            return int(row[0] if row else 0)

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
        table_name = self._table_name(embedding_model)
        self._ensure_table(table_name, embedding_model.dimension)

        with self._connect() as connection, connection.cursor() as cursor:
            for movie, vector in zip(movies, vectors, strict=True):
                cursor.execute(
                    f'''
                    INSERT INTO "{table_name}" (id, payload, embedding)
                    VALUES (%s, %s::jsonb, %s)
                    ON CONFLICT (id) DO UPDATE
                    SET payload = EXCLUDED.payload,
                        embedding = EXCLUDED.embedding
                    ''',
                    (movie.id, json.dumps(movie.model_dump()), vector),
                )
            connection.commit()

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        embedding_model: EmbeddingModelMetadata,
    ) -> list[Movie]:
        table_name = self._table_name(embedding_model)
        self._ensure_table(table_name, embedding_model.dimension)

        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f'''
                SELECT payload
                FROM "{table_name}"
                ORDER BY embedding <=> %s
                LIMIT %s
                ''',
                (query_vector, top_k),
            )
            rows = cursor.fetchall()

        return [Movie(**cast_payload(row[0])) for row in rows]

    def _connect(self) -> Any:
        connection = self._psycopg.connect(settings.pgvector_dsn)
        self._register_vector(connection)
        return connection

    def _ensure_table(self, table_name: str, dimension: int) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                f'''
                CREATE TABLE IF NOT EXISTS "{table_name}" (
                    id BIGINT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    embedding vector({dimension}) NOT NULL
                )
                '''
            )
            connection.commit()

    def _table_name(self, embedding_model: EmbeddingModelMetadata) -> str:
        target_name = self.target_name(embedding_model)
        return sanitize_collection_token(f"{settings.pgvector_schema}_{target_name}")


def cast_payload(payload: Any) -> dict[str, Any]:
    """Normalize psycopg payload values into Movie-ready dictionaries."""
    if isinstance(payload, str):
        return cast(dict[str, Any], json.loads(payload))
    return cast(dict[str, Any], dict(payload))
