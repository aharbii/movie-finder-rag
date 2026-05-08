from collections.abc import Iterable

from rag.chunking.base import Chunker, ChunkResult, compact_lines
from rag.models.movie import Movie

SUPPORTED_FIELDS = {"plot", "cast", "metadata"}


class FieldChunker(Chunker):
    """Emit separate chunks for plot, cast, and metadata-oriented retrieval."""

    def __init__(self, fields: Iterable[str] = ("plot", "cast", "metadata")) -> None:
        normalized = [field.strip().lower() for field in fields if field.strip()]
        unknown = sorted(set(normalized) - SUPPORTED_FIELDS)
        if unknown:
            raise ValueError(f"Unsupported field chunk(s): {', '.join(unknown)}")
        if not normalized:
            raise ValueError("fields must include at least one supported field.")
        self.fields = tuple(dict.fromkeys(normalized))

    @property
    def strategy_name(self) -> str:
        return "field"

    def chunk(self, movie: Movie) -> list[ChunkResult]:
        chunks: list[ChunkResult] = []
        for field in self.fields:
            text = self._field_text(movie, field)
            if not text:
                continue
            chunks.append(
                ChunkResult(
                    text=text,
                    chunk_index=len(chunks),
                    payload={"chunk_field": field},
                )
            )
        return chunks

    def _field_text(self, movie: Movie, field: str) -> str:
        if field == "plot":
            return compact_lines([("Title", movie.title), ("Plot", movie.plot)])
        if field == "cast":
            return compact_lines([("Title", movie.title), ("Cast", movie.cast)])
        if field == "metadata":
            return compact_lines(
                [
                    ("Title", movie.title),
                    ("Release Year", movie.release_year),
                    ("Director", movie.director),
                    ("Genre", movie.genre),
                ]
            )
        raise ValueError(f"Unsupported field chunk: {field}")
