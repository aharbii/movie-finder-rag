from rag.chunking.base import Chunker, ChunkResult, compact_lines
from rag.models.movie import Movie


class FlatChunker(Chunker):
    """Baseline strategy: one full movie text block per movie."""

    @property
    def strategy_name(self) -> str:
        return "flat"

    def chunk(self, movie: Movie) -> list[ChunkResult]:
        text = compact_lines(
            [
                ("Title", movie.title),
                ("Release Year", movie.release_year),
                ("Director", movie.director),
                ("Genre", movie.genre),
                ("Cast", movie.cast),
                ("Plot", movie.plot),
            ]
        )
        return [ChunkResult(text=text, chunk_index=0)]
