from rag.chunking.base import Chunker, ChunkResult
from rag.models.movie import Movie


class FixedSizeChunker(Chunker):
    """Split the flat movie text into whitespace-token windows with overlap."""

    def __init__(self, chunk_size: int = 160, overlap: int = 32) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")
        if overlap < 0:
            raise ValueError("overlap must not be negative.")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @property
    def strategy_name(self) -> str:
        return "fixed_size"

    def chunk(self, movie: Movie) -> list[ChunkResult]:
        from rag.chunking.flat import FlatChunker

        tokens = FlatChunker().chunk(movie)[0].text.split()
        if not tokens:
            return []

        chunks: list[ChunkResult] = []
        step = self.chunk_size - self.overlap
        start = 0
        index = 0
        while start < len(tokens):
            window = tokens[start : start + self.chunk_size]
            chunks.append(
                ChunkResult(
                    text=" ".join(window),
                    chunk_index=index,
                    payload={
                        "chunk_size": self.chunk_size,
                        "overlap": self.overlap,
                        "token_start": start,
                    },
                )
            )
            start += step
            if start + self.overlap >= len(tokens):
                start = len(tokens)
            index += 1
        return chunks
