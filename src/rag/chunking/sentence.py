import re

from rag.chunking.base import Chunker, ChunkResult, compact_lines
from rag.models.movie import Movie

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker(Chunker):
    """Group plot sentences into natural-language windows."""

    def __init__(self, min_sentences: int = 1, max_sentences: int = 4) -> None:
        if min_sentences < 1:
            raise ValueError("min_sentences must be at least 1.")
        if max_sentences < min_sentences:
            raise ValueError("max_sentences must be greater than or equal to min_sentences.")
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    @property
    def strategy_name(self) -> str:
        return "sentence"

    def chunk(self, movie: Movie) -> list[ChunkResult]:
        sentences = [part.strip() for part in _SENTENCE_RE.split(movie.plot) if part.strip()]
        if not sentences:
            return []

        chunks: list[ChunkResult] = []
        start = 0
        while start < len(sentences):
            end = min(start + self.max_sentences, len(sentences))
            remaining = len(sentences) - end
            if 0 < remaining < self.min_sentences:
                end = len(sentences)
            text = compact_lines(
                [
                    ("Title", movie.title),
                    ("Plot", " ".join(sentences[start:end])),
                ]
            )
            chunks.append(
                ChunkResult(
                    text=text,
                    chunk_index=len(chunks),
                    payload={
                        "min_sentences": self.min_sentences,
                        "max_sentences": self.max_sentences,
                        "sentence_start": start,
                    },
                )
            )
            start = end
        return chunks
