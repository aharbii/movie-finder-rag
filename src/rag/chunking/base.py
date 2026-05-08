from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag.models.movie import Movie


@dataclass(frozen=True, slots=True)
class ChunkResult:
    """Text plus metadata emitted by a chunking strategy."""

    text: str
    chunk_index: int
    payload: dict[str, Any] = field(default_factory=dict)


class Chunker(ABC):
    """Strategy interface for turning a movie record into embedding text chunks."""

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Stable strategy identifier used in payloads and reports."""

    @abstractmethod
    def chunk(self, movie: Movie) -> list[ChunkResult]:
        """Return embedding chunks for one movie."""


def compact_lines(lines: list[tuple[str, str | int | list[str]]]) -> str:
    """Render non-empty labeled fields into the flat text format used for embeddings."""
    rendered: list[str] = []
    for label, value in lines:
        if isinstance(value, list):
            text = ", ".join(item.strip() for item in value if item.strip())
        else:
            text = str(value).strip()
        if text:
            rendered.append(f"{label}: {text}")
    return "\n".join(rendered)
