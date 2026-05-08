from typing import cast

import pytest

from rag.chunking.base import compact_lines
from rag.chunking.factory import create_chunker, get_chunker
from rag.chunking.field import FieldChunker
from rag.chunking.fixed_size import FixedSizeChunker
from rag.chunking.flat import FlatChunker
from rag.chunking.sentence import SentenceChunker
from rag.config import ChunkingStrategyName, settings
from rag.models.movie import Movie


@pytest.fixture
def movie() -> Movie:
    return Movie(
        id=3,
        title="Chunk Movie",
        release_year=1999,
        director="Director",
        genre=["Sci-Fi", "Adventure"],
        cast=["Actor One", "Actor Two"],
        plot="First sentence. Second sentence! Third sentence? Fourth sentence. Fifth sentence.",
    )


def test_compact_lines_skips_empty_values() -> None:
    assert compact_lines([("Title", "Movie"), ("Cast", []), ("Plot", " Text ")]) == (
        "Title: Movie\nPlot: Text"
    )


def test_flat_chunker_emits_single_full_movie_chunk(movie: Movie) -> None:
    chunks = FlatChunker().chunk(movie)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert "Title: Chunk Movie" in chunks[0].text
    assert "Cast: Actor One, Actor Two" in chunks[0].text
    assert "Plot: First sentence." in chunks[0].text


def test_fixed_size_chunker_windows_with_overlap(movie: Movie) -> None:
    chunker = FixedSizeChunker(chunk_size=6, overlap=2)
    chunks = chunker.chunk(movie)

    assert chunker.strategy_name == "fixed_size"
    assert len(chunks) > 1
    assert chunks[0].payload == {"chunk_size": 6, "overlap": 2, "token_start": 0}
    assert chunks[1].payload["token_start"] == 4


def test_fixed_size_chunker_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        FixedSizeChunker(chunk_size=0)
    with pytest.raises(ValueError, match="overlap must not be negative"):
        FixedSizeChunker(chunk_size=5, overlap=-1)
    with pytest.raises(ValueError, match="overlap must be smaller"):
        FixedSizeChunker(chunk_size=5, overlap=5)


def test_fixed_size_chunker_handles_empty_text(movie: Movie) -> None:
    empty_movie = movie.model_copy(
        update={
            "title": "",
            "release_year": "",
            "director": "",
            "genre": [],
            "cast": [],
            "plot": "",
        }
    )
    assert FixedSizeChunker().chunk(empty_movie) == []


def test_field_chunker_emits_requested_fields(movie: Movie) -> None:
    chunker = FieldChunker(["plot", "cast", "metadata"])
    chunks = chunker.chunk(movie)

    assert chunker.strategy_name == "field"
    assert [chunk.payload["chunk_field"] for chunk in chunks] == ["plot", "cast", "metadata"]
    assert (
        chunks[0].text
        == "Title: Chunk Movie\nPlot: First sentence. Second sentence! Third sentence? Fourth sentence. Fifth sentence."
    )
    assert chunks[1].text == "Title: Chunk Movie\nCast: Actor One, Actor Two"
    assert "Director: Director" in chunks[2].text


def test_field_chunker_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError, match="Unsupported field"):
        FieldChunker(["plot", "unknown"])
    with pytest.raises(ValueError, match="at least one"):
        FieldChunker([])


def test_field_chunker_rejects_unknown_private_dispatch(movie: Movie) -> None:
    with pytest.raises(ValueError, match="Unsupported field chunk"):
        FieldChunker(["plot"])._field_text(movie, "unknown")


def test_field_chunker_skips_empty_field_text(movie: Movie) -> None:
    empty_plot_movie = movie.model_copy(update={"title": "", "plot": ""})
    assert FieldChunker(["plot"]).chunk(empty_plot_movie) == []


def test_sentence_chunker_groups_sentence_windows(movie: Movie) -> None:
    chunker = SentenceChunker(min_sentences=2, max_sentences=2)
    chunks = chunker.chunk(movie)

    assert chunker.strategy_name == "sentence"
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunks[0].payload == {
        "min_sentences": 2,
        "max_sentences": 2,
        "sentence_start": 0,
    }
    assert chunks[-1].text.endswith("Fifth sentence.")


def test_sentence_chunker_keeps_full_windows_without_short_tail_merge(movie: Movie) -> None:
    chunks = SentenceChunker(min_sentences=1, max_sentences=2).chunk(movie)
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_sentence_chunker_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="min_sentences must be at least 1"):
        SentenceChunker(min_sentences=0)
    with pytest.raises(ValueError, match="greater than or equal"):
        SentenceChunker(min_sentences=3, max_sentences=2)


def test_sentence_chunker_handles_empty_plot(movie: Movie) -> None:
    assert SentenceChunker().chunk(movie.model_copy(update={"plot": ""})) == []


def test_create_chunker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "chunk_size", 10)
    monkeypatch.setattr(settings, "chunk_overlap", 2)
    monkeypatch.setattr(settings, "chunk_min_sentences", 2)
    monkeypatch.setattr(settings, "chunk_max_sentences", 3)
    monkeypatch.setattr(settings, "chunk_fields", "plot,cast")

    assert isinstance(create_chunker("flat"), FlatChunker)
    assert isinstance(create_chunker("fixed_size"), FixedSizeChunker)
    assert isinstance(create_chunker("sentence"), SentenceChunker)
    assert isinstance(create_chunker("field"), FieldChunker)


def test_create_chunker_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        create_chunker(cast(ChunkingStrategyName, "unknown"))


def test_get_chunker_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "chunking_strategy", "flat")
    assert isinstance(get_chunker(), FlatChunker)
