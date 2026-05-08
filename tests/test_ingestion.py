from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import rag.ingestion.pipeline as pipeline_module
from rag.chunking.base import Chunker, ChunkResult
from rag.chunking.field import FieldChunker
from rag.chunking.flat import FlatChunker
from rag.ingestion.csv_loader import load_movies
from rag.ingestion.pipeline import ingest_csv
from rag.models.movie import Movie


class EmptyChunker(Chunker):
    @property
    def strategy_name(self) -> str:
        return "empty"

    def chunk(self, movie: Movie) -> list[ChunkResult]:
        return []


def test_load_movies_success(tmp_path: Path) -> None:
    """Test successful movie loading and filtering."""
    csv_file = tmp_path / "movies.csv"
    df = pd.DataFrame(
        {
            "Title": ["Movie 1", "Movie 2", "Movie 3", "Movie 4"],
            "Release Year": [2020, 2021, 2022, 2023],
            "Director": ["D1", "D2", "D3", "D4"],
            "Genre": ["G1", "G2", "G3", "G4"],
            "Cast": ["C1", "C2", "C3", "C4"],
            "Origin/Ethnicity": ["American", "British", "French", "American"],
            "Plot": ["Plot 1", "Plot 2", "Plot 3", None],  # One missing plot
        }
    )
    df.to_csv(csv_file, index=False)

    movies = load_movies(str(csv_file))
    # Should have 2: Movie 1 (American), Movie 2 (British).
    assert len(movies) == 2
    assert movies[0].title == "Movie 1"
    assert movies[1].title == "Movie 2"
    assert isinstance(movies[0].genre, list)


def test_load_movies_malformed_row(tmp_path: Path) -> None:
    """Test skipping of malformed rows during movie loading."""
    csv_file = tmp_path / "malformed.csv"
    df = pd.DataFrame(
        {
            "Title": ["Valid", "Invalid Year"],
            "Release Year": [2020, "Not a year"],
            "Director": ["D1", "D2"],
            "Genre": ["G1", "G2"],
            "Cast": ["C1", "C2"],
            "Origin/Ethnicity": ["American", "American"],
            "Plot": ["P1", "P2"],
        }
    )
    df.to_csv(csv_file, index=False)

    movies = load_movies(str(csv_file))
    assert len(movies) == 1
    assert movies[0].title == "Valid"


def test_load_movies_file_not_found() -> None:
    """Test error when CSV file is missing."""
    with pytest.raises(FileNotFoundError):
        load_movies("non_existent.csv")


def test_ingest_csv_success() -> None:
    """Test the full ingestion pipeline orchestration."""
    mock_provider = MagicMock()
    mock_provider.model_info.name = "test-model"
    mock_provider.model_info.dimension = 3
    mock_provider.embed_batch.return_value = [[0.1, 0.2]]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.estimated_cost_usd = 0.01
    mock_provider.get_model_usage.return_value = mock_usage

    mock_store = MagicMock()
    mock_store.target_name.return_value = "movies_test_model_3"

    mock_movie = Movie(
        id=1,
        title="Test Movie",
        release_year=2000,
        director="D",
        genre=["G"],
        cast=["C"],
        plot="Some plot",
    )

    with patch("rag.ingestion.csv_loader.load_movies", return_value=[mock_movie]):
        ingest_csv(mock_provider, mock_store, FlatChunker())

    mock_provider.embed_batch.assert_called_once()
    embedded_texts = mock_provider.embed_batch.call_args.args[0]
    assert embedded_texts == [
        "Title: Test Movie\nRelease Year: 2000\nDirector: D\nGenre: G\nCast: C\nPlot: Some plot"
    ]
    mock_store.upsert_batch.assert_called_once()


def test_ingest_csv_no_movies() -> None:
    """Test pipeline when no movies are loaded."""
    mock_provider = MagicMock()
    mock_store = MagicMock()
    mock_store.target_name.return_value = "movies_test_model_3"

    with patch("rag.ingestion.csv_loader.load_movies", return_value=[]):
        ingest_csv(mock_provider, mock_store, FlatChunker())

    mock_provider.embed_batch.assert_not_called()


def test_ingest_csv_no_chunks() -> None:
    mock_provider = MagicMock()
    mock_provider.model_info.name = "test-model"
    mock_provider.model_info.dimension = 3
    mock_store = MagicMock()
    mock_store.target_name.return_value = "movies_test_model_3"
    mock_movie = Movie(
        id=1,
        title="T",
        release_year=2000,
        director="D",
        genre=["G"],
        cast=["C"],
        plot="P",
    )

    with patch("rag.ingestion.csv_loader.load_movies", return_value=[mock_movie]):
        ingest_csv(mock_provider, mock_store, EmptyChunker())

    mock_provider.embed_batch.assert_not_called()
    mock_store.upsert_batch.assert_not_called()


def test_ingest_csv_embedding_failure() -> None:
    """Test pipeline handling of embedding failures."""
    mock_provider = MagicMock()
    mock_provider.model_info.name = "test-model"
    mock_provider.model_info.dimension = 3
    mock_provider.embed_batch.return_value = []  # Failure
    mock_provider.embed.return_value = []

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 0
    mock_usage.estimated_cost_usd = 0.0
    mock_provider.get_model_usage.return_value = mock_usage

    mock_store = MagicMock()
    mock_store.target_name.return_value = "movies_test_model_3"

    mock_movie = Movie(
        id=1,
        title="T",
        release_year=2000,
        director="D",
        genre=["G"],
        cast=["C"],
        plot="P",
    )

    with patch("rag.ingestion.csv_loader.load_movies", return_value=[mock_movie]):
        ingest_csv(mock_provider, mock_store, FlatChunker())

    mock_store.upsert_batch.assert_not_called()


def test_ingest_csv_falls_back_to_per_movie_embedding() -> None:
    """Test that a failed batch falls back to individual movie embeddings."""
    mock_provider = MagicMock()
    mock_provider.model_info.name = "test-model"
    mock_provider.model_info.dimension = 3
    mock_provider.embed_batch.return_value = []

    movie_one = Movie(
        id=1,
        title="First",
        release_year=2000,
        director="D1",
        genre=["G1"],
        cast=["C1"],
        plot="First plot",
    )
    movie_two = Movie(
        id=2,
        title="Second",
        release_year=2001,
        director="D2",
        genre=["G2"],
        cast=["C2"],
        plot="Second plot",
    )

    mock_provider.embed.side_effect = [[0.1, 0.2, 0.3], []]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 5
    mock_usage.estimated_cost_usd = 0.0
    mock_provider.get_model_usage.return_value = mock_usage

    mock_store = MagicMock()
    mock_store.target_name.return_value = "movies_test_model_3"

    with patch("rag.ingestion.csv_loader.load_movies", return_value=[movie_one, movie_two]):
        ingest_csv(mock_provider, mock_store, FlatChunker())

    mock_provider.embed_batch.assert_called_once_with(
        [
            "Title: First\nRelease Year: 2000\nDirector: D1\nGenre: G1\nCast: C1\nPlot: First plot",
            "Title: Second\nRelease Year: 2001\nDirector: D2\nGenre: G2\nCast: C2\nPlot: Second plot",
        ]
    )
    assert mock_provider.embed.call_count == 2
    inserted_chunk = mock_store.upsert_batch.call_args.args[0][0]
    assert inserted_chunk.source_movie_id == 1
    assert inserted_chunk.chunk_strategy == "flat"
    mock_store.upsert_batch.assert_called_once_with(
        [inserted_chunk],
        [[0.1, 0.2, 0.3]],
        mock_provider.model_info,
    )


def test_ingest_csv_writes_skipped_movies_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that ingestion persists machine-readable skipped movie details."""
    report_dir = tmp_path / "reports"
    monkeypatch.setattr(pipeline_module, "_REPORT_DIR", report_dir)
    monkeypatch.setattr(
        pipeline_module,
        "_SKIPPED_MOVIES_REPORT_PATH",
        report_dir / "skipped-movies.json",
    )
    monkeypatch.setattr(pipeline_module, "_COST_REPORT_PATH", report_dir / "cost-report.json")
    monkeypatch.setattr(pipeline_module, "_OUTPUT_ENV_PATH", tmp_path / "ingestion-outputs.env")

    mock_provider = MagicMock()
    mock_provider.model_info.name = "test-model"
    mock_provider.model_info.dimension = 3
    mock_provider.embed_batch.return_value = []
    mock_provider.embed.side_effect = [[0.1, 0.2, 0.3], []]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 5
    mock_usage.total_tokens = 5
    mock_usage.estimated_cost_usd = 0.0
    mock_provider.get_model_usage.return_value = mock_usage

    mock_store = MagicMock()
    mock_store.target_name.return_value = "movies_test_model_3"

    movie_one = Movie(
        id=1,
        title="First",
        release_year=2000,
        director="D1",
        genre=["G1"],
        cast=["C1"],
        plot="First plot",
    )
    movie_two = Movie(
        id=2,
        title="Second",
        release_year=2001,
        director="D2",
        genre=["G2"],
        cast=["C2"],
        plot="Second plot that fails",
    )

    with patch("rag.ingestion.csv_loader.load_movies", return_value=[movie_one, movie_two]):
        ingest_csv(mock_provider, mock_store, FlatChunker())

    report = (report_dir / "skipped-movies.json").read_text(encoding="utf-8")
    assert '"skipped_chunk_count": 1' in report
    assert '"id": 2' in report
    assert '"title": "Second"' in report
    assert '"chunking_strategy": "flat"' in report
    assert '"reason": "embedding_failed_after_batch_fallback"' in report


def test_ingest_csv_expands_field_chunks() -> None:
    mock_provider = MagicMock()
    mock_provider.model_info.name = "test-model"
    mock_provider.model_info.dimension = 3
    mock_provider.embed_batch.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.total_tokens = 10
    mock_usage.estimated_cost_usd = 0.0
    mock_provider.get_model_usage.return_value = mock_usage

    mock_store = MagicMock()
    mock_store.target_name.return_value = "movies_test_model_3"

    movie = Movie(
        id=7,
        title="Field Test",
        release_year=2001,
        director="Director",
        genre=["Drama"],
        cast=["Actor One", "Actor Two"],
        plot="A plot",
    )

    with patch("rag.ingestion.csv_loader.load_movies", return_value=[movie]):
        ingest_csv(mock_provider, mock_store, FieldChunker(["plot", "cast"]))

    embedded_texts = mock_provider.embed_batch.call_args.args[0]
    assert embedded_texts == [
        "Title: Field Test\nPlot: A plot",
        "Title: Field Test\nCast: Actor One, Actor Two",
    ]
    inserted_chunks = mock_store.upsert_batch.call_args.args[0]
    assert [chunk.chunk_field for chunk in inserted_chunks] == ["plot", "cast"]
    assert [chunk.source_movie_id for chunk in inserted_chunks] == [7, 7]
    assert [chunk.id for chunk in inserted_chunks] == [7_000_000, 7_000_001]
