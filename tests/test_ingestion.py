from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rag.ingestion.csv_loader import load_movies
from rag.ingestion.pipeline import ingest_csv
from rag.models.movie import Movie


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
    mock_provider.embed_batch.return_value = [[0.1, 0.2]]

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.estimated_cost_usd = 0.01
    mock_provider.get_model_usage.return_value = mock_usage

    mock_store = MagicMock()

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
        ingest_csv(mock_provider, mock_store)

    mock_provider.embed_batch.assert_called_once()
    mock_store.upsert_batch.assert_called_once()


def test_ingest_csv_no_movies() -> None:
    """Test pipeline when no movies are loaded."""
    mock_provider = MagicMock()
    mock_store = MagicMock()

    with patch("rag.ingestion.csv_loader.load_movies", return_value=[]):
        ingest_csv(mock_provider, mock_store)

    mock_provider.embed_batch.assert_not_called()


def test_ingest_csv_embedding_failure() -> None:
    """Test pipeline handling of embedding failures."""
    mock_provider = MagicMock()
    mock_provider.embed_batch.return_value = []  # Failure

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 0
    mock_usage.estimated_cost_usd = 0.0
    mock_provider.get_model_usage.return_value = mock_usage

    mock_store = MagicMock()

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
        ingest_csv(mock_provider, mock_store)

    mock_store.upsert_batch.assert_not_called()
