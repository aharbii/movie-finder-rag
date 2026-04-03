from pathlib import Path
from unittest.mock import patch

import pytest

from rag.dataset.dataset import download_data


def test_download_data_already_exists(tmp_path: Path) -> None:
    """Test download_data when the dataset directory already exists and is not empty."""
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "dummy.csv").write_text("data")

    with patch("rag.dataset.dataset.Path", return_value=dataset_path):
        path = download_data()
        assert path == str(dataset_path)


def test_download_data_success(tmp_path: Path) -> None:
    """Test successful dataset download and file movement."""
    dataset_path = tmp_path / "dataset"
    kaggle_cache = tmp_path / "kaggle_cache"
    kaggle_cache.mkdir()
    (kaggle_cache / "movie_plots.csv").write_text("csv content")
    (kaggle_cache / "subdir").mkdir()
    (kaggle_cache / "subdir" / "info.txt").write_text("info")

    with (
        patch("rag.dataset.dataset.Path", return_value=dataset_path),
        patch("kagglehub.dataset_download", return_value=str(kaggle_cache)),
        patch("os.listdir", return_value=["movie_plots.csv", "subdir"]),
        patch("shutil.copy2") as mock_copy,
        patch("shutil.copytree") as mock_copytree,
    ):
        path = download_data()
        assert path == str(dataset_path)
        # Check if mkdir was called (it would be called on the mocked Path)
        assert dataset_path.exists()
        mock_copy.assert_called()
        mock_copytree.assert_called()


def test_download_data_error() -> None:
    """Test error handling when kagglehub fails."""
    with (
        patch("rag.dataset.dataset.Path.exists", return_value=False),
        patch("kagglehub.dataset_download", side_effect=Exception("Download failed")),
        pytest.raises(RuntimeError, match="Could not download dataset"),
    ):
        download_data()
