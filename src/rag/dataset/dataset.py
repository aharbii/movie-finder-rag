import os
import shutil
from pathlib import Path

import kagglehub

from rag.utils.logger import get_logger


def download_data() -> str:
    """
    Download the movie dataset from Kaggle if not already present.

    Returns:
        str: The path to the downloaded dataset.
    """
    logger = get_logger(__name__)
    dataset_handle = "jrobischon/wikipedia-movie-plots"
    dataset_path = Path("dataset")

    if dataset_path.exists() and any(dataset_path.iterdir()):
        logger.info(f"Dataset already exists at {dataset_path}")
        return str(dataset_path)

    dataset_path.mkdir(exist_ok=True)

    try:
        logger.info(f"Downloading dataset {dataset_handle} from Kaggle Hub...")
        path = kagglehub.dataset_download(dataset_handle)
        logger.info(f"Dataset downloaded to {path}")

        # Move files to local dataset directory
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(dataset_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

        logger.info(f"Dataset moved to {dataset_path}")
        return str(dataset_path)
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise RuntimeError(f"Could not download dataset: {e}") from e
