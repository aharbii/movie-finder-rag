import os
import shutil
import sys
from pathlib import Path

import kagglehub
from dotenv import load_dotenv
from kagglehub import KaggleDatasetAdapter

from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

DATASET_DIR = os.path.join(os.getcwd(), "dataset", "data")


def download_data(update: bool = True) -> None:
    DATASET_HANDLE = os.path.join("jrobischon", "wikipedia-movie-plots")
    DATASET_NAME = "wiki_movie_plots_deduped.csv"

    TARGET_DATASET_DIR = os.path.join(DATASET_DIR, "wiki_movie")

    if not update and os.path.exists(TARGET_DATASET_DIR):
        logger.info("Dataset exists, skipping downloading")
        return
    else:
        os.makedirs(TARGET_DATASET_DIR, exist_ok=True)

    HOME_DIR = Path.home()
    KAGGLE_CACHE_DIR = os.path.join(HOME_DIR, ".cache", "kagglehub", "datasets")
    DATASET_PATH = os.path.join(KAGGLE_CACHE_DIR, DATASET_HANDLE, "versions", "1", DATASET_NAME)

    try:
        kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            DATASET_HANDLE,
            DATASET_NAME,
        )
        logger.info(f"Wiki-Movie Plots dataset downloaded to {DATASET_PATH}")
    except Exception as error:
        logger.error(f"Failed to download Wiki-Movie Plots dataset, error message {error}")
        sys.exit(1)

    shutil.move(DATASET_PATH, os.path.join(TARGET_DATASET_DIR, DATASET_NAME))
    logger.info(f"Wiki-Movie Plots dataset moved to {TARGET_DATASET_DIR}")
