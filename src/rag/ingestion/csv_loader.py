import os

import pandas as pd

from rag.models.movie import Movie
from rag.utils.logger import get_logger


def load_movies(
    file_path: str = "dataset/wiki_movie_plots_deduped.csv", debug: bool = False
) -> list[Movie]:
    """
    Filter the Kaggle dataset for American and British movies and map to Typed objects.

    Args:
        file_path (str): The path to the CSV file.
        debug (bool): Enable debug logging.

    Returns:
        list[Movie]: A list of Movie objects filtered for supported origins.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    logger = get_logger(__name__, debug)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    logger.info(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)

    # Basic data cleaning and mandatory fields check
    df = df.dropna(subset=["Plot", "Title"])

    # Filter for American and British movies only as per design mandate
    supported_origins = ["American", "British"]
    logger.debug(f"Filtering dataset for supported origins: {supported_origins}")

    df_filtered = df[df["Origin/Ethnicity"].isin(supported_origins)]
    logger.info(f"Filtered to {len(df_filtered)} / {len(df)} supported movies.")

    movies = []
    for index, row in df_filtered.iterrows():
        try:
            movie = Movie(
                id=index,  # Use index as a temporary ID
                title=str(row["Title"]),
                release_year=int(row["Release Year"]),
                director=str(row["Director"]),
                genre=[g.strip() for g in str(row["Genre"]).split(",") if g.strip()],
                cast=[c.strip() for c in str(row["Cast"]).split(",") if c.strip()],
                plot=str(row["Plot"]),
            )
            movies.append(movie)
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping malformed row {index}: {e}")

    return movies
