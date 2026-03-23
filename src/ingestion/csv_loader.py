import os

import pandas as pd

from models.movie import Movie


def parse_movie(movie_entry: pd.Series) -> Movie:
    movie = Movie(
        title=movie_entry["Title"],
        release_year=movie_entry["Release Year"],
        director=movie_entry["Director"],
        genre=str(movie_entry["Genre"]).split("/"),
        cast=str(movie_entry["Cast"]).split(", "),
        plot=movie_entry["Plot"],
    )
    return movie


def parse_wiki_movie() -> list[tuple[Movie, str]]:
    DATASET_DIR = os.path.join(os.getcwd(), "dataset", "data")
    DATASET_PATH = os.path.join(DATASET_DIR, "wiki_movie", "wiki_movie_plots_deduped.csv")

    df = pd.read_csv(DATASET_PATH)

    filtered_origin = ["American", "British"]
    df = df[df["Origin/Ethnicity"].isin(filtered_origin)]

    movies_description = []

    for id, movie_entry in df.iterrows():
        movie = parse_movie(movie_entry)
        movie.id = id

        description = (
            f"Title: {movie.title}\n\n"
            f"Release Year: {movie.release_year}\n\n"
            f"Director: {movie.director}\n\n"
            f"Genre: {movie.genre}\n\n"
            f"Cast: {movie.cast}\n\n"
            f"Plot:\n{movie.plot}\n"
        )
        movies_description.append((movie, description))

    return movies_description
