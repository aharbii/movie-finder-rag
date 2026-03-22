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
    DATASET_PATH = os.path.join(
        DATASET_DIR, "wiki_movie", "wiki_movie_plots_deduped.csv"
    )

    MOVIE_DESCRIPTION_TEMPLATE = """Title: {title}

Release Year: {release_year}

Director: {director}

Genre: {genre}

Cast: {cast}

Plot:
{plot}
"""

    df = pd.read_csv(DATASET_PATH)

    filtered_origin = ["American", "British"]
    df = df[df["Origin/Ethnicity"].isin(filtered_origin)]

    movies_description = []

    for id, movie_entry in df.iterrows():
        movie = parse_movie(movie_entry)
        movie.id = id

        description = MOVIE_DESCRIPTION_TEMPLATE.format(
            title=movie.title,
            release_year=movie.release_year,
            director=movie.director,
            genre=movie.genre,
            cast=movie.cast,
            plot=movie.plot,
        )
        movies_description.append((movie, description))

    return movies_description
