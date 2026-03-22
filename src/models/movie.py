from pydantic import BaseModel


class Movie(BaseModel):
    title: str
    id: int = 0
    release_year: int = 0
    director: str = ""
    genre: str | list[str] = []
    cast: str | list[str] = []
    plot: str = ""
