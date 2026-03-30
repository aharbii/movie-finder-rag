from pydantic import BaseModel, Field


class Movie(BaseModel):
    """
    Standard Typed model for representing a movie in the RAG pipeline.
    """

    id: int = Field(..., description="Unique record identifier")
    title: str = Field(..., description="Movie title")
    release_year: int = Field(..., description="Year of release")
    director: str = Field(..., description="Director name")
    genre: list[str] = Field(..., description="Movie genre")
    cast: list[str] = Field(..., description="Cast list")
    plot: str = Field(..., description="Detailed plot description used for embedding")
