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
    source_movie_id: int | None = Field(
        None, description="Original movie identifier when this record represents a chunk"
    )
    chunk_index: int | None = Field(None, description="Zero-based chunk index")
    chunk_count: int | None = Field(None, description="Total chunks emitted for the source movie")
    chunk_strategy: str | None = Field(None, description="Chunking strategy that produced the text")
    chunk_field: str | None = Field(None, description="Source field for field-based chunks")
