from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGConfig(BaseSettings):
    """
    Project-wide settings and configuration managed by Pydantic.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Qdrant Cloud (Write-capable key required for ingestion)
    qdrant_url: str = Field(..., validation_alias="QDRANT_URL")
    qdrant_api_key_rw: str = Field(..., validation_alias="QDRANT_API_KEY_RW")
    qdrant_collection_name: str = Field("movies", validation_alias="QDRANT_COLLECTION_NAME")

    # OpenAI Embeddings
    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        "text-embedding-3-large", validation_alias="OPENAI_EMBEDDING_MODEL"
    )

    # Gemini Embeddings
    google_api_key: str | None = Field(None, validation_alias="GOOGLE_API_KEY")
    gemini_embedding_model: str = Field(
        "text-embedding-004", validation_alias="GEMINI_EMBEDDING_MODEL"
    )

    # Ingestion Configuration
    batch_size: int = 100
    embedding_provider: Literal["openai", "gemini"] = "openai"

    # Kaggle Configuration
    kaggle_api_token: str | None = Field(None, validation_alias="KAGGLE_API_TOKEN")

    @property
    def embedding_dimension(self) -> int:
        """Returns the dimension of the configured embedding model."""
        if self.embedding_provider == "openai":
            if self.openai_embedding_model == "text-embedding-3-large":
                return 3072
            return 1536
        return 768


settings = RAGConfig()
