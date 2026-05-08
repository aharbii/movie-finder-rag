import os
import re
from typing import Any, Literal, cast

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProviderName = Literal[
    "openai",
    "ollama",
    "huggingface",
    "sentence-transformers",
    "google",
]
VectorStoreName = Literal["qdrant", "chromadb", "pinecone", "pgvector"]
ChunkingStrategyName = Literal["flat", "fixed_size", "sentence", "field"]

DEFAULT_EMBEDDING_MODELS: dict[EmbeddingProviderName, str] = {
    "openai": "text-embedding-3-large",
    "ollama": "bge-m3",
    "huggingface": "BAAI/bge-m3",
    "sentence-transformers": "BAAI/bge-m3",
    "google": "text-embedding-004",
}
DEFAULT_VALIDATION_QUERY = "A time-travel movie with a scientist and a DeLorean"
DEFAULT_ENV_FILE = os.getenv("RAG_ENV_FILE", ".env")

KNOWN_MODEL_DIMENSIONS: dict[EmbeddingProviderName, dict[str, int]] = {
    "openai": {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    },
    "ollama": {
        "all-minilm": 384,
        "all-minilm:latest": 384,
        "bge-m3": 1024,
        "bge-m3:latest": 1024,
        "embeddinggemma": 768,
        "embeddinggemma:latest": 768,
        "mxbai-embed-large": 1024,
        "mxbai-embed-large:latest": 1024,
        "nomic-embed-text": 768,
        "nomic-embed-text:latest": 768,
    },
    "sentence-transformers": {
        "BAAI/bge-m3": 1024,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
    },
    "huggingface": {
        "BAAI/bge-m3": 1024,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
    },
    "google": {
        "text-embedding-004": 768,
    },
}

_COLLECTION_PREFIX_RE = re.compile(r"^[a-z0-9_]+$")
_PINECONE_INDEX_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,43}[a-z0-9])?$")
_LEGACY_TOKEN_RE = re.compile(r"[\/.\-\s]+")
_INVALID_TOKEN_RE = re.compile(r"[^a-z0-9_]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def infer_embedding_dimension(provider: EmbeddingProviderName, model: str) -> int:
    """Best-effort embedding dimension lookup for config and reporting."""
    known_dimension = KNOWN_MODEL_DIMENSIONS.get(provider, {}).get(model)
    if known_dimension is not None:
        return known_dimension

    if provider == "openai":
        if "large" in model:
            return 3072
        if "small" in model or "ada" in model:
            return 1536

    if provider == "google":
        return 768

    return 0


class RAGConfig(BaseSettings):
    """Project-wide settings and configuration managed by Pydantic."""

    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str | None = Field(None, validation_alias="QDRANT_URL")
    qdrant_api_key_rw: str | None = Field(None, validation_alias="QDRANT_API_KEY_RW")
    vector_collection_prefix: str = Field("movies", validation_alias="VECTOR_COLLECTION_PREFIX")

    vector_store: VectorStoreName = Field("qdrant", validation_alias="VECTOR_STORE")
    chromadb_persist_path: str = Field(
        "outputs/chromadb/local", validation_alias="CHROMADB_PERSIST_PATH"
    )
    pinecone_api_key: str | None = Field(None, validation_alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field("movie-finder-rag", validation_alias="PINECONE_INDEX_NAME")
    pinecone_index_host: str | None = Field(None, validation_alias="PINECONE_INDEX_HOST")
    pinecone_cloud: str = Field("aws", validation_alias="PINECONE_CLOUD")
    pinecone_region: str = Field("us-east-1", validation_alias="PINECONE_REGION")
    pgvector_dsn: str | None = Field(None, validation_alias="PGVECTOR_DSN")
    pgvector_schema: str = Field("public", validation_alias="PGVECTOR_SCHEMA")

    embedding_provider: EmbeddingProviderName = Field(
        "openai", validation_alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(
        DEFAULT_EMBEDDING_MODELS["openai"], validation_alias="EMBEDDING_MODEL"
    )
    embedding_dimension_override: int | None = Field(
        None, validation_alias="EMBEDDING_DIMENSION", ge=1
    )
    openai_api_key: str | None = Field(None, validation_alias="OPENAI_API_KEY")
    google_api_key: str | None = Field(None, validation_alias="GOOGLE_API_KEY")
    ollama_base_url: str | None = Field(None, validation_alias="OLLAMA_BASE_URL")
    ollama_api_key: str | None = Field(None, validation_alias="OLLAMA_API_KEY")
    sentence_transformers_cache_dir: str | None = Field(
        None, validation_alias="SENTENCE_TRANSFORMERS_CACHE_DIR"
    )

    batch_size: int = Field(100, validation_alias="BATCH_SIZE", ge=1, le=10_000)
    chunking_strategy: ChunkingStrategyName = Field("flat", validation_alias="CHUNKING_STRATEGY")
    chunk_size: int = Field(160, validation_alias="CHUNK_SIZE", ge=1)
    chunk_overlap: int = Field(32, validation_alias="CHUNK_OVERLAP", ge=0)
    chunk_min_sentences: int = Field(1, validation_alias="CHUNK_MIN_SENTENCES", ge=1)
    chunk_max_sentences: int = Field(4, validation_alias="CHUNK_MAX_SENTENCES", ge=1)
    chunk_fields: str = Field("plot,cast,metadata", validation_alias="CHUNK_FIELDS")
    validation_query: str = Field(
        DEFAULT_VALIDATION_QUERY,
        validation_alias="VALIDATION_QUERY",
    )
    kaggle_api_token: str | None = Field(None, validation_alias="KAGGLE_API_TOKEN")

    @model_validator(mode="before")
    @classmethod
    def _apply_dynamic_defaults(cls, values: Any) -> Any:
        """Populate provider-dependent defaults before field validation."""
        if not isinstance(values, dict):
            return values

        normalized = dict(values)
        provider = normalized.get("EMBEDDING_PROVIDER") or normalized.get("embedding_provider")
        if not isinstance(provider, str) or not provider.strip():
            provider = "openai"
        provider = provider.strip()
        if provider not in DEFAULT_EMBEDDING_MODELS:
            provider = "openai"
        model = normalized.get("EMBEDDING_MODEL") or normalized.get("embedding_model")
        dimensions = normalized.get(
            "EMBEDDING_DIMENSION", normalized.get("embedding_dimension_override")
        )
        validation_query = normalized.get("VALIDATION_QUERY", normalized.get("validation_query"))

        if not isinstance(model, str) or not model.strip():
            provider_name = cast(EmbeddingProviderName, provider)
            normalized["EMBEDDING_MODEL"] = DEFAULT_EMBEDDING_MODELS[provider_name]
        if isinstance(dimensions, str) and not dimensions.strip():
            normalized["EMBEDDING_DIMENSION"] = None
        if not isinstance(validation_query, str) or not validation_query.strip():
            normalized["VALIDATION_QUERY"] = DEFAULT_VALIDATION_QUERY

        return normalized

    @field_validator("embedding_model", "validation_query")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value must not be empty.")
        return cleaned

    @field_validator("chunk_fields")
    @classmethod
    def _validate_chunk_fields(cls, value: str) -> str:
        cleaned = ",".join(field.strip().lower() for field in value.split(",") if field.strip())
        if not cleaned:
            raise ValueError("CHUNK_FIELDS must include at least one field.")
        return cleaned

    @field_validator("vector_collection_prefix")
    @classmethod
    def _validate_collection_prefix(cls, value: str) -> str:
        cleaned = value.strip().lower()
        cleaned = _LEGACY_TOKEN_RE.sub("_", cleaned)
        cleaned = _INVALID_TOKEN_RE.sub("_", cleaned)
        cleaned = _MULTI_UNDERSCORE_RE.sub("_", cleaned).strip("_")
        if not cleaned:
            raise ValueError("VECTOR_COLLECTION_PREFIX must not be empty.")
        if not _COLLECTION_PREFIX_RE.fullmatch(cleaned):
            raise ValueError(
                "VECTOR_COLLECTION_PREFIX must contain only lowercase letters, digits, and underscores."
            )
        return cleaned

    @model_validator(mode="after")
    def _validate_runtime_shape(self) -> "RAGConfig":
        """Validate cross-field provider and vector-store requirements."""
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai.")

        if self.embedding_provider == "google" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when EMBEDDING_PROVIDER=google.")

        if self.embedding_provider == "ollama" and not self.ollama_base_url:
            raise ValueError("OLLAMA_BASE_URL is required when EMBEDDING_PROVIDER=ollama.")

        if self.embedding_provider == "google" and self.embedding_dimension_override is not None:
            raise ValueError("EMBEDDING_DIMENSION is not supported when EMBEDDING_PROVIDER=google.")

        if self.vector_store == "qdrant":
            if not self.qdrant_url:
                raise ValueError("QDRANT_URL is required when VECTOR_STORE=qdrant.")
            if not self.qdrant_api_key_rw:
                raise ValueError("QDRANT_API_KEY_RW is required when VECTOR_STORE=qdrant.")

        if self.vector_store == "pinecone":
            if not self.pinecone_api_key:
                raise ValueError("PINECONE_API_KEY is required when VECTOR_STORE=pinecone.")
            if not _PINECONE_INDEX_NAME_RE.fullmatch(self.pinecone_index_name):
                raise ValueError(
                    "PINECONE_INDEX_NAME must be 1-45 chars, lowercase alphanumeric or '-'."
                )

        if self.vector_store == "pgvector" and not self.pgvector_dsn:
            raise ValueError("PGVECTOR_DSN is required when VECTOR_STORE=pgvector.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        if self.chunk_max_sentences < self.chunk_min_sentences:
            raise ValueError(
                "CHUNK_MAX_SENTENCES must be greater than or equal to CHUNK_MIN_SENTENCES."
            )

        return self

    @property
    def embedding_dimension(self) -> int:
        """Return the configured embedding model dimension when it is known."""
        if self.embedding_dimension_override is not None:
            return self.embedding_dimension_override
        return infer_embedding_dimension(self.embedding_provider, self.embedding_model)

    @property
    def chunk_fields_list(self) -> list[str]:
        """Return field chunk names parsed from CHUNK_FIELDS."""
        return [field.strip() for field in self.chunk_fields.split(",") if field.strip()]


settings = RAGConfig()
