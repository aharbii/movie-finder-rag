from typing import Any, cast

import pytest
from pydantic import ValidationError

from rag.config import (
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_VALIDATION_QUERY,
    RAGConfig,
    infer_embedding_dimension,
)


def apply_base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "chromadb")
    monkeypatch.setenv("CHROMADB_PERSIST_PATH", "outputs/chromadb-test")


def test_infer_embedding_dimension() -> None:
    assert infer_embedding_dimension("openai", "text-embedding-3-large") == 3072
    assert infer_embedding_dimension("openai", "text-embedding-3-small") == 1536
    assert infer_embedding_dimension("openai", "text-embedding-ada-002") == 1536
    assert infer_embedding_dimension("openai", "unknown-large-model") == 3072
    assert infer_embedding_dimension("openai", "unknown-small-model") == 1536
    assert infer_embedding_dimension("openai", "mystery-model") == 0
    assert infer_embedding_dimension("google", "text-embedding-004") == 768
    assert infer_embedding_dimension("google", "unknown") == 768
    assert infer_embedding_dimension(cast("Any", "unknown"), "unknown") == 0


def test_shape_validator_accepts_valid_pinecone_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "pinecone")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "valid-index-123")

    config = RAGConfig()
    assert config.pinecone_index_name == "valid-index-123"


def test_provider_default_model_is_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    config = RAGConfig()
    assert config.embedding_model == DEFAULT_EMBEDDING_MODELS["huggingface"]


def test_collection_prefix_normalizes_model_value(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.delenv("VECTOR_COLLECTION_PREFIX", raising=False)
    monkeypatch.setenv("VECTOR_COLLECTION_PREFIX", "text-embedding-3-large")

    config = RAGConfig()
    assert config.vector_collection_prefix == "text_embedding_3_large"


def test_embedding_dimension_uses_override(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")

    config = RAGConfig()
    assert config.embedding_dimension == 1024


def test_embedding_dimension_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.delenv("EMBEDDING_DIMENSION", raising=False)
    config = RAGConfig()
    assert config.embedding_dimension == 1536


def test_chunking_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    config = RAGConfig()

    assert config.chunking_strategy == "flat"
    assert config.chunk_fields_list == ["plot", "cast", "metadata"]


def test_chunk_fields_are_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("CHUNK_FIELDS", " Plot, Cast ,, Metadata ")
    config = RAGConfig()

    assert config.chunk_fields == "plot,cast,metadata"


def test_chunk_fields_rejects_empty_value(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("CHUNK_FIELDS", " , ")

    with pytest.raises(ValueError, match="CHUNK_FIELDS must include"):
        RAGConfig()


def test_chunk_overlap_must_be_smaller_than_size(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("CHUNK_SIZE", "8")
    monkeypatch.setenv("CHUNK_OVERLAP", "8")

    with pytest.raises(ValueError, match="CHUNK_OVERLAP must be smaller"):
        RAGConfig()


def test_chunk_sentence_bounds_are_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("CHUNK_MIN_SENTENCES", "4")
    monkeypatch.setenv("CHUNK_MAX_SENTENCES", "3")

    with pytest.raises(ValueError, match="CHUNK_MAX_SENTENCES must be"):
        RAGConfig()


def test_shape_validator_requires_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        RAGConfig()


def test_shape_validator_requires_google_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    with pytest.raises(ValueError, match="GOOGLE_API_KEY is required"):
        RAGConfig()


def test_shape_validator_requires_qdrant_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("QDRANT_API_KEY_RW", "")
    with pytest.raises(ValueError, match="QDRANT_URL is required"):
        RAGConfig()


def test_shape_validator_requires_qdrant_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "qdrant")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("QDRANT_URL", "http://test")
    monkeypatch.setenv("QDRANT_API_KEY_RW", "")

    with pytest.raises(ValueError, match="QDRANT_API_KEY_RW is required"):
        RAGConfig()


def test_shape_validator_rejects_google_dimension_override(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "256")

    with pytest.raises(ValueError, match="EMBEDDING_DIMENSION is not supported"):
        RAGConfig()


def test_shape_validator_requires_ollama_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="OLLAMA_BASE_URL is required"):
        RAGConfig()


def test_shape_validator_requires_pinecone_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "pinecone")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="PINECONE_API_KEY is required"):
        RAGConfig()


def test_shape_validator_requires_pinecone_index_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "pinecone")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("PINECONE_API_KEY", "test")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "invalid_name_with_underscores")

    with pytest.raises(ValueError, match="PINECONE_INDEX_NAME must be"):
        RAGConfig()


def test_shape_validator_requires_pgvector_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_STORE", "pgvector")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setenv("PGVECTOR_DSN", "")

    with pytest.raises(ValueError, match="PGVECTOR_DSN is required"):
        RAGConfig()


def test_apply_dynamic_defaults_empty_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "   ")

    with pytest.raises(ValidationError):
        RAGConfig()


def test_apply_dynamic_defaults_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "unknown-provider")

    with pytest.raises(ValidationError):
        RAGConfig()


def test_apply_dynamic_defaults_empty_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("EMBEDDING_DIMENSION", "   ")
    config = RAGConfig()

    assert config.embedding_dimension_override is None


def test_apply_dynamic_defaults_empty_validation_query(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("VALIDATION_QUERY", "   ")
    config = RAGConfig()

    assert config.validation_query == DEFAULT_VALIDATION_QUERY


def test_validate_non_empty_text() -> None:
    with pytest.raises(ValueError, match="Value must not be empty."):
        RAGConfig._validate_non_empty_text("   ")


def test_validate_collection_prefix_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("VECTOR_COLLECTION_PREFIX", " - . / ")

    with pytest.raises(ValueError, match="VECTOR_COLLECTION_PREFIX must not be empty."):
        RAGConfig()


def test_validate_collection_prefix_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("VECTOR_COLLECTION_PREFIX", "valid_prefix")

    from rag import config

    class MockRegex:
        def fullmatch(self, text: str) -> None:
            return None

    monkeypatch.setattr(config, "_COLLECTION_PREFIX_RE", MockRegex())
    with pytest.raises(
        ValueError, match="VECTOR_COLLECTION_PREFIX must contain only lowercase letters"
    ):
        RAGConfig()


def test_validate_collection_prefix_cleans(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_base_env(monkeypatch)
    monkeypatch.setenv("VECTOR_COLLECTION_PREFIX", "A-b/C.d_e")
    config = RAGConfig()

    assert config.vector_collection_prefix == "a_b_c_d_e"


def test_apply_dynamic_defaults_not_dict() -> None:
    # Just test the classmethod directly
    assert RAGConfig._apply_dynamic_defaults("not_a_dict") == "not_a_dict"  # type: ignore[operator]
