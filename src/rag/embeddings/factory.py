from collections.abc import Callable
from functools import lru_cache

from rag.config import EmbeddingProviderName, settings
from rag.embeddings.base import EmbeddingProvider
from rag.embeddings.gemini_provider import GeminiEmbeddingProvider
from rag.embeddings.ollama_provider import OllamaEmbeddingProvider
from rag.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag.embeddings.sentence_transformers_provider import SentenceTransformersEmbeddingProvider

ProviderBuilder = Callable[[str | None], EmbeddingProvider]

_PROVIDERS: dict[EmbeddingProviderName, ProviderBuilder] = {
    "openai": lambda model: OpenAIEmbeddingProvider(model=model),
    "ollama": lambda model: OllamaEmbeddingProvider(model=model),
    "huggingface": lambda model: SentenceTransformersEmbeddingProvider(model=model),
    "sentence-transformers": lambda model: SentenceTransformersEmbeddingProvider(model=model),
    "google": lambda model: GeminiEmbeddingProvider(model=model),
}


def create_embedding_provider(
    provider_name: EmbeddingProviderName | None = None,
    model: str | None = None,
) -> EmbeddingProvider:
    """Instantiate the configured embedding provider without caching."""
    resolved_provider = provider_name or settings.embedding_provider
    provider_builder = _PROVIDERS[resolved_provider]
    return provider_builder(model)


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider as a process singleton."""
    return create_embedding_provider()
