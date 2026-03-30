from typing import cast

from google import genai

from rag.config import settings
from rag.embeddings.base import (
    EmbeddingModelMetadata,
    EmbeddingModelUsage,
    EmbeddingProvider,
)
from rag.utils.logger import get_logger


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Google Gemini-specific embedding provider.
    """

    # Model metadata map
    MODELS: dict[str, EmbeddingModelMetadata] = {
        "text-embedding-004": EmbeddingModelMetadata(
            name="text-embedding-004", dimension=768, cost_per_1k_tokens=0.0
        ),
    }

    def __init__(self, model: str = settings.gemini_embedding_model, debug: bool = False) -> None:
        """Initialize the Gemini provider using settings."""
        self.logger = get_logger(self.__class__.__name__, debug)
        self.model = model

        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not defined in settings")

        self.client = genai.Client(api_key=settings.google_api_key)
        self._usage = EmbeddingModelUsage()

    @property
    def model_info(self) -> EmbeddingModelMetadata:
        """Metadata for the current Gemini model."""
        metadata = self.MODELS.get(self.model)
        if metadata:
            return metadata

        return EmbeddingModelMetadata(
            name=self.model,
            dimension=768,  # Default for Gemini text-embedding-004
            cost_per_1k_tokens=0.0,
        )

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text using Gemini."""
        try:
            response = self.client.models.embed_content(model=self.model, contents=[text])
            if not response.embeddings:
                return []

            return cast(list[float], response.embeddings[0].values)
        except Exception as e:
            self.logger.error(f"Error calling Gemini API: {e}")
            return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single batch call."""
        try:
            response = self.client.models.embed_content(model=self.model, contents=texts)
            if not response.embeddings:
                return []

            return [cast(list[float], emb.values) for emb in response.embeddings]
        except Exception as e:
            self.logger.error(f"Error calling Gemini API for batch: {e}")
            return []

    def get_model_usage(self) -> EmbeddingModelUsage:
        """Returns the calculated tokens and estimated cost."""
        return self._usage
