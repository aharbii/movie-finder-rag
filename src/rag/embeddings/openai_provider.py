from typing import cast

from openai import OpenAI
from openai.types.create_embedding_response import Usage

from rag.config import settings
from rag.embeddings.base import (
    EmbeddingModelMetadata,
    EmbeddingModelUsage,
    EmbeddingProvider,
)
from rag.utils.logger import get_logger


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI-specific embedding provider implementing cost and usage tracking.
    """

    # Authoritative model map using universal metadata structure
    MODELS: dict[str, EmbeddingModelMetadata] = {
        "text-embedding-3-small": EmbeddingModelMetadata(
            name="text-embedding-3-small", dimension=1536, cost_per_1k_tokens=0.00002
        ),
        "text-embedding-3-large": EmbeddingModelMetadata(
            name="text-embedding-3-large", dimension=3072, cost_per_1k_tokens=0.00013
        ),
        "text-embedding-ada-002": EmbeddingModelMetadata(
            name="text-embedding-ada-002", dimension=1536, cost_per_1k_tokens=0.00010
        ),
    }

    def __init__(self, model: str = settings.openai_embedding_model, debug: bool = False) -> None:
        """
        Initialize the OpenAI provider using settings.

        Args:
            model (str): The OpenAI model name to use.
            debug (bool): Enable verbose logging.
        """
        self.logger = get_logger(self.__class__.__name__, debug)
        self.model = model

        if self.model not in self.MODELS:
            self.logger.warning(
                f"Model '{self.model}' is not in the authoritative metadata map. "
                "Usage and cost tracking may be inaccurate."
            )

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not defined in settings")

        self.client = OpenAI(api_key=settings.openai_api_key)
        self._usage = EmbeddingModelUsage()

    @property
    def model_info(self) -> EmbeddingModelMetadata:
        """Return the standardized model metadata."""
        metadata = self.MODELS.get(self.model)
        if metadata:
            return metadata

        return EmbeddingModelMetadata(
            name=self.model, dimension=settings.embedding_dimension, cost_per_1k_tokens=0.0
        )

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single string of text."""
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
            self._update_usage(response.usage)
            return cast(list[float], response.data[0].embedding)
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of strings."""
        try:
            response = self.client.embeddings.create(model=self.model, input=texts)
            self._update_usage(response.usage)
            return [cast(list[float], data.embedding) for data in response.data]
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API for batch: {e}")
            return []

    def get_model_usage(self) -> EmbeddingModelUsage:
        """Return cumulative usage and estimated cost for this instance."""
        return self._usage

    def _update_usage(self, usage: Usage) -> None:
        """Calculate cost and update usage statistics."""
        metadata = self.MODELS.get(self.model)
        cost_per_1k = metadata.cost_per_1k_tokens if metadata else 0.0

        self._usage.prompt_tokens += usage.prompt_tokens
        self._usage.total_tokens += usage.total_tokens

        # Calculate USD cost: (tokens / 1000) * price_per_1k
        incremental_cost = (usage.total_tokens / 1000) * cost_per_1k
        self._usage.estimated_cost_usd += incremental_cost

        self.logger.debug(
            f"Model: {self.model} | Total Tokens: {self._usage.total_tokens} | "
            f"Est. Cost: ${self._usage.estimated_cost_usd:.5f}"
        )
