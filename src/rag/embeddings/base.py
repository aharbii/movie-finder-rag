from abc import ABC, abstractmethod

from pydantic import BaseModel


class EmbeddingModelMetadata(BaseModel):
    """
    Standardized metadata for any embedding model, regardless of provider.
    Includes dimensions and pricing for cost tracking.
    """

    name: str
    dimension: int
    cost_per_1k_tokens: float = 0.0


class EmbeddingModelUsage(BaseModel):
    """
    Usage statistics and estimated cost for an embedding run.
    """

    prompt_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers (e.g., OpenAI, Gemini).
    """

    @property
    @abstractmethod
    def model_info(self) -> EmbeddingModelMetadata:
        """Return standardized metadata about the current model."""
        pass

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Embed a single string of text into a vector.

        Args:
            text (str): The text string to embed.

        Returns:
            list[float]: The generated embedding vector.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of strings into vectors.

        Args:
            texts (list[str]): The list of text strings to embed.

        Returns:
            list[list[float]]: The list of generated embedding vectors.
        """
        pass

    @abstractmethod
    def get_model_usage(self) -> EmbeddingModelUsage:
        """
        Return the cumulative usage and cost for the current session.

        Returns:
            EmbeddingModelUsage: The usage statistics for the current provider instance.
        """
        pass
