from openai import OpenAI
from openai.types.create_embedding_response import Usage

from rag.config import infer_embedding_dimension, settings
from rag.embeddings.base import EmbeddingModelMetadata, EmbeddingModelUsage, EmbeddingProvider
from rag.utils.logger import get_logger


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider with usage and cost tracking."""

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

    def __init__(self, model: str | None = None) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.model = model or settings.embedding_model
        self.dimensions = settings.embedding_dimensions

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not defined in settings")

        self.client = OpenAI(api_key=settings.openai_api_key)
        self._usage = EmbeddingModelUsage()

    @property
    def model_info(self) -> EmbeddingModelMetadata:
        metadata = self.MODELS.get(self.model)
        if metadata is not None:
            if self.dimensions is not None:
                return metadata.model_copy(update={"dimension": self.dimensions})
            return metadata

        dimension = self.dimensions or infer_embedding_dimension("openai", self.model)
        if dimension <= 0:
            raise ValueError(
                f"Unknown OpenAI embedding dimension for model '{self.model}'. "
                "Add the model to the metadata map before using it for ingestion."
            )

        return EmbeddingModelMetadata(name=self.model, dimension=dimension, cost_per_1k_tokens=0.0)

    def embed(self, text: str) -> list[float]:
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            if self.dimensions is not None:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    dimensions=self.dimensions,
                )
            else:
                response = self.client.embeddings.create(model=self.model, input=texts)
            self._update_usage(response.usage)
            return [data.embedding for data in response.data]
        except Exception as exc:
            self.logger.error("Error calling OpenAI embeddings API: %s", exc)
            return []

    def get_model_usage(self) -> EmbeddingModelUsage:
        return self._usage

    def _update_usage(self, usage: Usage) -> None:
        metadata = self.MODELS.get(self.model)
        cost_per_1k = metadata.cost_per_1k_tokens if metadata else 0.0
        self._usage.prompt_tokens += usage.prompt_tokens
        self._usage.total_tokens += usage.total_tokens
        self._usage.estimated_cost_usd += (usage.total_tokens / 1000) * cost_per_1k
