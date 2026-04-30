from typing import cast

from google import genai

from rag.config import infer_embedding_dimension, settings
from rag.embeddings.base import EmbeddingModelMetadata, EmbeddingModelUsage, EmbeddingProvider
from rag.utils.logger import get_logger


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini embedding provider."""

    MODELS: dict[str, EmbeddingModelMetadata] = {
        "text-embedding-004": EmbeddingModelMetadata(
            name="text-embedding-004", dimension=768, cost_per_1k_tokens=0.0
        ),
    }

    def __init__(self, model: str | None = None) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.model = model or settings.embedding_model

        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not defined in settings")
        if settings.embedding_dimension_override is not None:
            raise ValueError("GOOGLE embeddings do not support EMBEDDING_DIMENSION in this repo.")

        self.client = genai.Client(api_key=settings.google_api_key)
        self._usage = EmbeddingModelUsage()

    @property
    def model_info(self) -> EmbeddingModelMetadata:
        metadata = self.MODELS.get(self.model)
        if metadata is not None:
            return metadata

        dimension = infer_embedding_dimension("google", self.model)
        if dimension <= 0:
            raise ValueError(
                f"Unknown Google embedding dimension for model '{self.model}'. "
                "Add the model to the metadata map before using it for ingestion."
            )

        return EmbeddingModelMetadata(name=self.model, dimension=dimension, cost_per_1k_tokens=0.0)

    def embed(self, text: str) -> list[float]:
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.models.embed_content(model=self.model, contents=texts)
            if not response.embeddings:
                return []
            self._usage.prompt_tokens += len(texts)
            self._usage.total_tokens += len(texts)
            return [cast(list[float], embedding.values) for embedding in response.embeddings]
        except Exception as exc:
            self.logger.error("Error calling Gemini embedding API: %s", exc)
            return []

    def get_model_usage(self) -> EmbeddingModelUsage:
        return self._usage
