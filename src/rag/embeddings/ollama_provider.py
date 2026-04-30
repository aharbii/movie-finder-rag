from typing import Any

from rag.config import infer_embedding_dimension, settings
from rag.embeddings.base import EmbeddingModelMetadata, EmbeddingModelUsage, EmbeddingProvider
from rag.utils.logger import get_logger


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider for local and cloud-hosted Ollama models."""

    def __init__(self, model: str | None = None) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.model = model or settings.embedding_model
        self.dimensions = settings.embedding_dimension_override
        self._usage = EmbeddingModelUsage()

        try:
            from ollama import Client
        except ImportError as exc:
            raise RuntimeError(
                "Ollama support requires the optional provider dependencies. "
                "Set WITH_PROVIDERS=local when building the Docker image."
            ) from exc

        headers = None
        if settings.ollama_api_key:
            headers = {"Authorization": f"Bearer {settings.ollama_api_key}"}
        self.client = Client(host=settings.ollama_base_url, headers=headers)

    @property
    def model_info(self) -> EmbeddingModelMetadata:
        dimension = self.dimensions or infer_embedding_dimension("ollama", self.model)
        if dimension <= 0:
            try:
                response = self.client.embed(model=self.model, input=["dimension probe"])
                embeddings = response.get("embeddings", [])
                if embeddings:
                    dimension = len(embeddings[0])
            except Exception as exc:
                self.logger.error("Unable to probe Ollama embedding dimension: %s", exc)
        if dimension <= 0:
            raise ValueError(
                f"Unknown Ollama embedding dimension for model '{self.model}'. "
                "Add the model to the metadata map before using it for ingestion."
            )
        return EmbeddingModelMetadata(name=self.model, dimension=dimension, cost_per_1k_tokens=0.0)

    def embed(self, text: str) -> list[float]:
        embeddings = self.embed_batch([text])
        return embeddings[0] if embeddings else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            request: dict[str, Any] = {"model": self.model, "input": texts}
            if self.dimensions is not None:
                request["dimensions"] = self.dimensions
            response = self.client.embed(**request)
            embeddings = response.get("embeddings", [])
            self._usage.total_tokens += int(response.get("prompt_eval_count", 0))
            self._usage.prompt_tokens = self._usage.total_tokens
            return [list(cast_embedding) for cast_embedding in embeddings]
        except Exception as exc:
            self.logger.error("Error calling Ollama embedding API: %s", exc)
            return []

    def get_model_usage(self) -> EmbeddingModelUsage:
        return self._usage
