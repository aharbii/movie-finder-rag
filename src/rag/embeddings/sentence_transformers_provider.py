from typing import cast

from rag.config import infer_embedding_dimension, settings
from rag.embeddings.base import EmbeddingModelMetadata, EmbeddingModelUsage, EmbeddingProvider
from rag.utils.logger import get_logger


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    """Sentence-Transformers provider for offline local development."""

    def __init__(self, model: str | None = None) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.model = model or settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self._usage = EmbeddingModelUsage()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Sentence-Transformers support requires the optional provider dependencies. "
                "Set WITH_PROVIDERS=local when building the Docker image."
            ) from exc

        self.client = SentenceTransformer(
            self.model,
            cache_folder=settings.sentence_transformers_cache_dir,
        )

    @property
    def model_info(self) -> EmbeddingModelMetadata:
        dimension = int(self.client.get_sentence_embedding_dimension() or 0)
        if self.dimensions is not None:
            dimension = self.dimensions
        elif dimension <= 0:
            dimension = infer_embedding_dimension("sentence-transformers", self.model)
        if dimension <= 0:
            raise ValueError(
                f"Unable to resolve embedding dimension for sentence-transformers model '{self.model}'."
            )
        return EmbeddingModelMetadata(name=self.model, dimension=dimension, cost_per_1k_tokens=0.0)

    def embed(self, text: str) -> list[float]:
        embeddings = self.embed_batch([text])
        return embeddings[0] if embeddings else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self.client.encode(
                texts,
                convert_to_numpy=True,
                truncate_dim=self.dimensions,
            )
            self._usage.prompt_tokens += len(texts)
            self._usage.total_tokens += len(texts)
            return cast(list[list[float]], vectors.tolist())
        except Exception as exc:
            self.logger.error("Error generating sentence-transformers embeddings: %s", exc)
            return []

    def get_model_usage(self) -> EmbeddingModelUsage:
        return self._usage
