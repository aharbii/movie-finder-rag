from abc import ABC, abstractmethod

from pydantic import BaseModel


class EmbeddingModel(BaseModel):
    name: str
    description: str = ""
    token_price: float = 0.0
    embedding_dimension: int = 0
    valid_embedding_dimensions: list[int] = []
    context_length: int = 0
    max_input: int = 0


class EmbeddingModelUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0


class EmbeddingProvider(ABC):
    @abstractmethod
    def __init__(self, model: EmbeddingModel, debug: bool) -> None:
        self.model: EmbeddingModel = EmbeddingModel(name="base")

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass

    @abstractmethod
    def get_model_usage(self) -> EmbeddingModelUsage:
        pass
