import os
import sys
from typing import cast

from openai import OpenAI
from openai.types.create_embedding_response import Usage

from embeddings.base import EmbeddingModel, EmbeddingModelUsage, EmbeddingProvider
from utils.logger import get_logger

OPENAI_EMBEDDING_PRICE_BATCH_SIZE = 1_000_000


def _calculate_token_price(batch_price: float) -> float:
    return batch_price / OPENAI_EMBEDDING_PRICE_BATCH_SIZE


class OpenAIEmbeddingProvider(EmbeddingProvider):
    EMBEDDING_3_SMALL_MODEL = EmbeddingModel(
        name="text-embedding-3-small",
        token_price=_calculate_token_price(0.02),
        embedding_dimension=1536,
        max_input=8192,
    )
    EMBEDDING_3_LARGE_MODEL = EmbeddingModel(
        name="text-embedding-3-large",
        token_price=_calculate_token_price(0.13),
        embedding_dimension=3072,
        max_input=8192,
    )
    EMBEDDING_ADA_002_MODEL = EmbeddingModel(
        name="text-embedding-ada-002",
        token_price=_calculate_token_price(0.10),
        embedding_dimension=1536,
        max_input=8192,
    )

    def __init__(
        self, model: EmbeddingModel = EMBEDDING_ADA_002_MODEL, debug: bool = False
    ):
        self.logger = get_logger(self.__class__.__name__, debug)

        self._validate_env()

        self.model = model
        self.usage = EmbeddingModelUsage()
        self.client = OpenAI()

    def embed(self, text: str) -> list[float]:
        self.logger.debug(f"Embedding '{text}'")
        response = self.client.embeddings.create(input=text, model=self.model.name)

        self.logger.debug(f"Data size: {len(response.data)}")
        for data in response.data:
            self.logger.debug(f"Embedded size[{data.index}]: {len(data.embedding)}")
        self.logger.debug(
            f"Prompt tokens: {response.usage.prompt_tokens}, Total tokens: {response.usage.total_tokens}"
        )
        self.logger.debug(
            f"Embedding Cost: {response.usage.total_tokens * self.model.token_price} USD"
        )

        self._process_usage(response.usage)
        return cast(list[float], response.data[0].embedding)

    def get_model_usage(self) -> EmbeddingModelUsage:
        self.logger.debug(f"Prompt tokens used: {self.usage.prompt_tokens}")
        self.logger.debug(f"Total tokens used: {self.usage.total_tokens}")
        self.logger.debug(f"Total cost: {self.usage.total_cost} USD")

        return self.usage

    def _process_usage(self, usage: Usage) -> None:
        self.usage.prompt_tokens += usage.prompt_tokens
        self.usage.total_tokens += usage.total_tokens
        self.usage.total_cost += usage.total_tokens * self.model.token_price

    def _validate_env(self) -> None:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            self.logger.error("OPENAI_API_KEY is not defined")
            sys.exit(1)
