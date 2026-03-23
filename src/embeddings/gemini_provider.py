import os
import sys
from typing import cast

from google import genai
from google.genai.types import ContentEmbeddingStatistics

from embeddings.base import EmbeddingModel, EmbeddingModelUsage, EmbeddingProvider
from utils.logger import get_logger

GEMINI_EMBEDDING_PRICE_BATCH_SIZE = 1_000_000


def _calculate_token_price(batch_price: float) -> float:
    return batch_price / GEMINI_EMBEDDING_PRICE_BATCH_SIZE


class GeminiEmbeddingProvider(EmbeddingProvider):
    GEMINI_EMBEDDING_001_MODEL = EmbeddingModel(
        name="gemini-embedding-001",
        token_price=_calculate_token_price(0.15),
        embedding_dimension=3072,
        valid_embedding_dimensions=[3072, 768, 1536, 128],
        context_length=0,
        max_input=2048,
    )

    def __init__(self, model: EmbeddingModel = GEMINI_EMBEDDING_001_MODEL, debug: bool = False):
        self.logger = get_logger(self.__class__.__name__, debug)

        self._validate_env()

        if self.gcp_enabled:
            self.logger.info("VertexAI API enabled")
        else:
            self.logger.warning("VertexAI API disabled")

        self.model = model
        self.usage = EmbeddingModelUsage()
        if self.gcp_enabled:
            self.client = genai.Client(
                vertexai=True,
                project=self.gcp_project_id,
                location=self.gcp_region,
            )
        else:
            self.client = genai.Client()

    def embed(self, text: str) -> list[float]:
        self.logger.debug(f"Embedding '{text}'")
        response = self.client.models.embed_content(
            contents=text,
            model=self.model.name,
        )

        if response.embeddings:
            self.logger.debug(f"Data size: {len(response.embeddings)}")
            for index in range(len(response.embeddings)):
                embedding_values = response.embeddings[index].values
                if embedding_values:
                    self.logger.debug(f"Embedded size[{index}]: {len(embedding_values)}")
                statistics = response.embeddings[index].statistics
                if statistics:
                    if statistics.truncated:
                        self.logger.debug(f"Truncated[{index}]: {statistics.truncated}")

            self._process_usage(response.embeddings[0].statistics)
            if response.embeddings[0].values:
                return cast(list[float], response.embeddings[0].values)
        return [0.0]

    def get_model_usage(self) -> EmbeddingModelUsage:
        self.logger.debug(f"Prompt tokens used: {self.usage.prompt_tokens}")
        self.logger.debug(f"Total tokens used: {self.usage.total_tokens}")
        self.logger.debug(f"Total cost: {self.usage.total_cost} USD")

        return self.usage

    def _process_usage(self, usage: ContentEmbeddingStatistics | None) -> None:
        if usage and usage.token_count:
            self.usage.prompt_tokens += int(usage.token_count)
            self.usage.total_tokens += int(usage.token_count)
            self.usage.total_cost += usage.token_count * self.model.token_price

    def _validate_env(self) -> None:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            self.logger.error("GEMINI_API_KEY is not defined")
            sys.exit(1)

        self.gcp_enabled = True

        self.gcp_project_id = os.getenv("GCP_PROJECT_ID")
        if not self.gcp_project_id:
            self.logger.warning("GCP_PROJECT_ID is not defined")
            self.gcp_enabled = False
            return

        self.gcp_region = os.getenv("GCP_REGION")
        if not self.gcp_region:
            self.logger.warning("GCP_REGION is not defined")
            self.gcp_enabled = False
            return

        GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        if not GOOGLE_APPLICATION_CREDENTIALS:
            self.logger.warning("GOOGLE_APPLICATION_CREDENTIALS is not defined")
            self.gcp_enabled = False
            return
        else:
            credentials_file_path = os.path.join(os.getcwd(), GOOGLE_APPLICATION_CREDENTIALS)

            if not os.path.exists(credentials_file_path):
                self.logger.warning(
                    f"GOOGLE_APPLICATION_CREDENTIALS file cannot be found in {credentials_file_path}"
                )
                self.gcp_enabled = False
