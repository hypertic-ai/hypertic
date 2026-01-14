from dataclasses import dataclass, field
from os import getenv
from typing import Any

from hypertic.embedders.base import BaseEmbedder
from hypertic.utils.log import get_logger

logger = get_logger(__name__)

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError as err:
    raise ImportError("`openai` not installed. Install with: pip install openai") from err


@dataclass
class OpenAIEmbedder(BaseEmbedder):
    api_key: str | None = None
    model: str = "text-embedding-3-small"
    dimensions: int | None = None
    organization: str | None = None
    base_url: str | None = None

    client: Any | None = field(default=None, init=False)
    async_client: Any | None = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("OPENAI_API_KEY")

        if self.dimensions is None:
            if self.model == "text-embedding-3-large":
                self.dimensions = 3072
            elif self.model == "text-embedding-3-small":
                self.dimensions = 1536
            else:
                self.dimensions = 1536

    async def initialize(self) -> bool:
        try:
            _ = self._get_async_client()
            return True
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}", exc_info=True)
            return False

    def _get_client(self):
        if self.client is None:
            client_params: dict[str, Any] = {}
            if self.api_key is not None:
                client_params["api_key"] = self.api_key
            if self.organization is not None:
                client_params["organization"] = self.organization
            if self.base_url is not None:
                client_params["base_url"] = self.base_url
            self.client = OpenAI(**client_params)
        return self.client

    def _get_async_client(self):
        if self.async_client is None:
            client_params: dict[str, Any] = {}
            if self.api_key is not None:
                client_params["api_key"] = self.api_key
            if self.organization is not None:
                client_params["organization"] = self.organization
            if self.base_url is not None:
                client_params["base_url"] = self.base_url
            self.async_client = AsyncOpenAI(**client_params)
        return self.async_client

    def _get_request_params(self, input_data):
        params = {
            "input": input_data,
            "model": self.model,
        }

        if self.model.startswith("text-embedding-3") and self.dimensions:
            params["dimensions"] = self.dimensions

        return params

    async def embed(self, text: str) -> list[float]:
        try:
            client = self._get_async_client()
            response = await client.embeddings.create(**self._get_request_params(text))
            embedding = response.data[0].embedding
            return list(embedding) if embedding else []
        except Exception as e:
            logger.error(f"Error creating embedding: {e}", exc_info=True)
            return []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._get_async_client()
            response = await client.embeddings.create(**self._get_request_params(texts))
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}", exc_info=True)
            return []

    def initialize_sync(self) -> bool:
        try:
            _ = self._get_client()
            return True
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}", exc_info=True)
            return False

    def embed_sync(self, text: str) -> list[float]:
        try:
            client = self._get_client()
            response = client.embeddings.create(**self._get_request_params(text))
            embedding = response.data[0].embedding
            return list(embedding) if embedding else []
        except Exception as e:
            logger.error(f"Error creating embedding: {e}", exc_info=True)
            return []

    def embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._get_client()
            response = client.embeddings.create(**self._get_request_params(texts))
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}", exc_info=True)
            return []
