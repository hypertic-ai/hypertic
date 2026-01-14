import asyncio
from dataclasses import dataclass, field
from os import getenv
from typing import Any

from hypertic.embedders.base import BaseEmbedder
from hypertic.utils.log import get_logger

logger = get_logger(__name__)

try:
    from mistralai import Mistral
    from mistralai.models.embeddingresponse import EmbeddingResponse
except ImportError as err:
    raise ImportError("`mistralai` not installed. Install with: pip install mistralai") from err


@dataclass
class MistralEmbedder(BaseEmbedder):
    model: str = "mistral-embed"
    api_key: str | None = None
    endpoint: str | None = None
    max_retries: int | None = None
    timeout: int | None = None
    request_params: dict[str, Any] | None = None
    client_params: dict[str, Any] | None = None

    client: Any | None = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("MISTRAL_API_KEY")

    def _get_client(self):
        if self.client is None:
            params: dict[str, Any] = {}
            if self.api_key is not None:
                params["api_key"] = self.api_key
            if self.endpoint is not None:
                params["endpoint"] = self.endpoint
            if self.max_retries is not None:
                params["max_retries"] = self.max_retries
            if self.timeout is not None:
                params["timeout"] = self.timeout

            if self.client_params:
                params.update(self.client_params)

            self.client = Mistral(**params)
        return self.client

    def _get_request_params(self, inputs):
        params: dict[str, Any] = {
            "model": self.model,
            "inputs": inputs if isinstance(inputs, list) else [inputs],
        }

        if self.request_params:
            params.update(self.request_params)

        return params

    def _response(self, text: str) -> EmbeddingResponse:
        request_params = self._get_request_params([text])
        response = self._get_client().embeddings.create(**request_params)
        if response is None:
            raise ValueError("Failed to get embedding response")
        if not isinstance(response, EmbeddingResponse):
            raise ValueError("Unexpected response type from embeddings.create")
        return response

    async def initialize(self) -> bool:
        try:
            _ = self._get_client()
            return True
        except Exception as e:
            logger.error(f"Error initializing Mistral client: {e}", exc_info=True)
            return False

    async def embed(self, text: str) -> list[float]:
        try:
            client = self._get_client()
            request_params = self._get_request_params([text])

            if hasattr(client.embeddings, "create_async"):
                response: EmbeddingResponse = await client.embeddings.create_async(**request_params)
            else:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, lambda: client.embeddings.create(**request_params))
                if not isinstance(response, EmbeddingResponse):
                    raise ValueError("Unexpected response type from embeddings.create")

            if response.data and len(response.data) > 0 and response.data[0].embedding:
                return list(response.data[0].embedding)
            return []
        except Exception as e:
            logger.error(f"Error creating embedding: {e}", exc_info=True)
            return []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._get_client()
            request_params = self._get_request_params(texts)

            if hasattr(client.embeddings, "create_async"):
                response: EmbeddingResponse = await client.embeddings.create_async(**request_params)
            else:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, lambda: client.embeddings.create(**request_params))
                if not isinstance(response, EmbeddingResponse):
                    raise ValueError("Unexpected response type from embeddings.create")

            if response.data:
                return [data.embedding for data in response.data if data.embedding]
            return []
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}", exc_info=True)
            return []

    def initialize_sync(self) -> bool:
        try:
            _ = self._get_client()
            return True
        except Exception as e:
            logger.error(f"Error initializing Mistral client: {e}", exc_info=True)
            return False

    def embed_sync(self, text: str) -> list[float]:
        try:
            response: EmbeddingResponse = self._response(text=text)
            if response.data and len(response.data) > 0 and response.data[0].embedding:
                return list(response.data[0].embedding)
            return []
        except Exception as e:
            logger.error(f"Error creating embedding: {e}", exc_info=True)
            return []

    def embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._get_client()
            request_params = self._get_request_params(texts)
            response: EmbeddingResponse = client.embeddings.create(**request_params)

            if response.data:
                return [data.embedding for data in response.data if data.embedding]
            return []
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}", exc_info=True)
            return []
