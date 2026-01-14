from dataclasses import dataclass, field
from os import getenv
from typing import Any

from hypertic.embedders.base import BaseEmbedder
from hypertic.utils.log import get_logger

logger = get_logger(__name__)

try:
    from cohere import AsyncClientV2 as CohereAsyncClient, ClientV2 as CohereClient
except ImportError as err:
    raise ImportError("`cohere` not installed. Install with: pip install cohere") from err


@dataclass
class CohereEmbedder(BaseEmbedder):
    api_key: str | None = None
    model: str = "embed-english-v3.0"
    input_type: str = "search_document"
    truncate: str | None = None

    client: Any | None = field(default=None, init=False)
    async_client: Any | None = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("COHERE_API_KEY")

    async def initialize(self) -> bool:
        try:
            _ = self._get_async_client()
            return True
        except Exception as e:
            logger.error(f"Error initializing Cohere client: {e}", exc_info=True)
            return False

    def _get_client(self):
        if self.client is None:
            if self.api_key is not None:
                self.client = CohereClient(api_key=self.api_key)
            else:
                self.client = CohereClient()
        return self.client

    def _get_async_client(self):
        if self.async_client is None:
            if self.api_key is not None:
                self.async_client = CohereAsyncClient(api_key=self.api_key)
            else:
                self.async_client = CohereAsyncClient()
        return self.async_client

    def _get_request_params(self, input_data):
        params = {
            "texts": input_data if isinstance(input_data, list) else [input_data],
            "model": self.model,
            "input_type": self.input_type,
        }

        if self.truncate:
            params["truncate"] = self.truncate

        return params

    async def embed(self, text: str) -> list[float]:
        try:
            client = self._get_async_client()
            response = await client.embed(**self._get_request_params(text))
            if response.embeddings and response.embeddings.float_:
                embedding = response.embeddings.float_[0]
                return list(embedding) if embedding else []
            return []
        except Exception as e:
            logger.error(f"Error creating embedding: {e}", exc_info=True)
            return []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._get_async_client()
            response = await client.embed(**self._get_request_params(texts))
            if response.embeddings and response.embeddings.float_:
                embeddings = response.embeddings.float_
                return [list(emb) for emb in embeddings] if embeddings else []
            return []
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}", exc_info=True)
            return []

    def initialize_sync(self) -> bool:
        try:
            _ = self._get_client()
            return True
        except Exception as e:
            logger.error(f"Error initializing Cohere client: {e}", exc_info=True)
            return False

    def embed_sync(self, text: str) -> list[float]:
        try:
            client = self._get_client()
            response = client.embed(**self._get_request_params(text))
            if response.embeddings and response.embeddings.float_:
                embedding = response.embeddings.float_[0]
                return list(embedding) if embedding else []
            return []
        except Exception as e:
            logger.error(f"Error creating embedding: {e}", exc_info=True)
            return []

    def embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._get_client()
            response = client.embed(**self._get_request_params(texts))
            if response.embeddings and response.embeddings.float_:
                embeddings = response.embeddings.float_
                return [list(emb) for emb in embeddings] if embeddings else []
            return []
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}", exc_info=True)
            return []
