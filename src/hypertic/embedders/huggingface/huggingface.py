from dataclasses import dataclass, field
from os import getenv
from typing import Any

from hypertic.embedders.base import BaseEmbedder
from hypertic.utils.log import get_logger

logger = get_logger(__name__)

try:
    from huggingface_hub import AsyncInferenceClient, InferenceClient
except ImportError as err:
    raise ImportError("`huggingface-hub` not installed. Install with: pip install huggingface-hub") from err


@dataclass
class HuggingFaceEmbedder(BaseEmbedder):
    model: str = "intfloat/multilingual-e5-large"
    api_key: str | None = None
    client_params: dict[str, Any] | None = None

    client: Any | None = field(default=None, init=False)
    async_client: Any | None = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("HUGGINGFACE_API_KEY")

    def _get_client(self):
        if self.client is None:
            params: dict[str, Any] = {}
            if self.api_key is not None:
                params["api_key"] = self.api_key
            if self.client_params:
                params.update(self.client_params)
            self.client = InferenceClient(**params)
        return self.client

    def _get_async_client(self):
        if self.async_client is None:
            params: dict[str, Any] = {}
            if self.api_key is not None:
                params["api_key"] = self.api_key
            if self.client_params:
                params.update(self.client_params)
            self.async_client = AsyncInferenceClient(**params)
        return self.async_client

    async def initialize(self) -> bool:
        try:
            _ = self._get_async_client()
            return True
        except Exception as e:
            logger.error(f"Error initializing Hugging Face client: {e}", exc_info=True)
            return False

    async def embed(self, text: str) -> list[float]:
        try:
            client = self._get_async_client()
            response = await client.feature_extraction(text=text, model=self.model)
            if isinstance(response, list):
                return [float(x) for x in response]
            elif hasattr(response, "tolist"):
                embedding = response.tolist()
                if embedding and isinstance(embedding[0], list):
                    return [float(x) for x in embedding[0]]
                return [float(x) for x in embedding]
            else:
                embedding_list = list(response)
                if embedding_list and isinstance(embedding_list[0], list):
                    return [float(x) for x in embedding_list[0]]
                return [float(x) for x in embedding_list]
        except Exception as e:
            logger.error(f"Error creating embedding: {e}", exc_info=True)
            return []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._get_async_client()
            embeddings = []
            for text in texts:
                response = await client.feature_extraction(text=text, model=self.model)
                if isinstance(response, list):
                    embeddings.append(response)
                elif hasattr(response, "tolist"):
                    embeddings.append(response.tolist())
                else:
                    embeddings.append(list(response))
            return embeddings
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}", exc_info=True)
            return []

    def initialize_sync(self) -> bool:
        try:
            _ = self._get_client()
            return True
        except Exception as e:
            logger.error(f"Error initializing Hugging Face client: {e}", exc_info=True)
            return False

    def embed_sync(self, text: str) -> list[float]:
        try:
            client = self._get_client()
            response = client.feature_extraction(text=text, model=self.model)
            if isinstance(response, list):
                return [float(x) for x in response]
            elif hasattr(response, "tolist"):
                embedding = response.tolist()
                if embedding and isinstance(embedding[0], list):
                    return [float(x) for x in embedding[0]]
                return [float(x) for x in embedding]
            else:
                embedding_list = list(response)
                if embedding_list and isinstance(embedding_list[0], list):
                    return [float(x) for x in embedding_list[0]]
                return [float(x) for x in embedding_list]
        except Exception as e:
            logger.error(f"Error creating embedding: {e}", exc_info=True)
            return []

    def embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._get_client()
            embeddings = []
            for text in texts:
                response = client.feature_extraction(text=text, model=self.model)
                if isinstance(response, list):
                    embeddings.append(response)
                elif hasattr(response, "tolist"):
                    embeddings.append(response.tolist())
                else:
                    embeddings.append(list(response))
            return embeddings
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}", exc_info=True)
            return []
