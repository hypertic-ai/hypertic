from dataclasses import dataclass, field
from os import getenv
from typing import Any

from hypertic.embedders.base import BaseEmbedder
from hypertic.utils.log import get_logger

logger = get_logger(__name__)

try:
    from google.genai import Client as GeminiClient
except ImportError as err:
    raise ImportError("`google-genai` not installed. Install with: pip install google-genai") from err


@dataclass
class GoogleEmbedder(BaseEmbedder):
    api_key: str | None = None
    model: str = "text-embedding-004"
    task_type: str | None = None
    title: str | None = None

    client: Any | None = field(default=None, init=False)
    aio_client: Any | None = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("GEMINI_API_KEY")

    async def initialize(self) -> bool:
        try:
            _ = self._get_aio_client()
            return True
        except Exception as e:
            logger.error(f"Error initializing Google client: {e}", exc_info=True)
            return False

    def _get_client(self):
        if self.client is None:
            if self.api_key is not None:
                self.client = GeminiClient(api_key=self.api_key)
            else:
                self.client = GeminiClient()
        return self.client

    def _get_aio_client(self):
        if self.aio_client is None:
            if self.api_key is not None:
                client = GeminiClient(api_key=self.api_key)
            else:
                client = GeminiClient()
            self.aio_client = client.aio
        return self.aio_client

    def _get_request_config(self):
        if self.task_type or self.title:
            from google.genai.types import EmbedContentConfig

            config = EmbedContentConfig()
            if self.task_type:
                config.task_type = self.task_type
            if self.title:
                config.title = self.title
            return config
        return None

    async def embed(self, text: str) -> list[float]:
        try:
            aio_client = self._get_aio_client()
            config = self._get_request_config()
            response = await aio_client.models.embed_content(model=self.model, contents=text, config=config)
            if response.embeddings and len(response.embeddings) > 0:
                return response.embeddings[0].values or []
            return []
        except Exception as e:
            logger.error(f"Error creating embedding: {e}", exc_info=True)
            return []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            aio_client = self._get_aio_client()
            config = self._get_request_config()
            response = await aio_client.models.embed_content(model=self.model, contents=texts, config=config)
            if response.embeddings:
                return [emb.values or [] for emb in response.embeddings]
            return []
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}", exc_info=True)
            return []

    def initialize_sync(self) -> bool:
        try:
            _ = self._get_client()
            return True
        except Exception as e:
            logger.error(f"Error initializing Google client: {e}", exc_info=True)
            return False

    def embed_sync(self, text: str) -> list[float]:
        try:
            client = self._get_client()
            config = self._get_request_config()
            response = client.models.embed_content(model=self.model, contents=text, config=config)
            if response.embeddings and len(response.embeddings) > 0:
                return response.embeddings[0].values or []
            return []
        except Exception as e:
            logger.error(f"Error creating embedding: {e}", exc_info=True)
            return []

    def embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        try:
            client = self._get_client()
            config = self._get_request_config()
            response = client.models.embed_content(model=self.model, contents=texts, config=config)
            if response.embeddings:
                return [emb.values or [] for emb in response.embeddings]
            return []
        except Exception as e:
            logger.error(f"Error creating batch embeddings: {e}", exc_info=True)
            return []
