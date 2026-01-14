import asyncio
from dataclasses import dataclass, field
from typing import Any

from hypertic.embedders.base import BaseEmbedder
from hypertic.utils.log import get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError as err:
    raise ImportError("`sentence-transformers` not installed. Install with: pip install sentence-transformers") from err


@dataclass
class SentenceTransformerEmbedder(BaseEmbedder):
    model: str = "all-MiniLM-L6-v2"
    device: str | None = None
    model_kwargs: dict[str, Any] | None = None

    _model: SentenceTransformer | None = field(default=None, init=False)

    def _get_device(self) -> str:
        if self.device:
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            device = self._get_device()
            kwargs = self.model_kwargs or {}
            self._model = SentenceTransformer(self.model, device=device, **kwargs)
            logger.info(f"Loaded SentenceTransformer model '{self.model}' on device '{device}'")
        return self._model

    async def initialize(self) -> bool:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._get_model)
            return True
        except Exception as e:
            logger.error(f"Error initializing SentenceTransformer model: {e}", exc_info=True)
            return False

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embed_sync_internal, text)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self._embed_batch_sync_internal, texts)
        return embeddings

    def initialize_sync(self) -> bool:
        try:
            _ = self._get_model()
            return True
        except Exception as e:
            logger.error(f"Error initializing SentenceTransformer model: {e}", exc_info=True)
            return False

    def embed_sync(self, text: str) -> list[float]:
        return self._embed_sync_internal(text)

    def embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch_sync_internal(texts)

    def _embed_sync_internal(self, text: str) -> list[float]:
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        result: list[float] = embedding.tolist()
        return result

    def _embed_batch_sync_internal(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        result: list[list[float]] = embeddings.tolist()
        return result

    def get_embedding_dimension(self) -> int:
        model = self._get_model()
        dummy_embedding = model.encode("test", convert_to_numpy=True, show_progress_bar=False)
        return len(dummy_embedding)
