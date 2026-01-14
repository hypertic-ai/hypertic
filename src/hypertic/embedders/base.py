from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Base class for all embedders"""

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the embedder"""
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Convert text to vector"""
        pass

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple texts to vectors"""
        return [await self.embed(text) for text in texts]

    # SYNC METHODS
    @abstractmethod
    def initialize_sync(self) -> bool:
        """Initialize the embedder synchronously"""
        pass

    @abstractmethod
    def embed_sync(self, text: str) -> list[float]:
        """Convert text to vector synchronously"""
        pass

    def embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple texts to vectors synchronously"""
        return [self.embed_sync(text) for text in texts]
