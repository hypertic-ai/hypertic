import pytest

from hypertic.embedders.base import BaseEmbedder


class TestBaseEmbedder:
    def test_base_embedder_is_abstract(self):
        with pytest.raises(TypeError):
            BaseEmbedder()  # type: ignore[abstract]

    def test_base_embedder_has_required_methods(self):
        assert hasattr(BaseEmbedder, "initialize")
        assert hasattr(BaseEmbedder, "embed")
        assert hasattr(BaseEmbedder, "embed_batch")
        assert hasattr(BaseEmbedder, "initialize_sync")
        assert hasattr(BaseEmbedder, "embed_sync")
        assert hasattr(BaseEmbedder, "embed_batch_sync")

    @pytest.mark.asyncio
    async def test_base_embedder_embed_batch(self):
        """Test embed_batch method implementation."""

        class ConcreteEmbedder(BaseEmbedder):
            async def initialize(self) -> bool:
                return True

            async def embed(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            def initialize_sync(self) -> bool:
                return True

            def embed_sync(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

        embedder = ConcreteEmbedder()
        texts = ["text1", "text2", "text3"]
        results = await embedder.embed_batch(texts)
        assert len(results) == 3
        assert all(len(emb) == 3 for emb in results)
        assert results[0] == [0.1, 0.2, 0.3]

    def test_base_embedder_embed_batch_sync(self):
        """Test embed_batch_sync method implementation."""

        class ConcreteEmbedder(BaseEmbedder):
            async def initialize(self) -> bool:
                return True

            async def embed(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            def initialize_sync(self) -> bool:
                return True

            def embed_sync(self, text: str) -> list[float]:
                return [0.4, 0.5, 0.6]

        embedder = ConcreteEmbedder()
        texts = ["text1", "text2"]
        results = embedder.embed_batch_sync(texts)
        assert len(results) == 2
        assert all(len(emb) == 3 for emb in results)
        assert results[0] == [0.4, 0.5, 0.6]

    def test_base_embedder_embed_batch_sync_empty_list(self):
        """Test embed_batch_sync with empty list."""

        class ConcreteEmbedder(BaseEmbedder):
            async def initialize(self) -> bool:
                return True

            async def embed(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            def initialize_sync(self) -> bool:
                return True

            def embed_sync(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

        embedder = ConcreteEmbedder()
        results = embedder.embed_batch_sync([])
        assert results == []

    @pytest.mark.asyncio
    async def test_base_embedder_embed_batch_empty_list(self):
        """Test embed_batch with empty list."""

        class ConcreteEmbedder(BaseEmbedder):
            async def initialize(self) -> bool:
                return True

            async def embed(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            def initialize_sync(self) -> bool:
                return True

            def embed_sync(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

        embedder = ConcreteEmbedder()
        results = await embedder.embed_batch([])
        assert results == []
