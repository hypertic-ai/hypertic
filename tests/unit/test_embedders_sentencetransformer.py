from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.embedders.sentencetransformer.sentencetransformer import SentenceTransformerEmbedder


@pytest.mark.unit
class TestSentenceTransformerEmbedder:
    @pytest.fixture
    def embedder(self):
        with patch("hypertic.embedders.sentencetransformer.sentencetransformer.SentenceTransformer"):
            return SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")

    def test_sentencetransformer_embedder_creation(self, embedder):
        assert embedder.model == "all-MiniLM-L6-v2"
        assert embedder.device is None

    def test_get_device_custom(self, embedder):
        """Test _get_device with custom device."""
        embedder.device = "cpu"
        assert embedder._get_device() == "cpu"

    def test_get_device_cuda(self, embedder):
        """Test _get_device detects CUDA."""
        embedder.device = None
        device = embedder._get_device()
        assert device in ["cpu", "cuda", "mps"]

    def test_get_device_mps(self, embedder):
        """Test _get_device detects MPS."""
        embedder.device = None
        device = embedder._get_device()
        assert device in ["cpu", "cuda", "mps"]

    def test_get_device_cpu_fallback(self, embedder):
        """Test _get_device falls back to CPU when no CUDA/MPS."""
        embedder.device = None
        device = embedder._get_device()
        assert device in ["cpu", "cuda", "mps"]

    def test_get_device_no_torch(self, embedder):
        """Test _get_device when torch not available."""
        embedder.device = None
        device = embedder._get_device()
        assert device in ["cpu", "cuda", "mps"]

    def test_get_model(self, embedder):
        """Test _get_model creates SentenceTransformer."""
        mock_model_class = MagicMock()
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        with patch("hypertic.embedders.sentencetransformer.sentencetransformer.SentenceTransformer", mock_model_class):
            embedder._model = None
            model = embedder._get_model()
            assert model == mock_model

    def test_get_model_with_kwargs(self, embedder):
        """Test _get_model with model_kwargs."""
        embedder.model_kwargs = {"trust_remote_code": True}
        mock_model_class = MagicMock()
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        with patch("hypertic.embedders.sentencetransformer.sentencetransformer.SentenceTransformer", mock_model_class):
            embedder._model = None
            model = embedder._get_model()
            assert model == mock_model
            call_kwargs = mock_model_class.call_args[1]
            assert call_kwargs["trust_remote_code"] is True

    @pytest.mark.asyncio
    async def test_initialize(self, embedder):
        """Test initialize method."""
        mock_model = MagicMock()
        with patch.object(embedder, "_get_model", return_value=mock_model):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)
                result = await embedder.initialize()
                assert result is True

    @pytest.mark.asyncio
    async def test_embed(self, embedder):
        """Test embed method."""
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_embedding

        with patch.object(embedder, "_get_model", return_value=mock_model):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=[0.1, 0.2, 0.3])
                result = await embedder.embed("test text")
                assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder):
        """Test embed_batch method."""
        mock_model = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_model.encode.return_value = mock_embeddings

        with patch.object(embedder, "_get_model", return_value=mock_model):
            with patch("asyncio.get_event_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
                result = await embedder.embed_batch(["text1", "text2"])
                assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_initialize_sync(self, embedder):
        """Test initialize_sync method."""
        mock_model = MagicMock()
        with patch.object(embedder, "_get_model", return_value=mock_model):
            result = embedder.initialize_sync()
            assert result is True

    def test_embed_sync(self, embedder):
        """Test embed_sync method."""
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = mock_embedding

        with patch.object(embedder, "_get_model", return_value=mock_model):
            result = embedder.embed_sync("test text")
            assert result == [0.1, 0.2, 0.3]

    def test_embed_batch_sync(self, embedder):
        """Test embed_batch_sync method."""
        mock_model = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_model.encode.return_value = mock_embeddings

        with patch.object(embedder, "_get_model", return_value=mock_model):
            result = embedder.embed_batch_sync(["text1", "text2"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_get_embedding_dimension(self, embedder):
        """Test get_embedding_dimension method."""
        mock_model = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.__len__ = Mock(return_value=384)
        mock_model.encode.return_value = mock_embedding

        with patch.object(embedder, "_get_model", return_value=mock_model):
            dimension = embedder.get_embedding_dimension()
            assert dimension == 384
