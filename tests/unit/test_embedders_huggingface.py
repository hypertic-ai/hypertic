from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.embedders.huggingface.huggingface import HuggingFaceEmbedder


@pytest.mark.unit
class TestHuggingFaceEmbedder:
    @pytest.fixture
    def embedder(self, mock_api_key):
        with (
            patch("hypertic.embedders.huggingface.huggingface.InferenceClient"),
            patch("hypertic.embedders.huggingface.huggingface.AsyncInferenceClient"),
        ):
            return HuggingFaceEmbedder(api_key=mock_api_key, model="test-model")

    def test_huggingface_embedder_creation(self, embedder, mock_api_key):
        assert embedder.api_key == mock_api_key
        assert embedder.model == "test-model"

    @patch("hypertic.embedders.huggingface.huggingface.getenv")
    def test_huggingface_embedder_with_env_api_key(self, mock_getenv, mock_api_key):
        mock_getenv.return_value = "env_key"
        with (
            patch("hypertic.embedders.huggingface.huggingface.InferenceClient"),
            patch("hypertic.embedders.huggingface.huggingface.AsyncInferenceClient"),
        ):
            embedder = HuggingFaceEmbedder(model="test-model")
            assert embedder.api_key == "env_key"

    def test_get_client(self, embedder):
        """Test _get_client creates InferenceClient."""
        mock_client_class = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("hypertic.embedders.huggingface.huggingface.InferenceClient", mock_client_class):
            embedder.client = None
            client = embedder._get_client()
            assert client == mock_client

    def test_get_async_client(self, embedder):
        """Test _get_async_client creates AsyncInferenceClient."""
        mock_client_class = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("hypertic.embedders.huggingface.huggingface.AsyncInferenceClient", mock_client_class):
            embedder.async_client = None
            client = embedder._get_async_client()
            assert client == mock_client

    @pytest.mark.asyncio
    async def test_initialize(self, embedder):
        """Test initialize method."""
        mock_client = MagicMock()
        with patch.object(embedder, "_get_async_client", return_value=mock_client):
            result = await embedder.initialize()
            assert result is True

    @pytest.mark.asyncio
    async def test_embed_list_response(self, embedder):
        """Test embed with list response."""
        mock_client = MagicMock()
        mock_client.feature_extraction = AsyncMock(return_value=[0.1, 0.2, 0.3])

        with patch.object(embedder, "_get_async_client", return_value=mock_client):
            result = await embedder.embed("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_tolist_response(self, embedder):
        """Test embed with tolist response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.tolist.return_value = [0.1, 0.2, 0.3]
        mock_client.feature_extraction = AsyncMock(return_value=mock_response)

        with patch.object(embedder, "_get_async_client", return_value=mock_client):
            result = await embedder.embed("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_nested_tolist_response(self, embedder):
        """Test embed with nested tolist response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.tolist.return_value = [[0.1, 0.2, 0.3]]
        mock_client.feature_extraction = AsyncMock(return_value=mock_response)

        with patch.object(embedder, "_get_async_client", return_value=mock_client):
            result = await embedder.embed("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_list_response_fallback(self, embedder):
        """Test embed with list fallback."""
        mock_client = MagicMock()
        # Return a list-like object that doesn't have tolist
        mock_response = [0.1, 0.2, 0.3]
        mock_client.feature_extraction = AsyncMock(return_value=mock_response)

        with patch.object(embedder, "_get_async_client", return_value=mock_client):
            result = await embedder.embed("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder):
        """Test embed_batch method."""
        mock_client = MagicMock()
        mock_client.feature_extraction = AsyncMock(side_effect=[[0.1, 0.2], [0.3, 0.4]])

        with patch.object(embedder, "_get_async_client", return_value=mock_client):
            result = await embedder.embed_batch(["text1", "text2"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_initialize_sync(self, embedder):
        """Test initialize_sync method."""
        mock_client = MagicMock()
        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.initialize_sync()
            assert result is True

    def test_embed_sync_list_response(self, embedder):
        """Test embed_sync with list response."""
        mock_client = MagicMock()
        mock_client.feature_extraction.return_value = [0.1, 0.2, 0.3]

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed_sync("test text")
            assert result == [0.1, 0.2, 0.3]

    def test_embed_sync_tolist_response(self, embedder):
        """Test embed_sync with tolist response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.tolist.return_value = [0.1, 0.2, 0.3]
        mock_client.feature_extraction.return_value = mock_response

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed_sync("test text")
            assert result == [0.1, 0.2, 0.3]

    def test_embed_batch_sync(self, embedder):
        """Test embed_batch_sync method."""
        mock_client = MagicMock()
        mock_client.feature_extraction.side_effect = [[0.1, 0.2], [0.3, 0.4]]

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed_batch_sync(["text1", "text2"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]
