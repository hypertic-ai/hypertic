from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.embedders.cohere.cohere import CohereEmbedder


@pytest.mark.unit
class TestCohereEmbedder:
    @pytest.fixture
    def embedder(self, mock_api_key):
        with patch("hypertic.embedders.cohere.cohere.CohereClient"), patch("hypertic.embedders.cohere.cohere.CohereAsyncClient"):
            return CohereEmbedder(api_key=mock_api_key, model="embed-english-v3.0")

    def test_cohere_embedder_creation(self, embedder, mock_api_key):
        assert embedder.api_key == mock_api_key
        assert embedder.model == "embed-english-v3.0"
        assert embedder.input_type == "search_document"

    @patch("hypertic.embedders.cohere.cohere.getenv")
    def test_cohere_embedder_with_env_api_key(self, mock_getenv, mock_api_key):
        mock_getenv.return_value = "env_key"
        with patch("hypertic.embedders.cohere.cohere.CohereClient"), patch("hypertic.embedders.cohere.cohere.CohereAsyncClient"):
            embedder = CohereEmbedder(model="embed-english-v3.0")
            assert embedder.api_key == "env_key"

    def test_get_client(self, embedder):
        """Test _get_client creates CohereClient."""
        mock_client_class = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("hypertic.embedders.cohere.cohere.CohereClient", mock_client_class):
            embedder.client = None
            client = embedder._get_client()
            assert client == mock_client

    def test_get_async_client(self, embedder):
        """Test _get_async_client creates CohereAsyncClient."""
        mock_client_class = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("hypertic.embedders.cohere.cohere.CohereAsyncClient", mock_client_class):
            embedder.async_client = None
            client = embedder._get_async_client()
            assert client == mock_client

    def test_get_request_params(self, embedder):
        """Test _get_request_params."""
        params = embedder._get_request_params("test text")
        assert params["texts"] == ["test text"]
        assert params["model"] == "embed-english-v3.0"
        assert params["input_type"] == "search_document"

    def test_get_request_params_with_truncate(self, embedder):
        """Test _get_request_params with truncate."""
        embedder.truncate = "END"
        params = embedder._get_request_params("test text")
        assert params["truncate"] == "END"

    def test_get_request_params_list(self, embedder):
        """Test _get_request_params with list input."""
        params = embedder._get_request_params(["text1", "text2"])
        assert params["texts"] == ["text1", "text2"]

    @pytest.mark.asyncio
    async def test_initialize(self, embedder):
        """Test initialize method."""
        mock_client = MagicMock()
        with patch.object(embedder, "_get_async_client", return_value=mock_client):
            result = await embedder.initialize()
            assert result is True

    @pytest.mark.asyncio
    async def test_embed(self, embedder):
        """Test embed method."""
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.float_ = [[0.1, 0.2, 0.3]]
        mock_response = MagicMock()
        mock_response.embeddings = mock_embeddings
        mock_client.embed = AsyncMock(return_value=mock_response)

        with patch.object(embedder, "_get_async_client", return_value=mock_client):
            result = await embedder.embed("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_empty_embeddings(self, embedder):
        """Test embed with empty embeddings."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = None
        mock_client.embed = AsyncMock(return_value=mock_response)

        with patch.object(embedder, "_get_async_client", return_value=mock_client):
            result = await embedder.embed("test text")
            assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder):
        """Test embed_batch method."""
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.float_ = [[0.1, 0.2], [0.3, 0.4]]
        mock_response = MagicMock()
        mock_response.embeddings = mock_embeddings
        mock_client.embed = AsyncMock(return_value=mock_response)

        with patch.object(embedder, "_get_async_client", return_value=mock_client):
            result = await embedder.embed_batch(["text1", "text2"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_initialize_sync(self, embedder):
        """Test initialize_sync method."""
        mock_client = MagicMock()
        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.initialize_sync()
            assert result is True

    def test_embed_sync(self, embedder):
        """Test embed_sync method."""
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.float_ = [[0.1, 0.2, 0.3]]
        mock_response = MagicMock()
        mock_response.embeddings = mock_embeddings
        mock_client.embed.return_value = mock_response

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed_sync("test text")
            assert result == [0.1, 0.2, 0.3]

    def test_embed_batch_sync(self, embedder):
        """Test embed_batch_sync method."""
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_embeddings.float_ = [[0.1, 0.2], [0.3, 0.4]]
        mock_response = MagicMock()
        mock_response.embeddings = mock_embeddings
        mock_client.embed.return_value = mock_response

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed_batch_sync(["text1", "text2"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]
