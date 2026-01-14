from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.embedders.openai.openai import OpenAIEmbedder


@pytest.mark.unit
class TestOpenAIEmbedder:
    @pytest.fixture
    def embedder(self, mock_api_key):
        with patch("hypertic.embedders.openai.openai.OpenAI"), patch("hypertic.embedders.openai.openai.AsyncOpenAI"):
            return OpenAIEmbedder(api_key=mock_api_key, model="text-embedding-3-small")

    def test_openai_embedder_creation(self, embedder, mock_api_key):
        assert embedder.api_key == mock_api_key
        assert embedder.model == "text-embedding-3-small"
        assert embedder.dimensions == 1536  # Default for text-embedding-3-small

    def test_openai_embedder_dimensions_large(self, mock_api_key):
        """Test dimensions for text-embedding-3-large."""
        with patch("hypertic.embedders.openai.openai.OpenAI"), patch("hypertic.embedders.openai.openai.AsyncOpenAI"):
            embedder = OpenAIEmbedder(api_key=mock_api_key, model="text-embedding-3-large")
            assert embedder.dimensions == 3072

    def test_openai_embedder_custom_dimensions(self, mock_api_key):
        """Test custom dimensions."""
        with patch("hypertic.embedders.openai.openai.OpenAI"), patch("hypertic.embedders.openai.openai.AsyncOpenAI"):
            embedder = OpenAIEmbedder(api_key=mock_api_key, model="text-embedding-3-small", dimensions=512)
            assert embedder.dimensions == 512

    @patch("hypertic.embedders.openai.openai.getenv")
    def test_openai_embedder_with_env_api_key(self, mock_getenv, mock_api_key):
        mock_getenv.return_value = "env_key"
        with patch("hypertic.embedders.openai.openai.OpenAI"), patch("hypertic.embedders.openai.openai.AsyncOpenAI"):
            embedder = OpenAIEmbedder(model="text-embedding-3-small")
            assert embedder.api_key == "env_key"

    def test_get_client(self, embedder):
        """Test _get_client creates OpenAI client."""
        mock_client_class = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("hypertic.embedders.openai.openai.OpenAI", mock_client_class):
            embedder.client = None
            client = embedder._get_client()
            assert client == mock_client

    def test_get_async_client(self, embedder):
        """Test _get_async_client creates AsyncOpenAI client."""
        mock_client_class = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("hypertic.embedders.openai.openai.AsyncOpenAI", mock_client_class):
            embedder.async_client = None
            client = embedder._get_async_client()
            assert client == mock_client

    def test_get_request_params(self, embedder):
        """Test _get_request_params."""
        params = embedder._get_request_params("test text")
        assert params["input"] == "test text"
        assert params["model"] == "text-embedding-3-small"
        assert params["dimensions"] == 1536

    def test_get_request_params_with_dimensions(self, embedder):
        """Test _get_request_params with custom dimensions."""
        embedder.dimensions = 512
        params = embedder._get_request_params("test text")
        assert params["dimensions"] == 512

    def test_get_request_params_non_embedding3_model(self, mock_api_key):
        """Test _get_request_params for non-embedding-3 model."""
        with patch("hypertic.embedders.openai.openai.OpenAI"), patch("hypertic.embedders.openai.openai.AsyncOpenAI"):
            embedder = OpenAIEmbedder(api_key=mock_api_key, model="text-embedding-ada-002", dimensions=1536)
            params = embedder._get_request_params("test text")
            assert "dimensions" not in params

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
        mock_data = MagicMock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_embeddings.create = AsyncMock(return_value=mock_response)
        mock_client.embeddings = mock_embeddings

        with patch.object(embedder, "_get_async_client", return_value=mock_client):
            result = await embedder.embed("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_empty_embedding(self, embedder):
        """Test embed with empty embedding."""
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = None
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_embeddings.create = AsyncMock(return_value=mock_response)
        mock_client.embeddings = mock_embeddings

        with patch.object(embedder, "_get_async_client", return_value=mock_client):
            result = await embedder.embed("test text")
            assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder):
        """Test embed_batch method."""
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_data1 = MagicMock()
        mock_data1.embedding = [0.1, 0.2]
        mock_data2 = MagicMock()
        mock_data2.embedding = [0.3, 0.4]
        mock_response = MagicMock()
        mock_response.data = [mock_data1, mock_data2]
        mock_embeddings.create = AsyncMock(return_value=mock_response)
        mock_client.embeddings = mock_embeddings

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
        mock_data = MagicMock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_embeddings.create.return_value = mock_response
        mock_client.embeddings = mock_embeddings

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed_sync("test text")
            assert result == [0.1, 0.2, 0.3]

    def test_embed_batch_sync(self, embedder):
        """Test embed_batch_sync method."""
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_data1 = MagicMock()
        mock_data1.embedding = [0.1, 0.2]
        mock_data2 = MagicMock()
        mock_data2.embedding = [0.3, 0.4]
        mock_response = MagicMock()
        mock_response.data = [mock_data1, mock_data2]
        mock_embeddings.create.return_value = mock_response
        mock_client.embeddings = mock_embeddings

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed_batch_sync(["text1", "text2"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]
