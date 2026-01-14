from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.embedders.google.google import GoogleEmbedder


@pytest.mark.unit
class TestGoogleEmbedder:
    @pytest.fixture
    def embedder(self, mock_api_key):
        with patch("hypertic.embedders.google.google.GeminiClient"):
            return GoogleEmbedder(api_key=mock_api_key, model="text-embedding-004")

    def test_google_embedder_creation(self, embedder, mock_api_key):
        assert embedder.api_key == mock_api_key
        assert embedder.model == "text-embedding-004"

    @patch("hypertic.embedders.google.google.getenv")
    def test_google_embedder_with_env_api_key(self, mock_getenv, mock_api_key):
        mock_getenv.return_value = "env_key"
        with patch("hypertic.embedders.google.google.GeminiClient"):
            embedder = GoogleEmbedder(model="text-embedding-004")
            assert embedder.api_key == "env_key"

    def test_get_client(self, embedder):
        """Test _get_client creates GeminiClient."""
        mock_client_class = MagicMock()
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        with patch("hypertic.embedders.google.google.GeminiClient", mock_client_class):
            embedder.client = None
            client = embedder._get_client()
            assert client == mock_client

    def test_get_aio_client(self, embedder):
        """Test _get_aio_client creates async client."""
        mock_client_class = MagicMock()
        mock_client = MagicMock()
        mock_aio = MagicMock()
        mock_client.aio = mock_aio
        mock_client_class.return_value = mock_client

        with patch("hypertic.embedders.google.google.GeminiClient", mock_client_class):
            embedder.aio_client = None
            aio_client = embedder._get_aio_client()
            assert aio_client == mock_aio

    def test_get_request_config_none(self, embedder):
        """Test _get_request_config returns None when no config."""
        config = embedder._get_request_config()
        assert config is None

    def test_get_request_config_with_task_type(self, embedder):
        """Test _get_request_config with task_type."""
        embedder.task_type = "RETRIEVAL_DOCUMENT"
        config = embedder._get_request_config()
        assert config is not None
        assert config.task_type == "RETRIEVAL_DOCUMENT"

    def test_get_request_config_with_title(self, embedder):
        """Test _get_request_config with title."""
        embedder.title = "Test Title"
        config = embedder._get_request_config()
        assert config is not None
        assert config.title == "Test Title"

    @pytest.mark.asyncio
    async def test_initialize(self, embedder):
        """Test initialize method."""
        mock_aio_client = MagicMock()
        with patch.object(embedder, "_get_aio_client", return_value=mock_aio_client):
            result = await embedder.initialize()
            assert result is True

    @pytest.mark.asyncio
    async def test_embed(self, embedder):
        """Test embed method."""
        mock_aio_client = MagicMock()
        mock_models = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        mock_models.embed_content = AsyncMock(return_value=mock_response)
        mock_aio_client.models = mock_models

        with patch.object(embedder, "_get_aio_client", return_value=mock_aio_client):
            result = await embedder.embed("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_empty_embeddings(self, embedder):
        """Test embed with empty embeddings."""
        mock_aio_client = MagicMock()
        mock_models = MagicMock()
        mock_response = MagicMock()
        mock_response.embeddings = []
        mock_models.embed_content = AsyncMock(return_value=mock_response)
        mock_aio_client.models = mock_models

        with patch.object(embedder, "_get_aio_client", return_value=mock_aio_client):
            result = await embedder.embed("test text")
            assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder):
        """Test embed_batch method."""
        mock_aio_client = MagicMock()
        mock_models = MagicMock()
        mock_embedding1 = MagicMock()
        mock_embedding1.values = [0.1, 0.2]
        mock_embedding2 = MagicMock()
        mock_embedding2.values = [0.3, 0.4]
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding1, mock_embedding2]
        mock_models.embed_content = AsyncMock(return_value=mock_response)
        mock_aio_client.models = mock_models

        with patch.object(embedder, "_get_aio_client", return_value=mock_aio_client):
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
        mock_models = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding]
        mock_models.embed_content.return_value = mock_response
        mock_client.models = mock_models

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed_sync("test text")
            assert result == [0.1, 0.2, 0.3]

    def test_embed_batch_sync(self, embedder):
        """Test embed_batch_sync method."""
        mock_client = MagicMock()
        mock_models = MagicMock()
        mock_embedding1 = MagicMock()
        mock_embedding1.values = [0.1, 0.2]
        mock_embedding2 = MagicMock()
        mock_embedding2.values = [0.3, 0.4]
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding1, mock_embedding2]
        mock_models.embed_content.return_value = mock_response
        mock_client.models = mock_models

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = embedder.embed_batch_sync(["text1", "text2"])
            assert result == [[0.1, 0.2], [0.3, 0.4]]
