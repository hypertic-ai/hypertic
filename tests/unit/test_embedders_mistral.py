from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.embedders.mistral.mistral import MistralEmbedder


@pytest.mark.unit
class TestMistralEmbedder:
    @pytest.fixture
    def embedder(self, mock_api_key):
        with patch("hypertic.embedders.mistral.mistral.Mistral"):
            return MistralEmbedder(api_key=mock_api_key, model="mistral-embed")

    def test_mistral_embedder_creation(self, embedder, mock_api_key):
        assert embedder.api_key == mock_api_key
        assert embedder.model == "mistral-embed"

    @patch("hypertic.embedders.mistral.mistral.getenv")
    def test_mistral_embedder_with_env_api_key(self, mock_getenv, mock_api_key):
        mock_getenv.return_value = "env_key"
        with patch("hypertic.embedders.mistral.mistral.Mistral"):
            embedder = MistralEmbedder(model="mistral-embed")
            assert embedder.api_key == "env_key"

    def test_mistral_embedder_with_custom_params(self, mock_api_key):
        with patch("hypertic.embedders.mistral.mistral.Mistral"):
            embedder = MistralEmbedder(
                api_key=mock_api_key,
                model="custom-model",
                endpoint="https://custom.endpoint",
                max_retries=5,
                timeout=30,
            )
            assert embedder.model == "custom-model"
            assert embedder.endpoint == "https://custom.endpoint"
            assert embedder.max_retries == 5
            assert embedder.timeout == 30

    def test_get_client(self, embedder):
        """Test _get_client creates Mistral client."""
        mock_mistral = MagicMock()
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client

        with patch("hypertic.embedders.mistral.mistral.Mistral", mock_mistral):
            embedder.client = None
            client = embedder._get_client()
            assert client == mock_client
            mock_mistral.assert_called_once_with(api_key=embedder.api_key)

    def test_get_client_with_all_params(self, embedder):
        """Test _get_client with all parameters."""
        embedder.endpoint = "https://custom.endpoint"
        embedder.max_retries = 5
        embedder.timeout = 30
        embedder.client_params = {"custom": "param"}

        mock_mistral = MagicMock()
        mock_client = MagicMock()
        mock_mistral.return_value = mock_client

        with patch("hypertic.embedders.mistral.mistral.Mistral", mock_mistral):
            embedder.client = None
            client = embedder._get_client()
            assert client == mock_client
            call_kwargs = mock_mistral.call_args[1]
            assert call_kwargs["api_key"] == embedder.api_key
            assert call_kwargs["endpoint"] == "https://custom.endpoint"
            assert call_kwargs["max_retries"] == 5
            assert call_kwargs["timeout"] == 30
            assert call_kwargs["custom"] == "param"

    def test_get_request_params(self, embedder):
        """Test _get_request_params."""
        params = embedder._get_request_params("test text")
        assert params["model"] == "mistral-embed"
        assert params["inputs"] == ["test text"]

    def test_get_request_params_with_custom(self, embedder):
        """Test _get_request_params with custom request_params."""
        embedder.request_params = {"custom": "value"}
        params = embedder._get_request_params("test")
        assert params["custom"] == "value"

    def test_response(self, embedder):
        """Test _response method."""
        from hypertic.embedders.mistral.mistral import EmbeddingResponse

        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock(spec=EmbeddingResponse)
        mock_response.data = [mock_data]
        mock_embeddings.create.return_value = mock_response
        mock_client.embeddings = mock_embeddings

        with patch.object(embedder, "_get_client", return_value=mock_client):
            response = embedder._response("test text")
            assert response == mock_response

    @pytest.mark.asyncio
    async def test_initialize(self, embedder):
        """Test initialize method."""
        mock_client = MagicMock()
        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = await embedder.initialize()
            assert result is True

    @pytest.mark.asyncio
    async def test_initialize_error(self, embedder):
        """Test initialize handles errors."""
        with patch.object(embedder, "_get_client", side_effect=Exception("Error")):
            result = await embedder.initialize()
            assert result is False

    @pytest.mark.asyncio
    async def test_embed(self, embedder):
        """Test embed method."""
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_data]
        mock_embeddings.create_async = AsyncMock(return_value=mock_response)
        mock_client.embeddings = mock_embeddings

        with patch.object(embedder, "_get_client", return_value=mock_client):
            result = await embedder.embed("test text")
            assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_without_async_method(self, embedder):
        """Test embed when create_async doesn't exist."""
        from hypertic.embedders.mistral.mistral import EmbeddingResponse

        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock(spec=EmbeddingResponse)
        mock_response.data = [mock_data]
        mock_embeddings.create = Mock(return_value=mock_response)
        # Remove create_async attribute to simulate it not existing
        del mock_embeddings.create_async
        mock_client.embeddings = mock_embeddings

        with patch.object(embedder, "_get_client", return_value=mock_client):
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_response)
                result = await embedder.embed("test text")
                assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_empty_response(self, embedder):
        """Test embed with empty response."""
        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_response = MagicMock()
        mock_response.data = []
        mock_embeddings.create_async = AsyncMock(return_value=mock_response)
        mock_client.embeddings = mock_embeddings

        with patch.object(embedder, "_get_client", return_value=mock_client):
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
        mock_embeddings.create_async = AsyncMock(return_value=mock_response)
        mock_client.embeddings = mock_embeddings

        with patch.object(embedder, "_get_client", return_value=mock_client):
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
        from hypertic.embedders.mistral.mistral import EmbeddingResponse

        mock_client = MagicMock()
        mock_embeddings = MagicMock()
        mock_data = MagicMock()
        mock_data.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock(spec=EmbeddingResponse)
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
