from unittest.mock import patch

import pytest

from hypertic.embedders.cohere import CohereEmbedder
from hypertic.embedders.google import GoogleEmbedder
from hypertic.embedders.huggingface import HuggingFaceEmbedder
from hypertic.embedders.mistral import MistralEmbedder
from hypertic.embedders.openai import OpenAIEmbedder


class TestEmbeddersIntegration:
    @pytest.mark.parametrize(
        "embedder_class,embedder_params",
        [
            (OpenAIEmbedder, {"api_key": "test", "model": "text-embedding-3-small"}),
            (CohereEmbedder, {"api_key": "test", "model": "embed-english-v3.0"}),
            (GoogleEmbedder, {"api_key": "test", "model": "text-embedding-004"}),
            (HuggingFaceEmbedder, {"api_key": "test", "model": "intfloat/multilingual-e5-large"}),
            (MistralEmbedder, {"api_key": "test", "model": "mistral-embed"}),
        ],
    )
    def test_embedder_initialization(self, embedder_class, embedder_params):
        embedder = embedder_class(**embedder_params)
        assert embedder is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "embedder_class,embedder_params",
        [
            (OpenAIEmbedder, {"api_key": "test", "model": "text-embedding-3-small"}),
            (CohereEmbedder, {"api_key": "test", "model": "embed-english-v3.0"}),
            (GoogleEmbedder, {"api_key": "test", "model": "text-embedding-004"}),
            (HuggingFaceEmbedder, {"api_key": "test", "model": "intfloat/multilingual-e5-large"}),
            (MistralEmbedder, {"api_key": "test", "model": "mistral-embed"}),
        ],
    )
    async def test_embedder_async_initialize(self, embedder_class, embedder_params):
        embedder = embedder_class(**embedder_params)

        if hasattr(embedder, "_get_async_client"):
            method_name = "_get_async_client"
        elif hasattr(embedder, "_get_aio_client"):
            method_name = "_get_aio_client"
        else:
            method_name = "_get_client"
        with patch.object(embedder, method_name):
            result = await embedder.initialize()
            assert isinstance(result, bool)

    @pytest.mark.parametrize(
        "embedder_class,embedder_params",
        [
            (OpenAIEmbedder, {"api_key": "test", "model": "text-embedding-3-small"}),
            (CohereEmbedder, {"api_key": "test", "model": "embed-english-v3.0"}),
            (GoogleEmbedder, {"api_key": "test", "model": "text-embedding-004"}),
            (HuggingFaceEmbedder, {"api_key": "test", "model": "intfloat/multilingual-e5-large"}),
            (MistralEmbedder, {"api_key": "test", "model": "mistral-embed"}),
        ],
    )
    def test_embedder_sync_initialize(self, embedder_class, embedder_params):
        embedder = embedder_class(**embedder_params)

        with patch.object(embedder, "_get_client"):
            result = embedder.initialize_sync()
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "embedder_class,embedder_params",
        [
            (OpenAIEmbedder, {"api_key": "test", "model": "text-embedding-3-small"}),
            (CohereEmbedder, {"api_key": "test", "model": "embed-english-v3.0"}),
            (GoogleEmbedder, {"api_key": "test", "model": "text-embedding-004"}),
            (HuggingFaceEmbedder, {"api_key": "test", "model": "intfloat/multilingual-e5-large"}),
            (MistralEmbedder, {"api_key": "test", "model": "mistral-embed"}),
        ],
    )
    async def test_embedder_async_embed(self, embedder_class, embedder_params, mock_embedding):
        embedder = embedder_class(**embedder_params)

        with patch.object(embedder, "embed", return_value=mock_embedding):
            result = await embedder.embed("test text")
            assert isinstance(result, list)
            assert len(result) > 0

    @pytest.mark.parametrize(
        "embedder_class,embedder_params",
        [
            (OpenAIEmbedder, {"api_key": "test", "model": "text-embedding-3-small"}),
            (CohereEmbedder, {"api_key": "test", "model": "embed-english-v3.0"}),
            (GoogleEmbedder, {"api_key": "test", "model": "text-embedding-004"}),
            (HuggingFaceEmbedder, {"api_key": "test", "model": "intfloat/multilingual-e5-large"}),
            (MistralEmbedder, {"api_key": "test", "model": "mistral-embed"}),
        ],
    )
    def test_embedder_sync_embed(self, embedder_class, embedder_params, mock_embedding):
        embedder = embedder_class(**embedder_params)

        with patch.object(embedder, "embed_sync", return_value=mock_embedding):
            result = embedder.embed_sync("test text")
            assert isinstance(result, list)
            assert len(result) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "embedder_class,embedder_params",
        [
            (OpenAIEmbedder, {"api_key": "test", "model": "text-embedding-3-small"}),
            (CohereEmbedder, {"api_key": "test", "model": "embed-english-v3.0"}),
            (GoogleEmbedder, {"api_key": "test", "model": "text-embedding-004"}),
            (HuggingFaceEmbedder, {"api_key": "test", "model": "intfloat/multilingual-e5-large"}),
            (MistralEmbedder, {"api_key": "test", "model": "mistral-embed"}),
        ],
    )
    async def test_embedder_async_batch(self, embedder_class, embedder_params, mock_embeddings):
        embedder = embedder_class(**embedder_params)

        with patch.object(embedder, "embed_batch", return_value=mock_embeddings):
            result = await embedder.embed_batch(["text1", "text2", "text3"])
            assert isinstance(result, list)
            assert len(result) == 3

    @pytest.mark.parametrize(
        "embedder_class,embedder_params",
        [
            (OpenAIEmbedder, {"api_key": "test", "model": "text-embedding-3-small"}),
            (CohereEmbedder, {"api_key": "test", "model": "embed-english-v3.0"}),
            (GoogleEmbedder, {"api_key": "test", "model": "text-embedding-004"}),
            (HuggingFaceEmbedder, {"api_key": "test", "model": "intfloat/multilingual-e5-large"}),
            (MistralEmbedder, {"api_key": "test", "model": "mistral-embed"}),
        ],
    )
    def test_embedder_sync_batch(self, embedder_class, embedder_params, mock_embeddings):
        embedder = embedder_class(**embedder_params)

        with patch.object(embedder, "embed_batch_sync", return_value=mock_embeddings):
            result = embedder.embed_batch_sync(["text1", "text2", "text3"])
            assert isinstance(result, list)
            assert len(result) == 3
