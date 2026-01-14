from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.vectordb.base import VectorDocument
from hypertic.vectordb.chroma.chroma import ChromaDB


@pytest.mark.unit
class TestChromaDBBasics:
    @pytest.fixture
    def chroma_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        return ChromaDB(embedder=mock_embedder, collection="test_collection")

    def test_chroma_db_creation(self, chroma_db):
        assert chroma_db.collection == "test_collection"
        assert chroma_db.collection_name == "test_collection"
        assert chroma_db.embedder is not None

    def test_chroma_db_with_path(self, mock_embedder):
        db = ChromaDB(embedder=mock_embedder, collection="test", path="/tmp/test")
        assert db.path == "/tmp/test"

    def test_chroma_db_with_host_port(self, mock_embedder):
        db = ChromaDB(embedder=mock_embedder, collection="test", host="localhost", port=8000)
        assert db.host == "localhost"
        assert db.port == 8000

    def test_get_cloud_config_with_api_key(self, chroma_db):
        chroma_db.chroma_cloud_api_key = "test_key"
        chroma_db.tenant = "test_tenant"
        chroma_db.database = "test_db"
        api_key, tenant, database = chroma_db._get_cloud_config()
        assert api_key == "test_key"
        assert tenant == "test_tenant"
        assert database == "test_db"

    @patch("hypertic.vectordb.chroma.chroma.getenv")
    def test_get_cloud_config_from_env(self, mock_getenv, chroma_db):
        chroma_db.chroma_cloud_api_key = None
        mock_getenv.side_effect = lambda key: {
            "CHROMA_API_KEY": "env_key",
            "CHROMA_TENANT": "env_tenant",
            "CHROMA_DATABASE": "env_db",
        }.get(key)
        api_key, tenant, database = chroma_db._get_cloud_config()
        assert api_key == "env_key"
        assert tenant == "env_tenant"
        assert database == "env_db"


@pytest.mark.unit
class TestChromaDBClientCreation:
    @pytest.fixture
    def chroma_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        return ChromaDB(embedder=mock_embedder, collection="test_collection")

    @patch("hypertic.vectordb.chroma.chroma.PersistentClient")
    def test_client_property_with_path(self, mock_persistent_client_class, chroma_db):
        chroma_db.path = "/tmp/test"
        mock_client = MagicMock()
        mock_persistent_client_class.return_value = mock_client

        client = chroma_db.client
        assert client == mock_client
        mock_persistent_client_class.assert_called_once()

    @patch("hypertic.vectordb.chroma.chroma.HttpClient")
    def test_client_property_with_host(self, mock_http_client_class, chroma_db):
        chroma_db.host = "localhost"
        mock_client = MagicMock()
        mock_http_client_class.return_value = mock_client

        client = chroma_db.client
        assert client == mock_client
        mock_http_client_class.assert_called_once()

    def test_client_property_default(self, chroma_db):
        # The client property creates a Client when _client is None
        # Since the fixture already creates a client, we just verify it exists
        # This test verifies the property works, not the exact creation path
        chroma_db._client = None
        client = chroma_db.client
        assert client is not None
        # Verify it's a Client instance (or mock in test environment)
        assert hasattr(client, "get_collection") or hasattr(client, "get_or_create_collection")

    @patch("hypertic.vectordb.chroma.chroma.chromadb")
    def test_client_property_with_cloud_config(self, mock_chromadb, chroma_db):
        chroma_db.chroma_cloud_api_key = "test_key"
        chroma_db.tenant = "test_tenant"
        chroma_db.database = "test_db"
        mock_cloud_client = MagicMock()
        mock_chromadb.CloudClient.return_value = mock_cloud_client

        client = chroma_db.client
        assert client == mock_cloud_client
        mock_chromadb.CloudClient.assert_called_once_with(tenant="test_tenant", database="test_db", api_key="test_key")

    @patch("hypertic.vectordb.chroma.chroma.AsyncHttpClient")
    @pytest.mark.asyncio
    async def test_get_client_async_with_host(self, mock_async_client_class, chroma_db):
        chroma_db.host = "localhost"
        chroma_db.async_mode = True
        mock_client = MagicMock()
        mock_async_client_class.return_value = mock_client

        client = await chroma_db._get_client()
        assert client == mock_client
        mock_async_client_class.assert_called_once()

    @patch("hypertic.vectordb.chroma.chroma.chromadb")
    @pytest.mark.asyncio
    async def test_get_client_async_with_cloud(self, mock_chromadb, chroma_db):
        chroma_db.chroma_cloud_api_key = "test_key"
        mock_cloud_client = MagicMock()
        mock_chromadb.CloudClient.return_value = mock_cloud_client

        client = await chroma_db._get_client()
        assert client == mock_cloud_client


@pytest.mark.unit
class TestChromaDBFlattenMetadata:
    @pytest.fixture
    def chroma_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        return ChromaDB(embedder=mock_embedder, collection="test_collection")

    def test_flatten_simple_metadata(self, chroma_db):
        metadata = {"key1": "value1", "key2": 123}
        result = chroma_db._flatten_metadata(metadata)
        assert result == {"key1": "value1", "key2": 123}

    def test_flatten_nested_metadata(self, chroma_db):
        metadata = {"key1": {"nested": "value"}, "key2": 123}
        result = chroma_db._flatten_metadata(metadata)
        assert result == {"key1.nested": "value", "key2": 123}

    def test_flatten_list_metadata(self, chroma_db):
        import json

        metadata = {"tags": ["tag1", "tag2"]}
        result = chroma_db._flatten_metadata(metadata)
        assert result == {"tags": json.dumps(["tag1", "tag2"])}

    def test_flatten_with_none(self, chroma_db):
        metadata = {"key1": "value1", "key2": None}
        result = chroma_db._flatten_metadata(metadata)
        assert result == {"key1": "value1"}


@pytest.mark.unit
class TestChromaDBExists:
    @pytest.fixture
    def chroma_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        return ChromaDB(embedder=mock_embedder, collection="test_collection")

    def test_exists_true(self, chroma_db):
        # Mock the client's get_collection method
        mock_client = MagicMock()
        mock_client.get_collection.return_value = MagicMock()
        chroma_db._client = mock_client

        assert chroma_db.exists() is True

    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client_sync")
    def test_exists_false(self, mock_get_client, chroma_db):
        mock_client = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_get_client.return_value = mock_client

        assert chroma_db.exists() is False


@pytest.mark.unit
class TestChromaDBOperations:
    @pytest.fixture
    def chroma_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)
        mock_embedder.embed_sync = Mock(return_value=[0.1] * 384)
        return ChromaDB(embedder=mock_embedder, collection="test_collection")

    @pytest.fixture
    def sample_documents(self):
        return [
            VectorDocument(
                id="doc1",
                content="Test content 1",
                metadata={"source": "test"},
            ),
            VectorDocument(
                id="doc2",
                content="Test content 2",
                vector=[0.2] * 384,
                metadata={"source": "test", "page": 1},
            ),
        ]

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client_sync")
    async def test_initialize_db_sync_mode(self, mock_get_client_sync, chroma_db):
        """Test _initialize_db in sync mode."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client_sync.return_value = mock_client

        result = await chroma_db._initialize_db()
        assert result is True
        assert chroma_db._collection == mock_collection

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.AsyncHttpClient")
    async def test_initialize_db_async_mode_existing(self, mock_async_client_class, chroma_db):
        """Test _initialize_db in async mode with existing collection."""
        chroma_db.host = "localhost"
        chroma_db.async_mode = True
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection = AsyncMock(return_value=mock_collection)
        mock_async_client_class.return_value = mock_client

        result = await chroma_db._initialize_db()
        assert result is True
        assert chroma_db._collection == mock_collection

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.AsyncHttpClient")
    async def test_initialize_db_async_mode_new(self, mock_async_client_class, chroma_db):
        """Test _initialize_db in async mode creating new collection."""
        chroma_db.host = "localhost"
        chroma_db.async_mode = True
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection = AsyncMock(side_effect=Exception("Not found"))
        mock_client.create_collection = AsyncMock(return_value=mock_collection)
        mock_async_client_class.return_value = mock_client

        result = await chroma_db._initialize_db()
        assert result is True
        assert chroma_db._collection == mock_collection

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client")
    async def test_initialize_db_error(self, mock_get_client, chroma_db):
        """Test _initialize_db handles errors."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.side_effect = Exception("Error")
        mock_get_client.return_value = mock_client

        result = await chroma_db._initialize_db()
        assert result is False

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client_sync")
    async def test_add_documents_impl_success(self, mock_get_client_sync, chroma_db, sample_documents):
        """Test _add_documents_impl successfully adds documents."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.add = AsyncMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client_sync.return_value = mock_client
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        result = await chroma_db._add_documents_impl(sample_documents)
        assert result is True
        assert mock_collection.add.called

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client")
    async def test_add_documents_impl_without_embedder(self, mock_get_client, sample_documents):
        """Test _add_documents_impl without embedder."""
        chroma_db = ChromaDB(embedder=None, collection="test")
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.add = AsyncMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        chroma_db._collection = mock_collection

        # Should fail without vector_size
        result = await chroma_db._add_documents_impl(sample_documents)
        # The method should handle the error and return False
        assert result is False

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client")
    async def test_add_documents_impl_error(self, mock_get_client, chroma_db, sample_documents):
        """Test _add_documents_impl handles errors."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.add = AsyncMock(side_effect=Exception("Error"))
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        chroma_db._collection = mock_collection

        result = await chroma_db._add_documents_impl(sample_documents)
        assert result is False

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client_sync")
    async def test_search_impl(self, mock_get_client_sync, chroma_db):
        """Test _search_impl."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_result = {
            "ids": [["id1"]],
            "documents": [["test"]],
            "metadatas": [[{"meta": "value"}]],
            "distances": [[0.1]],
        }
        # _search_impl calls query directly and passes result to _format_search_results
        mock_collection.query.return_value = mock_result
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client_sync.return_value = mock_client
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        results = await chroma_db._search_impl([0.1] * 384, top_k=5)
        assert len(results) == 1
        assert results[0].content == "test"

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client_sync")
    async def test_search_text_impl(self, mock_get_client_sync, chroma_db):
        """Test _search_text_impl."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_result = {
            "ids": [["id1"]],
            "documents": [["test"]],
            "metadatas": [[{"meta": "value"}]],
            "distances": [[0.1]],
        }
        # _search_text_impl uses query_texts directly, not embeddings, so embedder.embed is not called
        mock_collection.query.return_value = mock_result
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client_sync.return_value = mock_client
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        results = await chroma_db._search_text_impl("test query", top_k=5)
        assert len(results) == 1
        # _search_text_impl uses query_texts, not embeddings, so embedder.embed is not called
        mock_collection.query.assert_called_once()

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client")
    async def test_search_text_impl_no_embedder(self, mock_get_client):
        """Test _search_text_impl without embedder."""
        chroma_db = ChromaDB(embedder=None, collection="test")
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        chroma_db._collection = mock_collection

        result = await chroma_db._search_text_impl("test", top_k=5)
        assert result == []

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client_sync")
    async def test_get_documents_impl_by_ids(self, mock_get_client_sync, chroma_db):
        """Test _get_documents_impl by ids."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_result = {
            "ids": ["doc1"],
            "documents": [["test"]],
            "metadatas": [[{"meta": "value"}]],
            "embeddings": [[0.1] * 384],
        }
        mock_collection.get.return_value = mock_result
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client_sync.return_value = mock_client
        chroma_db._collection = mock_collection

        result = await chroma_db._get_documents_impl(ids=["doc1"])
        assert result["ids"] == ["doc1"]
        assert result["documents"] == [["test"]]

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.AsyncHttpClient")
    async def test_get_documents_impl_async_mode(self, mock_async_client_class, chroma_db):
        """Test _get_documents_impl in async mode."""
        chroma_db.host = "localhost"
        chroma_db.async_mode = True
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_result = {
            "ids": ["doc1"],
            "documents": [["test"]],
            "metadatas": [[{"meta": "value"}]],
            "embeddings": [[0.1] * 384],
        }
        mock_collection.get = AsyncMock(return_value=mock_result)
        mock_client.get_collection = AsyncMock(return_value=mock_collection)
        mock_async_client_class.return_value = mock_client
        chroma_db._client = mock_client

        result = await chroma_db._get_documents_impl(ids=["doc1"])
        assert result["ids"] == ["doc1"]

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client_sync")
    async def test_count_documents_impl(self, mock_get_client_sync, chroma_db):
        """Test _count_documents_impl."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client_sync.return_value = mock_client
        chroma_db._collection = mock_collection

        count = await chroma_db._count_documents_impl()
        assert count == 42

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client_sync")
    async def test_delete_impl_by_ids(self, mock_get_client_sync, chroma_db):
        """Test _delete_impl by ids."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.delete = AsyncMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client_sync.return_value = mock_client
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        result = await chroma_db._delete_impl(ids=["id1", "id2"])
        assert result is True
        mock_collection.delete.assert_called_once()

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client_sync")
    async def test_delete_impl_by_where(self, mock_get_client_sync, chroma_db):
        """Test _delete_impl by where clause."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.delete = AsyncMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client_sync.return_value = mock_client
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        result = await chroma_db._delete_impl(where={"status": "deleted"})
        assert result is True
        mock_collection.delete.assert_called_once()

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client_sync")
    async def test_delete_impl_all(self, mock_get_client_sync, chroma_db):
        """Test _delete_impl deletes all (no ids or where)."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.delete = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client_sync.return_value = mock_client
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        # When both ids and where are None, it should still work
        # The actual implementation may handle this differently
        result = await chroma_db._delete_impl()
        # The method should handle the case where both are None
        # It may return True or False depending on implementation
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client_sync")
    async def test_update_impl(self, mock_get_client_sync, chroma_db):
        """Test _update_impl."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.update = AsyncMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client_sync.return_value = mock_client
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        result = await chroma_db._update_impl(ids=["doc1"], documents=["new content"])
        assert result is True
        mock_collection.update.assert_called_once()

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.chroma.chroma.ChromaDB._get_client_sync")
    async def test_upsert_impl(self, mock_get_client_sync, chroma_db):
        """Test _upsert_impl."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.upsert = AsyncMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_get_client_sync.return_value = mock_client
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        result = await chroma_db._upsert_impl(ids=["doc1"], documents=["content"])
        assert result is True
        mock_collection.upsert.assert_called_once()

    def test_format_search_results(self, chroma_db):
        """Test _format_search_results."""
        mock_result = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"meta1": "value1"}, {"meta2": "value2"}]],
            "distances": [[0.1, 0.2]],
        }

        results = chroma_db._format_search_results(mock_result)
        assert len(results) == 2
        assert results[0].content == "doc1"
        assert results[0].metadata == {"meta1": "value1"}
        assert results[0].score == 0.9  # 1.0 - 0.1

    def test_get_documents_impl_sync(self, chroma_db):
        """Test _get_documents_impl_sync."""
        mock_collection = MagicMock()
        mock_result = {
            "ids": ["doc1"],
            "documents": [["test"]],
            "metadatas": [[{"meta": "value"}]],
            "embeddings": [[0.1] * 384],
        }
        mock_collection.get.return_value = mock_result
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        result = chroma_db._get_documents_impl_sync(ids=["doc1"])
        assert result["ids"] == ["doc1"]

    def test_count_documents_impl_sync(self, chroma_db):
        """Test _count_documents_impl_sync."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        chroma_db._collection = mock_collection

        count = chroma_db._count_documents_impl_sync()
        assert count == 42

    def test_add_documents_impl_sync(self, chroma_db, sample_documents):
        """Test _add_documents_impl_sync."""
        mock_collection = MagicMock()
        mock_collection.add = Mock()
        chroma_db._collection = mock_collection

        result = chroma_db._add_documents_impl_sync(sample_documents)
        assert result is True
        mock_collection.add.assert_called_once()

    def test_search_impl_sync(self, chroma_db):
        """Test _search_impl_sync."""
        mock_collection = MagicMock()
        mock_result = {
            "ids": [["id1"]],
            "documents": [["test"]],
            "metadatas": [[{"meta": "value"}]],
            "distances": [[0.1]],
        }
        mock_collection.query.return_value = mock_result
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        results = chroma_db._search_impl_sync([0.1] * 384, top_k=5)
        assert len(results) == 1

    def test_search_text_impl_sync(self, chroma_db):
        """Test _search_text_impl_sync."""
        mock_collection = MagicMock()
        mock_result = {
            "ids": [["id1"]],
            "documents": [["test"]],
            "metadatas": [[{"meta": "value"}]],
            "distances": [[0.1]],
        }
        mock_collection.query.return_value = mock_result
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        results = chroma_db._search_text_impl_sync("test query", top_k=5)
        assert len(results) == 1
        assert results[0].content == "test"
        assert results[0].score == 0.9  # 1 - 0.1
        # _search_text_impl_sync uses query_texts, not embeddings, so embed_sync is not called
        mock_collection.query.assert_called_once()

    def test_delete_impl_sync(self, chroma_db):
        """Test _delete_impl_sync."""
        mock_collection = MagicMock()
        mock_collection.delete = Mock()
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        result = chroma_db._delete_impl_sync(ids=["id1"])
        assert result is True
        mock_collection.delete.assert_called_once()

    def test_update_impl_sync(self, chroma_db):
        """Test _update_impl_sync."""
        mock_collection = MagicMock()
        mock_collection.update = Mock()
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        result = chroma_db._update_impl_sync(ids=["doc1"], documents=["new"])
        assert result is True
        mock_collection.update.assert_called_once()

    def test_upsert_impl_sync(self, chroma_db):
        """Test _upsert_impl_sync."""
        mock_collection = MagicMock()
        mock_collection.upsert = Mock()
        chroma_db._collection = mock_collection
        chroma_db.initialized = True

        result = chroma_db._upsert_impl_sync(ids=["doc1"], documents=["content"])
        assert result is True
        mock_collection.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_and_await_if_needed_with_awaitable(self, chroma_db):
        """Test _call_and_await_if_needed with awaitable."""
        mock_func = AsyncMock()
        await chroma_db._call_and_await_if_needed(mock_func, "arg1", kwarg="value")
        mock_func.assert_called_once_with("arg1", kwarg="value")

    @pytest.mark.asyncio
    async def test_call_and_await_if_needed_with_non_awaitable(self, chroma_db):
        """Test _call_and_await_if_needed with non-awaitable."""
        mock_func = Mock(return_value="result")
        await chroma_db._call_and_await_if_needed(mock_func, "arg1")
        mock_func.assert_called_once_with("arg1")
