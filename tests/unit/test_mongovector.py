from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.vectordb.base import VectorDocument
from hypertic.vectordb.mongovector.mongovector import MongoDBAtlas


@pytest.mark.unit
class TestMongoDBAtlasBasics:
    @pytest.fixture
    def mongodb_atlas(self, mock_embedder):
        with patch("hypertic.vectordb.mongovector.mongovector.getenv", return_value="mongodb://test"):
            return MongoDBAtlas(embedder=mock_embedder, collection="test_collection", connection_string="mongodb://test")

    def test_mongodb_atlas_creation(self, mongodb_atlas):
        assert mongodb_atlas.collection == "test_collection"
        assert mongodb_atlas.collection_name == "test_collection"
        assert mongodb_atlas.embedder is not None
        assert mongodb_atlas.database_name == "vectordb"

    def test_mongodb_atlas_with_connection_string(self, mock_embedder):
        db = MongoDBAtlas(embedder=mock_embedder, collection="test", connection_string="mongodb://custom")
        assert db.connection_string == "mongodb://custom"

    @patch("hypertic.vectordb.mongovector.mongovector.getenv")
    def test_mongodb_atlas_with_env_connection_string(self, mock_getenv, mock_embedder):
        mock_getenv.return_value = "mongodb://env"
        db = MongoDBAtlas(embedder=mock_embedder, collection="test")
        assert db.connection_string == "mongodb://env"

    @patch("hypertic.vectordb.mongovector.mongovector.getenv")
    def test_mongodb_atlas_no_connection_string(self, mock_getenv, mock_embedder):
        mock_getenv.return_value = None
        with pytest.raises(ValueError, match="MongoDB connection string is required"):
            MongoDBAtlas(embedder=mock_embedder, collection="test")

    def test_mongodb_atlas_with_custom_database(self, mock_embedder):
        with patch("hypertic.vectordb.mongovector.mongovector.getenv", return_value="mongodb://test"):
            db = MongoDBAtlas(embedder=mock_embedder, collection="test", database_name="custom_db")
            assert db.database_name == "custom_db"

    def test_mongodb_atlas_with_custom_vector_field(self, mock_embedder):
        with patch("hypertic.vectordb.mongovector.mongovector.getenv", return_value="mongodb://test"):
            db = MongoDBAtlas(embedder=mock_embedder, collection="test", vector_field="custom_vector")
            assert db.vector_field == "custom_vector"

    def test_get_client(self, mongodb_atlas):
        """Test _get_client creates MongoClient."""
        mock_mongo_client = MagicMock()
        with patch("hypertic.vectordb.mongovector.mongovector.MongoClient", return_value=mock_mongo_client):
            mongodb_atlas._client = None
            client = mongodb_atlas._get_client()
            assert client == mock_mongo_client

    def test_get_database(self, mongodb_atlas):
        """Test _get_database creates Database."""
        mock_client = MagicMock()
        mock_database = MagicMock()
        mock_client.__getitem__.return_value = mock_database

        with patch.object(mongodb_atlas, "_get_client", return_value=mock_client):
            mongodb_atlas._database = None
            database = mongodb_atlas._get_database()
            assert database == mock_database

    def test_get_collection(self, mongodb_atlas):
        """Test _get_collection creates Collection."""
        mock_database = MagicMock()
        mock_collection = MagicMock()
        mock_database.__getitem__.return_value = mock_collection

        with patch.object(mongodb_atlas, "_get_database", return_value=mock_database):
            mongodb_atlas._collection = None
            collection = mongodb_atlas._get_collection()
            assert collection == mock_collection

    @pytest.mark.asyncio
    async def test_get_async_client(self, mongodb_atlas):
        """Test _get_async_client creates AsyncMongoClient."""
        mock_async_client = MagicMock()
        mock_admin = MagicMock()
        mock_command = AsyncMock()
        mock_admin.command = mock_command
        mock_async_client.admin = mock_admin
        mock_async_client.__getitem__.return_value = MagicMock()

        with patch("hypertic.vectordb.mongovector.mongovector.AsyncMongoClient", return_value=mock_async_client):
            mongodb_atlas._async_client = None
            client = await mongodb_atlas._get_async_client()
            assert client == mock_async_client

    @pytest.mark.asyncio
    async def test_get_async_collection(self, mongodb_atlas):
        """Test _get_async_collection creates Collection."""
        mock_async_client = MagicMock()
        mock_database = MagicMock()
        mock_collection = MagicMock()
        mock_async_client.__getitem__.return_value = mock_database
        mock_database.__getitem__.return_value = mock_collection

        async def get_async_client():
            mongodb_atlas._async_database = mock_database
            return mock_async_client

        with patch.object(mongodb_atlas, "_get_async_client", side_effect=get_async_client):
            mongodb_atlas._async_collection = None
            collection = await mongodb_atlas._get_async_collection()
            assert collection == mock_collection

    @pytest.mark.asyncio
    async def test_maybe_await_async(self, mongodb_atlas):
        """Test _maybe_await_async handles awaitable and non-awaitable."""
        # Non-awaitable
        result = await mongodb_atlas._maybe_await_async("test")
        assert result == "test"

        # Awaitable
        async def async_func():
            return "async_result"

        result = await mongodb_atlas._maybe_await_async(async_func())
        assert result == "async_result"


@pytest.mark.unit
class TestMongoDBAtlasInitialization:
    @pytest.fixture
    def mongodb_atlas(self, mock_embedder):
        with patch("hypertic.vectordb.mongovector.mongovector.getenv", return_value="mongodb://test"):
            return MongoDBAtlas(embedder=mock_embedder, collection="test_collection", connection_string="mongodb://test")

    @pytest.mark.asyncio
    async def test_initialize_db(self, mongodb_atlas):
        """Test _initialize_db."""
        mock_async_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(return_value={"_id": "test"})
        mock_collection.create_search_index = AsyncMock()

        async def get_async_collection():
            return mock_collection

        with patch.object(mongodb_atlas, "_get_async_client", return_value=mock_async_client):
            with patch.object(mongodb_atlas, "_get_async_collection", side_effect=get_async_collection):
                with patch.object(mongodb_atlas, "_create_vector_search_index", new_callable=AsyncMock):
                    result = await mongodb_atlas._initialize_db()
                    assert result is True
                    assert mongodb_atlas._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_db_collection_not_ready(self, mongodb_atlas):
        """Test _initialize_db when collection is not ready."""
        mock_async_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(side_effect=Exception("Not ready"))
        mock_collection.insert_one = AsyncMock()
        mock_collection.delete_one = AsyncMock()

        async def get_async_collection():
            return mock_collection

        with patch.object(mongodb_atlas, "_get_async_client", return_value=mock_async_client):
            with patch.object(mongodb_atlas, "_get_async_collection", side_effect=get_async_collection):
                with patch.object(mongodb_atlas, "_create_vector_search_index", new_callable=AsyncMock):
                    with patch("asyncio.sleep", new_callable=AsyncMock):
                        result = await mongodb_atlas._initialize_db()
                        assert result is True

    def test_initialize_db_sync(self, mongodb_atlas):
        """Test _initialize_db_sync."""
        mock_client = MagicMock()
        mock_admin = MagicMock()
        mock_admin.command = Mock()
        mock_client.admin = mock_admin
        mock_collection = MagicMock()
        mock_collection.find_one = Mock(return_value={"_id": "test"})
        mock_database = MagicMock()
        mock_database.__getitem__.return_value = mock_collection

        with patch.object(mongodb_atlas, "_get_client", return_value=mock_client):
            with patch.object(mongodb_atlas, "_get_database", return_value=mock_database):
                with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
                    with patch.object(mongodb_atlas, "_create_vector_search_index_sync", new_callable=Mock):
                        result = mongodb_atlas._initialize_db_sync()
                        assert result is True

    @pytest.mark.asyncio
    async def test_create_vector_search_index(self, mongodb_atlas):
        """Test _create_vector_search_index."""
        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(return_value={"_id": "test"})
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_collection.list_search_indexes = AsyncMock(return_value=mock_cursor)
        mock_collection.create_search_index = AsyncMock()

        with patch.object(mongodb_atlas, "_get_async_collection", return_value=mock_collection):
            await mongodb_atlas._create_vector_search_index()
            mock_collection.create_search_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_vector_search_index_already_exists(self, mongodb_atlas):
        """Test _create_vector_search_index when index already exists."""
        mock_collection = MagicMock()
        mock_collection.find_one = AsyncMock(return_value={"_id": "test"})
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[{"name": "test_collection_vector_index"}])
        mock_collection.list_search_indexes = AsyncMock(return_value=mock_cursor)

        with patch.object(mongodb_atlas, "_get_async_collection", return_value=mock_collection):
            await mongodb_atlas._create_vector_search_index()
            mock_collection.create_search_index.assert_not_called()

    def test_create_vector_search_index_sync(self, mongodb_atlas):
        """Test _create_vector_search_index_sync."""
        mock_collection = MagicMock()
        mock_collection.list_search_indexes.return_value = []
        mock_collection.create_search_index = Mock()

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            mongodb_atlas._create_vector_search_index_sync()
            mock_collection.create_search_index.assert_called_once()


@pytest.mark.unit
class TestMongoDBAtlasOperations:
    @pytest.fixture
    def mongodb_atlas(self, mock_embedder):
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 1536)
        mock_embedder.embed_sync = Mock(return_value=[0.1] * 1536)
        with patch("hypertic.vectordb.mongovector.mongovector.getenv", return_value="mongodb://test"):
            return MongoDBAtlas(embedder=mock_embedder, collection="test_collection", connection_string="mongodb://test")

    @pytest.fixture
    def sample_documents(self):
        return [
            VectorDocument(
                id="doc1",
                content="Test content 1",
                metadata={"source": "test"},
                vector=[0.1] * 1536,
            ),
            VectorDocument(
                id="doc2",
                content="Test content 2",
                vector=[0.2] * 1536,
                metadata={"source": "test", "page": 1},
            ),
        ]

    def test_exists(self, mongodb_atlas):
        """Test exists method."""
        mock_collection = MagicMock()
        mock_database = MagicMock()
        mock_database.list_collection_names.return_value = ["test_collection", "other"]
        mock_collection.database = mock_database

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            with patch.object(mongodb_atlas, "_initialize_db_sync", return_value=True):
                mongodb_atlas._initialized = True
                assert mongodb_atlas.exists() is True

    def test_exists_false(self, mongodb_atlas):
        """Test exists returns False when collection doesn't exist."""
        mock_client = MagicMock()
        mock_database = MagicMock()
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None
        mock_database.__getitem__.return_value = mock_collection
        mock_client.__getitem__.return_value = mock_database

        with patch.object(mongodb_atlas, "_get_client", return_value=mock_client):
            assert mongodb_atlas.exists() is False

    @pytest.mark.asyncio
    async def test_add_documents_impl(self, mongodb_atlas, sample_documents):
        """Test _add_documents_impl."""
        mock_collection = MagicMock()
        mock_collection.replace_one = AsyncMock()

        async def get_async_collection():
            return mock_collection

        with patch.object(mongodb_atlas, "_get_async_collection", side_effect=get_async_collection):
            with patch.object(mongodb_atlas, "_create_vector_search_index", new_callable=AsyncMock):
                mongodb_atlas._initialized = True
                result = await mongodb_atlas._add_documents_impl(sample_documents)
                assert result is True
                assert mock_collection.replace_one.call_count == 2

    @pytest.mark.asyncio
    async def test_add_documents_impl_no_vectors(self, mongodb_atlas):
        """Test _add_documents_impl with documents without vectors."""
        documents = [
            VectorDocument(id="doc1", content="test", vector=None),
        ]
        mock_collection = MagicMock()

        async def get_async_collection():
            return mock_collection

        with patch.object(mongodb_atlas, "_get_async_collection", side_effect=get_async_collection):
            mongodb_atlas._initialized = True
            result = await mongodb_atlas._add_documents_impl(documents)
            assert result is True
            mock_collection.replace_one.assert_not_called()

    def test_add_documents_impl_sync(self, mongodb_atlas, sample_documents):
        """Test _add_documents_impl_sync."""
        mock_collection = MagicMock()
        mock_collection.replace_one = Mock()

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            with patch.object(mongodb_atlas, "_create_vector_search_index_sync", new_callable=Mock):
                mongodb_atlas._initialized = True
                result = mongodb_atlas._add_documents_impl_sync(sample_documents)
                assert result is True
                assert mock_collection.replace_one.call_count == 2

    @pytest.mark.asyncio
    async def test_search_impl(self, mongodb_atlas):
        """Test _search_impl."""
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list = AsyncMock(return_value=[{"_id": "doc1", "content": "test", "metadata": {}, "score": 0.95}])
        mock_collection.aggregate = AsyncMock(return_value=mock_cursor)

        async def get_async_collection():
            return mock_collection

        with patch.object(mongodb_atlas, "_get_async_collection", side_effect=get_async_collection):
            mongodb_atlas._initialized = True
            results = await mongodb_atlas._search_impl([0.1] * 1536, top_k=5)
            assert len(results) == 1
            assert results[0].content == "test"
            assert results[0].score == 0.95

    def test_search_impl_sync(self, mongodb_atlas):
        """Test _search_impl_sync."""
        mock_collection = MagicMock()
        mock_collection.aggregate.return_value = [{"_id": "doc1", "content": "test", "metadata": {}, "score": 0.95}]

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            mongodb_atlas._initialized = True
            results = mongodb_atlas._search_impl_sync([0.1] * 1536, top_k=5)
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_text_impl(self, mongodb_atlas):
        """Test _search_text_impl."""
        with patch.object(mongodb_atlas, "_search_impl", return_value=[]) as mock_search:
            mongodb_atlas._initialized = True
            results = await mongodb_atlas._search_text_impl("test query", top_k=5)
            assert results == []
            mock_search.assert_called_once()
            mongodb_atlas.embedder.embed.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_search_text_impl_no_embedder(self):
        """Test _search_text_impl without embedder."""
        with patch("hypertic.vectordb.mongovector.mongovector.getenv", return_value="mongodb://test"):
            mongodb_atlas = MongoDBAtlas(embedder=None, collection="test", connection_string="mongodb://test")
            result = await mongodb_atlas._search_text_impl("test", top_k=5)
            assert result == []

    def test_search_text_impl_sync(self, mongodb_atlas):
        """Test _search_text_impl_sync."""
        with patch.object(mongodb_atlas, "_search_impl_sync", return_value=[]) as mock_search:
            mongodb_atlas._initialized = True
            results = mongodb_atlas._search_text_impl_sync("test query", top_k=5)
            assert results == []
            mock_search.assert_called_once()
            mongodb_atlas.embedder.embed_sync.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_delete_impl_by_ids(self, mongodb_atlas):
        """Test _delete_impl by ids."""
        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.deleted_count = 1
        mock_collection.delete_many = Mock(return_value=mock_result)

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_result):
                mongodb_atlas._initialized = True
                result = await mongodb_atlas._delete_impl(ids=["id1", "id2"])
                assert result is True

    @pytest.mark.asyncio
    async def test_delete_impl_by_where(self, mongodb_atlas):
        """Test _delete_impl by where clause."""
        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.deleted_count = 1
        mock_collection.delete_many = Mock(return_value=mock_result)

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            with patch("asyncio.to_thread", new_callable=AsyncMock, return_value=mock_result):
                mongodb_atlas._initialized = True
                result = await mongodb_atlas._delete_impl(where={"source": "test"})
                assert result is True

    def test_delete_impl_sync_by_ids(self, mongodb_atlas):
        """Test _delete_impl_sync by ids."""
        mock_collection = MagicMock()
        mock_result = MagicMock()
        mock_result.deleted_count = 1
        mock_collection.delete_many = Mock(return_value=mock_result)

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            mongodb_atlas._initialized = True
            result = mongodb_atlas._delete_impl_sync(ids=["id1", "id2"])
            assert result is True
            mock_collection.delete_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_impl(self, mongodb_atlas):
        """Test _update_impl."""
        mock_collection = MagicMock()
        mock_collection.update_one = Mock()

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            with patch("asyncio.to_thread", new_callable=AsyncMock):
                mongodb_atlas._initialized = True
                result = await mongodb_atlas._update_impl(ids=["doc1"], documents=["new content"])
                assert result is True

    def test_update_impl_sync(self, mongodb_atlas):
        """Test _update_impl_sync."""
        mock_collection = MagicMock()
        mock_collection.update_one = Mock()

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            mongodb_atlas._initialized = True
            result = mongodb_atlas._update_impl_sync(ids=["doc1"], documents=["new content"])
            assert result is True
            mock_collection.update_one.assert_called()

    @pytest.mark.asyncio
    async def test_upsert_impl(self, mongodb_atlas):
        """Test _upsert_impl."""
        mock_collection = MagicMock()
        mock_collection.replace_one = Mock()

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            with patch("asyncio.to_thread", new_callable=AsyncMock):
                mongodb_atlas._initialized = True
                result = await mongodb_atlas._upsert_impl(ids=["doc1"], documents=["content"])
                assert result is True

    def test_upsert_impl_sync(self, mongodb_atlas):
        """Test _upsert_impl_sync."""
        mock_collection = MagicMock()
        mock_collection.replace_one = Mock()

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            mongodb_atlas._initialized = True
            result = mongodb_atlas._upsert_impl_sync(ids=["doc1"], documents=["content"])
            assert result is True
            mock_collection.replace_one.assert_called()

    @pytest.mark.asyncio
    async def test_get_documents_impl(self, mongodb_atlas):
        """Test _get_documents_impl."""
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list = Mock(return_value=[{"_id": "doc1", "content": "test", "metadata": {}}])
        mock_collection.find.return_value = mock_cursor

        async def get_async_collection():
            return mock_collection

        with patch.object(mongodb_atlas, "_get_async_collection", side_effect=get_async_collection):
            mongodb_atlas._initialized = True
            result = await mongodb_atlas._get_documents_impl(ids=["doc1"])
            assert "ids" in result
            assert "contents" in result
            assert "metadatas" in result

    def test_get_documents_impl_sync(self, mongodb_atlas):
        """Test _get_documents_impl_sync."""
        mock_collection = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.to_list.return_value = [{"_id": "doc1", "content": "test", "metadata": {}}]
        mock_collection.find.return_value = mock_cursor

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            mongodb_atlas._initialized = True
            result = mongodb_atlas._get_documents_impl_sync(ids=["doc1"])
            assert "ids" in result

    @pytest.mark.asyncio
    async def test_count_documents_impl(self, mongodb_atlas):
        """Test _count_documents_impl."""
        mock_collection = MagicMock()
        mock_collection.count_documents = AsyncMock(return_value=42)

        async def get_async_collection():
            return mock_collection

        with patch.object(mongodb_atlas, "_get_async_collection", side_effect=get_async_collection):
            mongodb_atlas._initialized = True
            count = await mongodb_atlas._count_documents_impl()
            assert count == 42

    def test_count_documents_impl_sync(self, mongodb_atlas):
        """Test _count_documents_impl_sync."""
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 42

        with patch.object(mongodb_atlas, "_get_collection", return_value=mock_collection):
            mongodb_atlas._initialized = True
            count = mongodb_atlas._count_documents_impl_sync()
            assert count == 42
