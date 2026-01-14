from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.vectordb.base import VectorDocument
from hypertic.vectordb.pinecone.pinecone import PineconeDB


@pytest.mark.unit
class TestPineconeDBBasics:
    @pytest.fixture
    def pinecone_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        with patch("hypertic.vectordb.pinecone.pinecone.Pinecone"):
            with patch("hypertic.vectordb.pinecone.pinecone.getenv", return_value="test_api_key"):
                return PineconeDB(embedder=mock_embedder, collection="test_collection", api_key="test_key")

    def test_pinecone_db_creation(self, pinecone_db):
        assert pinecone_db.collection == "test_collection"
        assert pinecone_db.collection_name == "test_collection"
        assert pinecone_db.embedder is not None

    def test_pinecone_db_with_api_key(self, mock_embedder):
        with patch("hypertic.vectordb.pinecone.pinecone.Pinecone"):
            db = PineconeDB(embedder=mock_embedder, collection="test", api_key="test_key")
            assert db.api_key == "test_key"

    @patch("hypertic.vectordb.pinecone.pinecone.getenv")
    def test_pinecone_db_with_env_api_key(self, mock_getenv, mock_embedder):
        mock_getenv.return_value = "env_api_key"
        with patch("hypertic.vectordb.pinecone.pinecone.Pinecone"):
            db = PineconeDB(embedder=mock_embedder, collection="test")
            assert db._get_api_key() == "env_api_key"

    @patch("hypertic.vectordb.pinecone.pinecone.getenv")
    def test_pinecone_db_no_api_key(self, mock_getenv, mock_embedder):
        mock_getenv.return_value = None
        with patch("hypertic.vectordb.pinecone.pinecone.Pinecone"):
            db = PineconeDB(embedder=mock_embedder, collection="test")
            with pytest.raises(ValueError, match="Pinecone API key is required"):
                db._get_api_key()

    def test_pinecone_db_with_dimension(self, mock_embedder):
        with patch("hypertic.vectordb.pinecone.pinecone.Pinecone"):
            with patch("hypertic.vectordb.pinecone.pinecone.getenv", return_value="test_key"):
                db = PineconeDB(embedder=mock_embedder, collection="test", dimension=256)
                assert db.dimension == 256

    def test_pinecone_db_with_metric(self, mock_embedder):
        with patch("hypertic.vectordb.pinecone.pinecone.Pinecone"):
            with patch("hypertic.vectordb.pinecone.pinecone.getenv", return_value="test_key"):
                db = PineconeDB(embedder=mock_embedder, collection="test", metric="euclidean")
                assert db.metric == "euclidean"

    def test_get_cloud_provider(self, pinecone_db):
        from pinecone import CloudProvider

        pinecone_db.cloud_provider = "aws"
        assert pinecone_db._get_cloud_provider() == CloudProvider.AWS

        pinecone_db.cloud_provider = "gcp"
        assert pinecone_db._get_cloud_provider() == CloudProvider.GCP

        pinecone_db.cloud_provider = "azure"
        assert pinecone_db._get_cloud_provider() == CloudProvider.AZURE

        pinecone_db.cloud_provider = "unknown"
        assert pinecone_db._get_cloud_provider() == CloudProvider.AWS  # Default

    def test_get_region(self, pinecone_db):
        from pinecone import AwsRegion

        pinecone_db.region = "us-east-1"
        assert pinecone_db._get_region() == AwsRegion.US_EAST_1

        pinecone_db.region = "us-west-2"
        assert pinecone_db._get_region() == AwsRegion.US_WEST_2

        pinecone_db.region = "eu-west-1"
        assert pinecone_db._get_region() == AwsRegion.EU_WEST_1

        pinecone_db.region = "unknown"
        assert pinecone_db._get_region() == AwsRegion.US_EAST_1  # Default

    def test_get_vector_type(self, pinecone_db):
        from pinecone import VectorType

        pinecone_db.vector_type = "dense"
        assert pinecone_db._get_vector_type() == VectorType.DENSE

        pinecone_db.vector_type = "sparse"
        assert pinecone_db._get_vector_type() == VectorType.SPARSE

        pinecone_db.vector_type = "unknown"
        assert pinecone_db._get_vector_type() == VectorType.DENSE  # Default


@pytest.mark.unit
class TestPineconeDBFlattenMetadata:
    @pytest.fixture
    def pinecone_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        with patch("hypertic.vectordb.pinecone.pinecone.Pinecone"):
            with patch("hypertic.vectordb.pinecone.pinecone.getenv", return_value="test_key"):
                return PineconeDB(embedder=mock_embedder, collection="test_collection", api_key="test_key")

    def test_flatten_simple_metadata(self, pinecone_db):
        metadata = {"key1": "value1", "key2": 123}
        result = pinecone_db._flatten_metadata(metadata)
        assert result == {"key1": "value1", "key2": 123}

    def test_flatten_nested_metadata(self, pinecone_db):
        metadata = {"key1": {"nested": "value"}, "key2": 123}
        result = pinecone_db._flatten_metadata(metadata)
        assert result == {"key1.nested": "value", "key2": 123}

    def test_flatten_list_metadata(self, pinecone_db):
        import json

        metadata = {"tags": ["tag1", "tag2"]}
        result = pinecone_db._flatten_metadata(metadata)
        assert result == {"tags": json.dumps(["tag1", "tag2"])}

    def test_flatten_non_dict_metadata(self, pinecone_db):
        result = pinecone_db._flatten_metadata("string_value")
        assert result == {"value": "string_value"}

        result = pinecone_db._flatten_metadata(123)
        assert result == {"value": 123}

        result = pinecone_db._flatten_metadata(None)
        assert result == {}


@pytest.mark.unit
class TestPineconeDBClientOperations:
    @pytest.fixture
    def pinecone_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        with patch("hypertic.vectordb.pinecone.pinecone.Pinecone"):
            with patch("hypertic.vectordb.pinecone.pinecone.getenv", return_value="test_key"):
                return PineconeDB(embedder=mock_embedder, collection="test_collection", api_key="test_key")

    def test_get_client(self, pinecone_db):
        """Test _get_client creates Pinecone client."""
        mock_pinecone = MagicMock()
        mock_client = MagicMock()
        mock_pinecone.return_value = mock_client

        with patch("hypertic.vectordb.pinecone.pinecone.Pinecone", mock_pinecone):
            pinecone_db._client = None
            client = pinecone_db._get_client()
            assert client == mock_client
            mock_pinecone.assert_called_once_with(api_key="test_key")

    def test_get_index(self, pinecone_db):
        """Test _get_index creates Index."""
        mock_client = MagicMock()
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index

        with patch.object(pinecone_db, "_get_client", return_value=mock_client):
            pinecone_db._index = None
            index = pinecone_db._get_index()
            assert index == mock_index
            mock_client.Index.assert_called_once_with("test_collection")


@pytest.mark.unit
class TestPineconeDBInitialization:
    @pytest.fixture
    def pinecone_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        with patch("hypertic.vectordb.pinecone.pinecone.Pinecone"):
            with patch("hypertic.vectordb.pinecone.pinecone.getenv", return_value="test_key"):
                return PineconeDB(embedder=mock_embedder, collection="test_collection", api_key="test_key")

    @pytest.mark.asyncio
    async def test_initialize_db_already_initialized(self, pinecone_db):
        """Test _initialize_db when already initialized."""
        pinecone_db._initialized = True
        result = await pinecone_db._initialize_db()
        assert result is True

    @pytest.mark.asyncio
    async def test_initialize_db_index_exists(self, pinecone_db):
        """Test _initialize_db when index already exists."""
        mock_client = MagicMock()
        mock_index_obj = MagicMock()
        mock_index_obj.name = "test_collection"
        mock_client.list_indexes.return_value = [mock_index_obj]
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index

        with patch.object(pinecone_db, "_get_client", return_value=mock_client):
            result = await pinecone_db._initialize_db()
            assert result is True
            assert pinecone_db._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_db_create_new_index(self, pinecone_db):
        """Test _initialize_db creates new index."""
        mock_client = MagicMock()
        mock_client.list_indexes.return_value = []
        mock_index = MagicMock()
        mock_client.Index.return_value = mock_index

        with patch.object(pinecone_db, "_get_client", return_value=mock_client):
            with patch("hypertic.vectordb.pinecone.pinecone.ServerlessSpec"):
                with patch("hypertic.vectordb.pinecone.pinecone.VectorType"):
                    pinecone_db.dimension = 384
                    result = await pinecone_db._initialize_db()
                    assert result is True
                    mock_client.create_index.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_db_error(self, pinecone_db):
        """Test _initialize_db handles errors."""
        mock_client = MagicMock()
        mock_client.list_indexes.side_effect = Exception("Error")

        with patch.object(pinecone_db, "_get_client", return_value=mock_client):
            result = await pinecone_db._initialize_db()
            assert result is False

    def test_initialize_db_sync_already_initialized(self, pinecone_db):
        """Test _initialize_db_sync when already initialized."""
        pinecone_db._initialized = True
        result = pinecone_db._initialize_db_sync()
        assert result is True

    def test_initialize_db_sync_index_exists(self, pinecone_db):
        """Test _initialize_db_sync when index exists."""
        mock_client = MagicMock()
        mock_client.has_index.return_value = True

        with patch.object(pinecone_db, "_get_client", return_value=mock_client):
            result = pinecone_db._initialize_db_sync()
            assert result is True

    def test_initialize_db_sync_create_new_index(self, pinecone_db):
        """Test _initialize_db_sync creates new index."""
        mock_client = MagicMock()
        mock_client.has_index.return_value = False
        mock_index_status = MagicMock()
        mock_index_status.status = {"ready": True}
        mock_client.describe_index.return_value = mock_index_status

        with patch.object(pinecone_db, "_get_client", return_value=mock_client):
            with patch("hypertic.vectordb.pinecone.pinecone.ServerlessSpec"):
                pinecone_db.dimension = 384
                result = pinecone_db._initialize_db_sync()
                assert result is True
                mock_client.create_index.assert_called_once()


@pytest.mark.unit
class TestPineconeDBOperations:
    @pytest.fixture
    def pinecone_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)
        mock_embedder.embed_sync = Mock(return_value=[0.1] * 384)
        with patch("hypertic.vectordb.pinecone.pinecone.Pinecone"):
            with patch("hypertic.vectordb.pinecone.pinecone.getenv", return_value="test_key"):
                return PineconeDB(embedder=mock_embedder, collection="test_collection", api_key="test_key")

    @pytest.fixture
    def sample_documents(self):
        return [
            VectorDocument(
                id="doc1",
                content="Test content 1",
                metadata={"source": "test"},
                vector=[0.1] * 384,
            ),
            VectorDocument(
                id="doc2",
                content="Test content 2",
                vector=[0.2] * 384,
                metadata={"source": "test", "page": 1},
            ),
        ]

    def test_exists(self, pinecone_db):
        """Test exists method."""
        mock_client = MagicMock()
        mock_index_obj = MagicMock()
        mock_index_obj.name = "test_collection"
        mock_client.list_indexes.return_value = [mock_index_obj]

        with patch.object(pinecone_db, "_get_client", return_value=mock_client):
            assert pinecone_db.exists() is True

    def test_exists_false(self, pinecone_db):
        """Test exists returns False when index doesn't exist."""
        mock_client = MagicMock()
        mock_index_obj = MagicMock()
        mock_index_obj.name = "other_collection"
        mock_client.list_indexes.return_value = [mock_index_obj]

        with patch.object(pinecone_db, "_get_client", return_value=mock_client):
            assert pinecone_db.exists() is False

    @pytest.mark.asyncio
    async def test_async_exists(self, pinecone_db):
        """Test async_exists method."""
        mock_client = MagicMock()
        mock_index_obj = MagicMock()
        mock_index_obj.name = "test_collection"
        mock_client.list_indexes.return_value = [mock_index_obj]

        with patch.object(pinecone_db, "_get_client", return_value=mock_client):
            assert await pinecone_db.async_exists() is True

    @pytest.mark.asyncio
    async def test_add_documents_impl(self, pinecone_db, sample_documents):
        """Test _add_documents_impl."""
        mock_index = MagicMock()
        mock_index.upsert = Mock()

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            result = await pinecone_db._add_documents_impl(sample_documents)
            assert result is True
            mock_index.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_documents_impl_no_vectors(self, pinecone_db):
        """Test _add_documents_impl with documents without vectors."""
        documents = [
            VectorDocument(id="doc1", content="test", vector=None),
        ]
        mock_index = MagicMock()

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            result = await pinecone_db._add_documents_impl(documents)
            assert result is False
            mock_index.upsert.assert_not_called()

    def test_add_documents_impl_sync(self, pinecone_db, sample_documents):
        """Test _add_documents_impl_sync."""
        mock_index = MagicMock()
        mock_index.upsert = Mock()

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            result = pinecone_db._add_documents_impl_sync(sample_documents)
            assert result is True
            mock_index.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_impl(self, pinecone_db):
        """Test _search_impl."""
        mock_index = MagicMock()
        mock_match = MagicMock()
        mock_match.metadata = {"content": "test content", "meta": "value"}
        mock_match.score = 0.95
        mock_response = MagicMock()
        mock_response.matches = [mock_match]
        mock_index.query.return_value = mock_response

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            results = await pinecone_db._search_impl([0.1] * 384, top_k=5)
            assert len(results) == 1
            assert results[0].content == "test content"
            assert results[0].score == 0.95

    @pytest.mark.asyncio
    async def test_search_impl_with_filters(self, pinecone_db):
        """Test _search_impl with filters."""
        mock_index = MagicMock()
        mock_response = MagicMock()
        mock_response.matches = []
        mock_index.query.return_value = mock_response

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            results = await pinecone_db._search_impl([0.1] * 384, top_k=5, filters={"source": "test"})
            assert isinstance(results, list)
            mock_index.query.assert_called_once()
            call_kwargs = mock_index.query.call_args[1]
            assert "filter" in call_kwargs

    def test_search_impl_sync(self, pinecone_db):
        """Test _search_impl_sync."""
        mock_index = MagicMock()
        mock_match = MagicMock()
        mock_match.metadata = {"content": "test content"}
        mock_match.score = 0.95
        mock_response = MagicMock()
        mock_response.matches = [mock_match]
        mock_index.query.return_value = mock_response

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            results = pinecone_db._search_impl_sync([0.1] * 384, top_k=5)
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_text_impl(self, pinecone_db):
        """Test _search_text_impl."""
        mock_index = MagicMock()
        mock_response = MagicMock()
        mock_response.matches = []
        mock_index.query.return_value = mock_response

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            with patch.object(pinecone_db, "_search_impl", return_value=[]) as mock_search:
                pinecone_db._initialized = True
                results = await pinecone_db._search_text_impl("test query", top_k=5)
                assert results == []
                mock_search.assert_called_once()
                pinecone_db.embedder.embed.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_search_text_impl_no_embedder(self):
        """Test _search_text_impl without embedder."""
        with patch("hypertic.vectordb.pinecone.pinecone.Pinecone"):
            with patch("hypertic.vectordb.pinecone.pinecone.getenv", return_value="test_key"):
                pinecone_db = PineconeDB(embedder=None, collection="test", api_key="test_key")
                result = await pinecone_db._search_text_impl("test", top_k=5)
                assert result == []

    def test_search_text_impl_sync(self, pinecone_db):
        """Test _search_text_impl_sync."""
        with patch.object(pinecone_db, "_search_impl_sync", return_value=[]) as mock_search:
            pinecone_db._initialized = True
            results = pinecone_db._search_text_impl_sync("test query", top_k=5)
            assert results == []
            mock_search.assert_called_once()
            pinecone_db.embedder.embed_sync.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_delete_impl_by_ids(self, pinecone_db):
        """Test _delete_impl by ids."""
        mock_index = MagicMock()
        mock_index.delete = AsyncMock()

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            result = await pinecone_db._delete_impl(ids=["id1", "id2"])
            assert result is True
            mock_index.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_impl_by_where(self, pinecone_db):
        """Test _delete_impl by where clause."""
        mock_index = MagicMock()
        mock_index.delete = AsyncMock()

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            result = await pinecone_db._delete_impl(where={"source": "test"})
            assert result is True
            mock_index.delete.assert_called_once()

    def test_delete_impl_sync_by_ids(self, pinecone_db):
        """Test _delete_impl_sync by ids."""
        mock_index = MagicMock()
        mock_index.delete = Mock()

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            result = pinecone_db._delete_impl_sync(ids=["id1", "id2"])
            assert result is True
            mock_index.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_impl(self, pinecone_db):
        """Test _update_impl."""
        mock_index = MagicMock()
        mock_index.update = Mock()
        mock_vector_data = MagicMock()
        mock_vector_data.values = [0.1] * 384
        mock_vector_data.metadata = {}
        mock_fetch_response = MagicMock()
        mock_fetch_response.vectors = {"doc1": mock_vector_data}
        # fetch is sync in Pinecone, not async
        mock_index.fetch = Mock(return_value=mock_fetch_response)
        mock_index.upsert = Mock()  # upsert is sync, not async

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            result = await pinecone_db._update_impl(ids=["doc1"], documents=["new content"])
            assert result is True
            mock_index.upsert.assert_called_once()

    def test_update_impl_sync(self, pinecone_db):
        """Test _update_impl_sync."""
        mock_index = MagicMock()
        mock_index.update = Mock()
        mock_index.fetch = Mock()
        mock_vector_data = MagicMock()
        mock_vector_data.values = [0.1] * 384
        mock_vector_data.metadata = {}
        mock_fetch_response = MagicMock()
        mock_fetch_response.vectors = {"doc1": mock_vector_data}
        mock_index.fetch.return_value = mock_fetch_response
        mock_index.upsert = Mock()

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            result = pinecone_db._update_impl_sync(ids=["doc1"], documents=["new content"])
            assert result is True
            mock_index.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_upsert_impl(self, pinecone_db):
        """Test _upsert_impl."""
        mock_index = MagicMock()
        mock_index.upsert = AsyncMock()

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            result = await pinecone_db._upsert_impl(ids=["doc1"], documents=["content"])
            assert result is True
            mock_index.upsert.assert_called_once()

    def test_upsert_impl_sync(self, pinecone_db):
        """Test _upsert_impl_sync."""
        mock_index = MagicMock()
        mock_index.upsert = Mock()

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            result = pinecone_db._upsert_impl_sync(ids=["doc1"], documents=["content"])
            assert result is True
            mock_index.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_documents_impl(self, pinecone_db):
        """Test _get_documents_impl."""
        mock_index = MagicMock()
        mock_index.fetch = AsyncMock()
        mock_vector_data = MagicMock()
        mock_vector_data.metadata = {"content": "test content"}
        mock_fetch_response = MagicMock()
        mock_fetch_response.vectors = {"doc1": mock_vector_data}
        mock_index.fetch.return_value = mock_fetch_response

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            pinecone_db.dimension = 384
            result = await pinecone_db._get_documents_impl(ids=["doc1"])
            assert "ids" in result
            assert "contents" in result
            assert "metadatas" in result

    def test_get_documents_impl_sync(self, pinecone_db):
        """Test _get_documents_impl_sync."""
        mock_index = MagicMock()
        mock_match = MagicMock()
        mock_match.metadata = {"content": "test content"}
        mock_response = MagicMock()
        mock_response.matches = [mock_match]
        mock_index.query.return_value = mock_response

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            result = pinecone_db._get_documents_impl_sync(ids=["doc1"])
            assert "ids" in result

    @pytest.mark.asyncio
    async def test_count_documents_impl(self, pinecone_db):
        """Test _count_documents_impl."""
        mock_index = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_vector_count = 42
        mock_index.describe_index_stats.return_value = mock_stats

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            count = await pinecone_db._count_documents_impl()
            assert count == 42

    def test_count_documents_impl_sync(self, pinecone_db):
        """Test _count_documents_impl_sync."""
        mock_index = MagicMock()
        mock_stats = MagicMock()
        mock_stats.total_vector_count = 42
        mock_index.describe_index_stats.return_value = mock_stats

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            count = pinecone_db._count_documents_impl_sync()
            assert count == 42

    @pytest.mark.asyncio
    async def test_search_impl_unexpected_response(self, pinecone_db):
        """Test _search_impl handles unexpected response type."""
        mock_index = MagicMock()
        mock_response = MagicMock()
        # Response without matches attribute
        del mock_response.matches
        mock_index.query.return_value = mock_response

        with patch.object(pinecone_db, "_get_index", return_value=mock_index):
            pinecone_db._initialized = True
            results = await pinecone_db._search_impl([0.1] * 384, top_k=5)
            assert results == []
