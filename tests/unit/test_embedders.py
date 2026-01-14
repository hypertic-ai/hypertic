from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.vectordb.base import VectorDocument
from hypertic.vectordb.qdrant.qdrant import QdrantDB


@pytest.mark.unit
class TestQdrantDBBasics:
    @pytest.fixture
    def qdrant_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        return QdrantDB(embedder=mock_embedder, collection="test_collection")

    def test_qdrant_db_creation(self, qdrant_db):
        assert qdrant_db.collection == "test_collection"
        assert qdrant_db.collection_name == "test_collection"
        assert qdrant_db.embedder is not None

    def test_qdrant_db_with_path(self, mock_embedder):
        db = QdrantDB(embedder=mock_embedder, collection="test", path="/tmp/test")
        assert db.path == "/tmp/test"

    def test_qdrant_db_with_url(self, mock_embedder):
        db = QdrantDB(embedder=mock_embedder, collection="test", url="http://localhost:6333")
        assert db.url == "http://localhost:6333"

    def test_qdrant_db_with_host_port(self, mock_embedder):
        db = QdrantDB(embedder=mock_embedder, collection="test", host="localhost", port=6333)
        assert db.host == "localhost"
        assert db.port == 6333

    def test_qdrant_db_distance_metrics(self, qdrant_db):
        from qdrant_client.models import Distance

        qdrant_db.distance_metric = "Cosine"
        assert qdrant_db._get_distance_metric() == Distance.COSINE

        qdrant_db.distance_metric = "Dot"
        assert qdrant_db._get_distance_metric() == Distance.DOT

        qdrant_db.distance_metric = "Euclidean"
        assert qdrant_db._get_distance_metric() == Distance.EUCLID

        qdrant_db.distance_metric = "Manhattan"
        assert qdrant_db._get_distance_metric() == Distance.MANHATTAN

        qdrant_db.distance_metric = "Unknown"
        assert qdrant_db._get_distance_metric() == Distance.COSINE  # Default

    def test_get_cloud_config_with_api_key(self, qdrant_db):
        qdrant_db.api_key = "test_key"
        assert qdrant_db._get_cloud_config() == "test_key"

    @patch("hypertic.vectordb.qdrant.qdrant.getenv")
    def test_get_cloud_config_from_env(self, mock_getenv, qdrant_db):
        qdrant_db.api_key = None
        mock_getenv.return_value = "env_key"
        assert qdrant_db._get_cloud_config() == "env_key"

    @patch("hypertic.vectordb.qdrant.qdrant.getenv")
    def test_get_cloud_config_none(self, mock_getenv, qdrant_db):
        qdrant_db.api_key = None
        mock_getenv.return_value = None
        assert qdrant_db._get_cloud_config() is None


@pytest.mark.unit
class TestQdrantDBClientCreation:
    @pytest.fixture
    def qdrant_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        return QdrantDB(embedder=mock_embedder, collection="test_collection")

    @patch("hypertic.vectordb.qdrant.qdrant.QdrantClient")
    def test_get_client_with_api_key_and_url(self, mock_client_class, qdrant_db):
        qdrant_db.api_key = "test_key"
        qdrant_db.url = "http://localhost:6333"
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        client = qdrant_db._get_client()
        assert client == mock_client
        mock_client_class.assert_called_once_with(url="http://localhost:6333", api_key="test_key", timeout=None)

    @patch("hypertic.vectordb.qdrant.qdrant.QdrantClient")
    def test_get_client_with_path(self, mock_client_class, qdrant_db):
        qdrant_db.path = "/tmp/test"
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        client = qdrant_db._get_client()
        assert client == mock_client
        mock_client_class.assert_called_once_with(path="/tmp/test")

    @patch("hypertic.vectordb.qdrant.qdrant.QdrantClient")
    def test_get_client_in_memory(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        client = qdrant_db._get_client()
        assert client == mock_client
        mock_client_class.assert_called_once_with(":memory:")

    @patch("hypertic.vectordb.qdrant.qdrant.QdrantClient")
    def test_get_client_with_host_port(self, mock_client_class, qdrant_db):
        qdrant_db.host = "localhost"
        qdrant_db.port = 6333
        qdrant_db.grpc_port = 6334
        qdrant_db.prefer_grpc = True
        qdrant_db.timeout = 30
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        client = qdrant_db._get_client()
        assert client == mock_client
        mock_client_class.assert_called_once_with(host="localhost", port=6333, grpc_port=6334, prefer_grpc=True, timeout=30)

    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    @pytest.mark.asyncio
    async def test_get_async_client_with_api_key_and_url(self, mock_client_class, qdrant_db):
        qdrant_db.api_key = "test_key"
        qdrant_db.url = "http://localhost:6333"
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        client = await qdrant_db._get_async_client()
        assert client == mock_client
        mock_client_class.assert_called_once_with(url="http://localhost:6333", api_key="test_key", timeout=None)

    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    @pytest.mark.asyncio
    async def test_get_async_client_reuses_instance(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        client1 = await qdrant_db._get_async_client()
        client2 = await qdrant_db._get_async_client()
        assert client1 == client2
        assert mock_client_class.call_count == 1


@pytest.mark.unit
class TestQdrantDBFlattenMetadata:
    @pytest.fixture
    def qdrant_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        return QdrantDB(embedder=mock_embedder, collection="test_collection")

    def test_flatten_simple_metadata(self, qdrant_db):
        metadata = {"key1": "value1", "key2": 123}
        result = qdrant_db._flatten_metadata(metadata)
        assert result == {"key1": "value1", "key2": 123}

    def test_flatten_nested_metadata(self, qdrant_db):
        metadata = {"key1": {"nested": "value"}, "key2": 123}
        result = qdrant_db._flatten_metadata(metadata)
        assert result == {"key1.nested": "value", "key2": 123}

    def test_flatten_deeply_nested_metadata(self, qdrant_db):
        metadata = {"level1": {"level2": {"level3": "value"}}}
        result = qdrant_db._flatten_metadata(metadata)
        assert result == {"level1.level2.level3": "value"}

    def test_flatten_list_metadata(self, qdrant_db):
        import json

        metadata = {"tags": ["tag1", "tag2"]}
        result = qdrant_db._flatten_metadata(metadata)
        assert result == {"tags": json.dumps(["tag1", "tag2"])}

    def test_flatten_empty_dict(self, qdrant_db):
        import json

        metadata = {"empty": {}}
        result = qdrant_db._flatten_metadata(metadata)
        assert result == {"empty": json.dumps({})}

    def test_flatten_with_none(self, qdrant_db):
        metadata = {"key1": "value1", "key2": None}
        result = qdrant_db._flatten_metadata(metadata)
        assert result == {"key1": "value1"}

    def test_flatten_complex_object(self, qdrant_db):
        class CustomObject:
            def __str__(self):
                return "custom"

        metadata = {"obj": CustomObject()}
        result = qdrant_db._flatten_metadata(metadata)
        assert result == {"obj": "custom"}


@pytest.mark.unit
class TestQdrantDBCreateFilter:
    @pytest.fixture
    def qdrant_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        return QdrantDB(embedder=mock_embedder, collection="test_collection")

    def test_create_filter_empty(self, qdrant_db):
        result = qdrant_db._create_filter({})
        assert result is None

    def test_create_filter_simple_match(self, qdrant_db):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        where = {"key": "value"}
        result = qdrant_db._create_filter(where)
        assert isinstance(result, Filter)
        assert len(result.must) == 1
        assert isinstance(result.must[0], FieldCondition)
        assert result.must[0].key == "key"
        assert isinstance(result.must[0].match, MatchValue)
        assert result.must[0].match.value == "value"

    def test_create_filter_range_gte(self, qdrant_db):
        from qdrant_client.models import FieldCondition, Filter, Range

        where = {"age": {"gte": 18}}
        result = qdrant_db._create_filter(where)
        assert isinstance(result, Filter)
        assert isinstance(result.must[0], FieldCondition)
        assert isinstance(result.must[0].range, Range)
        assert result.must[0].range.gte == 18

    def test_create_filter_range_multiple(self, qdrant_db):
        where = {"age": {"gte": 18, "lte": 65}}
        result = qdrant_db._create_filter(where)
        assert result.must[0].range.gte == 18
        assert result.must[0].range.lte == 65

    def test_create_filter_in(self, qdrant_db):
        from qdrant_client.models import MatchAny

        where = {"status": {"in": ["active", "pending"]}}
        result = qdrant_db._create_filter(where)
        assert isinstance(result.must[0].match, MatchAny)
        assert set(result.must[0].match.any) == {"active", "pending"}

    def test_create_filter_in_single_value(self, qdrant_db):
        from qdrant_client.models import MatchAny

        where = {"status": {"in": "active"}}
        result = qdrant_db._create_filter(where)
        assert isinstance(result.must[0].match, MatchAny)
        assert result.must[0].match.any == ["active"]

    def test_create_filter_nin(self, qdrant_db):
        from qdrant_client.models import MatchValue

        where = {"status": {"nin": ["deleted", "archived"]}}
        result = qdrant_db._create_filter(where)
        assert len(result.must) == 2
        assert all(isinstance(cond.match, MatchValue) for cond in result.must)

    def test_create_filter_multiple_conditions(self, qdrant_db):
        where = {"status": "active", "age": {"gte": 18}}
        result = qdrant_db._create_filter(where)
        assert len(result.must) == 2


@pytest.mark.unit
class TestQdrantDBExists:
    @pytest.fixture
    def qdrant_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        return QdrantDB(embedder=mock_embedder, collection="test_collection")

    @patch("hypertic.vectordb.qdrant.qdrant.QdrantClient")
    def test_exists_true(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client_class.return_value = mock_client

        assert qdrant_db.exists() is True

    @patch("hypertic.vectordb.qdrant.qdrant.QdrantClient")
    def test_exists_false(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client_class.return_value = mock_client

        assert qdrant_db.exists() is False

    @patch("hypertic.vectordb.qdrant.qdrant.QdrantClient")
    def test_exists_exception(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client.collection_exists.side_effect = Exception("Error")
        mock_client_class.return_value = mock_client

        assert qdrant_db.exists() is False

    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    @pytest.mark.asyncio
    async def test_async_exists_true(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client_class.return_value = mock_client

        assert await qdrant_db.async_exists() is True

    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    @pytest.mark.asyncio
    async def test_async_exists_false(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        assert await qdrant_db.async_exists() is False


@pytest.mark.unit
class TestQdrantDBFormatSearchResults:
    @pytest.fixture
    def qdrant_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        return QdrantDB(embedder=mock_embedder, collection="test_collection")

    def test_format_search_results(self, qdrant_db):
        from qdrant_client.models import ScoredPoint

        from hypertic.vectordb.base import VectorSearchResult

        mock_point1 = MagicMock(spec=ScoredPoint)
        mock_point1.payload = {"content": "text1", "meta": "value1"}
        mock_point1.score = 0.95

        mock_point2 = MagicMock(spec=ScoredPoint)
        mock_point2.payload = {"content": "text2"}
        mock_point2.score = 0.85

        results = qdrant_db._format_search_results([mock_point1, mock_point2])
        assert len(results) == 2
        assert isinstance(results[0], VectorSearchResult)
        assert results[0].content == "text1"
        assert results[0].metadata == {"meta": "value1"}
        assert results[0].score == 0.95
        assert results[1].content == "text2"
        assert results[1].score == 0.85

    def test_format_search_results_empty_payload(self, qdrant_db):
        from qdrant_client.models import ScoredPoint

        mock_point = MagicMock(spec=ScoredPoint)
        mock_point.payload = None
        mock_point.score = 0.9

        results = qdrant_db._format_search_results([mock_point])
        assert results[0].content == ""
        assert results[0].metadata == {}


@pytest.mark.unit
class TestQdrantDBOperations:
    @pytest.fixture
    def qdrant_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)
        mock_embedder.embed_sync = Mock(return_value=[0.1] * 384)
        return QdrantDB(embedder=mock_embedder, collection="test_collection")

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
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_add_documents_impl_success(self, mock_client_class, qdrant_db, sample_documents):
        mock_client = MagicMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client.retrieve = AsyncMock(return_value=[])
        mock_client.upsert = AsyncMock()
        mock_client_class.return_value = mock_client

        result = await qdrant_db._add_documents_impl(sample_documents)
        assert result is True
        assert mock_client.upsert.called

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_add_documents_impl_with_existing(self, mock_client_class, qdrant_db, sample_documents):
        mock_client = MagicMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_existing_point = MagicMock()
        mock_client.retrieve = AsyncMock(side_effect=[[mock_existing_point], []])
        mock_client.upsert = AsyncMock()
        mock_client_class.return_value = mock_client

        result = await qdrant_db._add_documents_impl(sample_documents)
        assert result is True
        assert mock_client.upsert.called

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_add_documents_impl_without_embedder(self, mock_client_class, sample_documents):
        qdrant_db = QdrantDB(embedder=None, collection="test", vector_size=384)
        mock_client = MagicMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client.retrieve = AsyncMock(return_value=[])
        mock_client.upsert = AsyncMock()
        mock_client_class.return_value = mock_client

        result = await qdrant_db._add_documents_impl(sample_documents)
        assert result is True

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_add_documents_impl_error(self, mock_client_class, qdrant_db, sample_documents):
        mock_client = MagicMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client.retrieve = AsyncMock(side_effect=Exception("Error"))
        mock_client_class.return_value = mock_client

        result = await qdrant_db._add_documents_impl(sample_documents)
        assert result is False

    @patch("hypertic.vectordb.qdrant.qdrant.QdrantClient")
    def test_add_documents_impl_sync_success(self, mock_client_class, qdrant_db, sample_documents):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client.retrieve.return_value = []
        mock_client.upsert = Mock()
        mock_client_class.return_value = mock_client

        result = qdrant_db._add_documents_impl_sync(sample_documents)
        assert result is True
        assert mock_client.upsert.called

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_search_impl(self, mock_client_class, qdrant_db):
        from qdrant_client.models import ScoredPoint

        mock_client = MagicMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_point = MagicMock(spec=ScoredPoint)
        mock_point.payload = {"content": "test"}
        mock_point.score = 0.95
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        results = await qdrant_db._search_impl([0.1] * 384, top_k=5)
        assert len(results) == 1
        assert results[0].content == "test"
        assert results[0].score == 0.95

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_search_text_impl(self, mock_client_class, qdrant_db):
        from qdrant_client.models import ScoredPoint

        mock_client = MagicMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_point = MagicMock(spec=ScoredPoint)
        mock_point.payload = {"content": "test"}
        mock_point.score = 0.95
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        results = await qdrant_db._search_text_impl("test query", top_k=5)
        assert len(results) == 1
        qdrant_db.embedder.embed.assert_called_once_with("test query")

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_search_text_impl_no_embedder(self, mock_client_class):
        qdrant_db = QdrantDB(embedder=None, collection="test")
        mock_client = MagicMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_client_class.return_value = mock_client

        # The method catches the ValueError and returns empty list
        result = await qdrant_db._search_text_impl("test", top_k=5)
        assert result == []

    @patch("hypertic.vectordb.qdrant.qdrant.QdrantClient")
    def test_search_impl_sync(self, mock_client_class, qdrant_db):
        from qdrant_client.models import ScoredPoint

        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_point = MagicMock(spec=ScoredPoint)
        mock_point.payload = {"content": "test"}
        mock_point.score = 0.95
        mock_response = MagicMock()
        mock_response.points = [mock_point]
        mock_client.query_points.return_value = mock_response
        mock_client_class.return_value = mock_client

        results = qdrant_db._search_impl_sync([0.1] * 384, top_k=5)
        assert len(results) == 1

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_delete_impl_by_ids(self, mock_client_class, qdrant_db):
        from qdrant_client.models import PointIdsList

        mock_client = MagicMock()
        mock_client.delete = AsyncMock()
        mock_client_class.return_value = mock_client
        qdrant_db.initialized = True

        result = await qdrant_db._delete_impl(ids=["id1", "id2"])
        assert result is True
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert isinstance(call_args[1]["points_selector"], PointIdsList)

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_delete_impl_by_where(self, mock_client_class, qdrant_db):
        from qdrant_client.models import FilterSelector

        mock_client = MagicMock()
        mock_client.delete = AsyncMock()
        mock_client_class.return_value = mock_client
        qdrant_db.initialized = True

        result = await qdrant_db._delete_impl(where={"status": "deleted"})
        assert result is True
        call_args = mock_client.delete.call_args
        assert isinstance(call_args[1]["points_selector"], FilterSelector)

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_delete_impl_all(self, mock_client_class, qdrant_db):
        from qdrant_client.models import Filter, FilterSelector

        mock_client = MagicMock()
        mock_client.delete = AsyncMock()
        mock_client_class.return_value = mock_client
        qdrant_db.initialized = True

        result = await qdrant_db._delete_impl()
        assert result is True
        call_args = mock_client.delete.call_args
        assert isinstance(call_args[1]["points_selector"], FilterSelector)
        assert isinstance(call_args[1]["points_selector"].filter, Filter)

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_delete_impl_empty_where(self, mock_client_class, qdrant_db):
        from qdrant_client.models import Filter, FilterSelector

        mock_client = MagicMock()
        mock_client.delete = AsyncMock()
        mock_client_class.return_value = mock_client
        qdrant_db.initialized = True

        # Empty dict {} is falsy, so elif where: is False
        # Falls through to else: branch which deletes all
        result = await qdrant_db._delete_impl(where={})
        assert result is True
        # Should call delete with empty filter (delete all)
        call_args = mock_client.delete.call_args
        assert isinstance(call_args[1]["points_selector"], FilterSelector)
        assert isinstance(call_args[1]["points_selector"].filter, Filter)

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_update_impl(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_existing_point = MagicMock()
        mock_existing_point.payload = {"content": "old"}
        mock_existing_point.vector = [0.1] * 384
        mock_client.retrieve = AsyncMock(return_value=[mock_existing_point])
        mock_client.upsert = AsyncMock()
        mock_client_class.return_value = mock_client
        qdrant_db.initialized = True

        result = await qdrant_db._update_impl(ids=["doc1"], documents=["new content"], metadatas=[{"updated": True}])
        assert result is True
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_update_impl_not_found(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client.retrieve = AsyncMock(return_value=[])
        mock_client.upsert = AsyncMock()
        mock_client_class.return_value = mock_client
        qdrant_db.initialized = True

        result = await qdrant_db._update_impl(ids=["doc1"], documents=["new"])
        assert result is True
        mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_upsert_impl(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client.upsert = AsyncMock()
        mock_client_class.return_value = mock_client
        qdrant_db.initialized = True

        result = await qdrant_db._upsert_impl(ids=["doc1"], documents=["content"], metadatas=[{"meta": "value"}])
        assert result is True
        mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_get_documents_impl_by_ids(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_point = MagicMock()
        mock_point.id = "doc1"
        mock_point.payload = {"content": "test"}
        mock_point.vector = [0.1] * 384
        mock_client.retrieve = AsyncMock(return_value=[mock_point])
        mock_client_class.return_value = mock_client

        result = await qdrant_db._get_documents_impl(ids=["doc1"])
        assert "ids" in result
        assert "documents" in result
        assert "metadatas" in result
        assert "embeddings" in result
        assert result["ids"] == ["doc1"]

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_get_documents_impl_scroll(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_point = MagicMock()
        mock_point.id = "doc1"
        mock_point.payload = {"content": "test"}
        mock_point.vector = [0.1] * 384
        mock_client.scroll = AsyncMock(return_value=([mock_point], None))
        mock_client_class.return_value = mock_client

        result = await qdrant_db._get_documents_impl(where={"status": "active"})
        assert "ids" in result
        assert len(result["ids"]) == 1

    @pytest.mark.asyncio
    @patch("hypertic.vectordb.qdrant.qdrant.AsyncQdrantClient")
    async def test_count_documents_impl(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client.collection_exists = AsyncMock(return_value=True)
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 42
        mock_client.get_collection = AsyncMock(return_value=mock_collection_info)
        mock_client_class.return_value = mock_client

        count = await qdrant_db._count_documents_impl()
        assert count == 42

    @patch("hypertic.vectordb.qdrant.qdrant.QdrantClient")
    def test_count_documents_impl_sync(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_collection_info = MagicMock()
        mock_collection_info.points_count = 42
        mock_client.get_collection.return_value = mock_collection_info
        mock_client_class.return_value = mock_client

        count = qdrant_db._count_documents_impl_sync()
        assert count == 42
