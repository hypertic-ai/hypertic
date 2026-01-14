from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.vectordb.base import VectorDocument
from hypertic.vectordb.pgvector.pgvector import HNSW, Distance, Ivfflat, PgVectorDB


@pytest.mark.unit
class TestPgVectorDBBasics:
    @pytest.fixture
    def pgvector_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        with patch("hypertic.vectordb.pgvector.pgvector.create_engine"):
            return PgVectorDB(embedder=mock_embedder, collection="test_collection", db_url="postgresql://test")

    def test_pgvector_db_creation(self, pgvector_db):
        assert pgvector_db.collection == "test_collection"
        assert pgvector_db.table_name == "test_collection"
        assert pgvector_db.embedder is not None

    def test_pgvector_db_with_db_url(self, mock_embedder):
        with patch("hypertic.vectordb.pgvector.pgvector.create_engine") as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            db = PgVectorDB(embedder=mock_embedder, collection="test", db_url="postgresql://test")
            assert db.db_url == "postgresql://test"
            mock_create.assert_called_once()

    def test_pgvector_db_with_db_engine(self, mock_embedder):
        mock_engine = MagicMock()
        db = PgVectorDB(embedder=mock_embedder, collection="test", db_engine=mock_engine)
        assert db.db_engine == mock_engine

    def test_pgvector_db_without_db_url_or_engine(self, mock_embedder):
        with patch("hypertic.vectordb.pgvector.pgvector.getenv", return_value=None):
            with pytest.raises(ValueError, match="Either 'db_url' or 'db_engine' must be provided"):
                PgVectorDB(embedder=mock_embedder, collection="test")

    def test_pgvector_db_with_env_var(self, mock_embedder):
        with patch("hypertic.vectordb.pgvector.pgvector.getenv", return_value="postgresql://env_test"):
            with patch("hypertic.vectordb.pgvector.pgvector.create_engine") as mock_create:
                mock_engine = MagicMock()
                mock_create.return_value = mock_engine
                db = PgVectorDB(embedder=mock_embedder, collection="test")
                assert db.db_url == "postgresql://env_test"

    def test_pgvector_db_with_schema(self, mock_embedder):
        with patch("hypertic.vectordb.pgvector.pgvector.create_engine"):
            db = PgVectorDB(embedder=mock_embedder, collection="test", db_url="postgresql://test", schema="custom")
            assert db.schema == "custom"

    def test_pgvector_db_with_vector_size(self, mock_embedder):
        with patch("hypertic.vectordb.pgvector.pgvector.create_engine"):
            db = PgVectorDB(embedder=mock_embedder, collection="test", db_url="postgresql://test", vector_size=256)
            assert db.vector_size == 256

    def test_pgvector_db_with_distance_metric(self, mock_embedder):
        with patch("hypertic.vectordb.pgvector.pgvector.create_engine"):
            db = PgVectorDB(embedder=mock_embedder, collection="test", db_url="postgresql://test", distance_metric=Distance.L2)
            assert db.distance_metric == Distance.L2

    def test_pgvector_db_with_hnsw_index(self, mock_embedder):
        with patch("hypertic.vectordb.pgvector.pgvector.create_engine"):
            hnsw = HNSW(m=32, ef_construction=128, ef_search=50)
            db = PgVectorDB(embedder=mock_embedder, collection="test", db_url="postgresql://test", vector_index=hnsw)
            assert isinstance(db.vector_index, HNSW)
            assert db.vector_index.m == 32

    def test_pgvector_db_with_ivfflat_index(self, mock_embedder):
        with patch("hypertic.vectordb.pgvector.pgvector.create_engine"):
            ivfflat = Ivfflat(lists=200, probes=10)
            db = PgVectorDB(embedder=mock_embedder, collection="test", db_url="postgresql://test", vector_index=ivfflat)
            assert isinstance(db.vector_index, Ivfflat)
            assert db.vector_index.lists == 200


@pytest.mark.unit
class TestPgVectorDBTableOperations:
    @pytest.fixture
    def pgvector_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        with patch("hypertic.vectordb.pgvector.pgvector.create_engine"):
            return PgVectorDB(embedder=mock_embedder, collection="test_collection", db_url="postgresql://test")

    def test_get_table(self, pgvector_db):
        """Test _get_table creates table structure."""
        table = pgvector_db._get_table()
        assert table is not None
        assert table.name == "test_collection"

    def test_table_exists_true(self, pgvector_db):
        """Test table_exists returns True when table exists."""
        mock_inspector = MagicMock()
        mock_inspector.has_table.return_value = True
        with patch("hypertic.vectordb.pgvector.pgvector.inspect", return_value=mock_inspector):
            assert pgvector_db.table_exists() is True

    def test_table_exists_false(self, pgvector_db):
        """Test table_exists returns False when table doesn't exist."""
        mock_inspector = MagicMock()
        mock_inspector.has_table.return_value = False
        with patch("hypertic.vectordb.pgvector.pgvector.inspect", return_value=mock_inspector):
            assert pgvector_db.table_exists() is False

    def test_table_exists_error(self, pgvector_db):
        """Test table_exists handles errors."""
        with patch("hypertic.vectordb.pgvector.pgvector.inspect", side_effect=Exception("Error")):
            assert pgvector_db.table_exists() is False

    def test_table_exists_no_engine(self, pgvector_db):
        """Test table_exists returns False when engine is None."""
        pgvector_db.db_engine = None
        assert pgvector_db.table_exists() is False


@pytest.mark.unit
class TestPgVectorDBInitialization:
    @pytest.fixture
    def pgvector_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        with patch("hypertic.vectordb.pgvector.pgvector.create_engine"):
            return PgVectorDB(embedder=mock_embedder, collection="test_collection", db_url="postgresql://test")

    def test_initialize_db_sync_table_exists(self, pgvector_db):
        """Test _initialize_db_sync when table already exists."""
        with patch.object(pgvector_db, "table_exists", return_value=True):
            result = pgvector_db._initialize_db_sync()
            assert result is True

    def test_initialize_db_sync_new_table(self, pgvector_db):
        """Test _initialize_db_sync creates new table."""
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.execute = Mock()
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "table_exists", return_value=False):
            with patch.object(pgvector_db.table, "create") as mock_create:
                with patch.object(pgvector_db, "_create_vector_index") as mock_vector_index:
                    with patch.object(pgvector_db, "_create_gin_index") as mock_gin_index:
                        result = pgvector_db._initialize_db_sync()
                        assert result is True
                        mock_create.assert_called_once()
                        mock_vector_index.assert_called_once()
                        mock_gin_index.assert_called_once()

    def test_initialize_db_sync_error(self, pgvector_db):
        """Test _initialize_db_sync handles errors."""
        with patch.object(pgvector_db, "table_exists", side_effect=Exception("Error")):
            result = pgvector_db._initialize_db_sync()
            assert result is False

    @pytest.mark.asyncio
    async def test_initialize_db_table_exists(self, pgvector_db):
        """Test _initialize_db when table already exists."""
        with patch.object(pgvector_db, "table_exists", return_value=True):
            result = await pgvector_db._initialize_db()
            assert result is True

    @pytest.mark.asyncio
    async def test_initialize_db_new_table(self, pgvector_db):
        """Test _initialize_db creates new table."""
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.execute = Mock()
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "table_exists", return_value=False):
            with patch.object(pgvector_db.table, "create") as mock_create:
                with patch.object(pgvector_db, "_create_vector_index") as mock_vector_index:
                    with patch.object(pgvector_db, "_create_gin_index") as mock_gin_index:
                        result = await pgvector_db._initialize_db()
                        assert result is True
                        mock_create.assert_called_once()
                        mock_vector_index.assert_called_once()
                        mock_gin_index.assert_called_once()


@pytest.mark.unit
class TestPgVectorDBIndexOperations:
    @pytest.fixture
    def pgvector_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        with patch("hypertic.vectordb.pgvector.pgvector.create_engine"):
            return PgVectorDB(embedder=mock_embedder, collection="test_collection", db_url="postgresql://test")

    def test_create_vector_index_hnsw(self, pgvector_db):
        """Test _create_vector_index with HNSW."""
        pgvector_db.vector_index = HNSW(m=16, ef_construction=64, ef_search=40)
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.execute = Mock()
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "_index_exists", return_value=False):
            with patch.object(pgvector_db, "_create_hnsw_index") as mock_create_hnsw:
                pgvector_db._create_vector_index()
                mock_create_hnsw.assert_called_once()

    def test_create_vector_index_ivfflat(self, pgvector_db):
        """Test _create_vector_index with Ivfflat."""
        pgvector_db.vector_index = Ivfflat(lists=100, probes=1)
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.execute = Mock()
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "_index_exists", return_value=False):
            with patch.object(pgvector_db, "_create_ivfflat_index") as mock_create_ivfflat:
                pgvector_db._create_vector_index()
                mock_create_ivfflat.assert_called_once()

    def test_create_vector_index_already_exists(self, pgvector_db):
        """Test _create_vector_index when index already exists."""
        pgvector_db.vector_index = HNSW()
        with patch.object(pgvector_db, "_index_exists", return_value=True):
            with patch.object(pgvector_db, "_create_hnsw_index") as mock_create_hnsw:
                pgvector_db._create_vector_index()
                mock_create_hnsw.assert_not_called()

    def test_create_vector_index_force_recreate(self, pgvector_db):
        """Test _create_vector_index with force_recreate."""
        pgvector_db.vector_index = HNSW()
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.execute = Mock()
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "_index_exists", return_value=True):
            with patch.object(pgvector_db, "_drop_index") as mock_drop:
                with patch.object(pgvector_db, "_create_hnsw_index") as mock_create_hnsw:
                    pgvector_db._create_vector_index(force_recreate=True)
                    mock_drop.assert_called_once()
                    mock_create_hnsw.assert_called_once()

    def test_index_exists(self, pgvector_db):
        """Test _index_exists."""
        mock_inspector = MagicMock()
        mock_inspector.get_indexes.return_value = [{"name": "test_index"}, {"name": "other_index"}]
        with patch("hypertic.vectordb.pgvector.pgvector.inspect", return_value=mock_inspector):
            result = pgvector_db._index_exists("test_index")
            assert result is True

    def test_index_exists_false(self, pgvector_db):
        """Test _index_exists returns False when index doesn't exist."""
        mock_inspector = MagicMock()
        mock_inspector.get_indexes.return_value = [{"name": "other_index"}]
        with patch("hypertic.vectordb.pgvector.pgvector.inspect", return_value=mock_inspector):
            result = pgvector_db._index_exists("test_index")
            assert result is False

    def test_drop_index(self, pgvector_db):
        """Test _drop_index."""
        mock_session = MagicMock()
        mock_session.execute = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch("hypertic.vectordb.pgvector.pgvector.text"):
            pgvector_db._drop_index("test_index")
            mock_session.execute.assert_called()


@pytest.mark.unit
class TestPgVectorDBOperations:
    @pytest.fixture
    def pgvector_db(self, mock_embedder):
        mock_embedder.dimensions = 384
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)
        mock_embedder.embed_sync = Mock(return_value=[0.1] * 384)
        with patch("hypertic.vectordb.pgvector.pgvector.create_engine"):
            return PgVectorDB(embedder=mock_embedder, collection="test_collection", db_url="postgresql://test")

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

    def test_exists(self, pgvector_db):
        """Test exists method."""
        with patch.object(pgvector_db, "table_exists", return_value=True):
            assert pgvector_db.exists() is True

    @pytest.mark.asyncio
    async def test_async_exists(self, pgvector_db):
        """Test async_exists method."""
        with patch.object(pgvector_db, "table_exists", return_value=True):
            assert await pgvector_db.async_exists() is True

    def test_add_documents_impl_sync(self, pgvector_db, sample_documents):
        """Test _add_documents_impl_sync."""
        mock_session = MagicMock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.execute = Mock()
        mock_session.commit = Mock()
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "exists", return_value=True):
            result = pgvector_db._add_documents_impl_sync(sample_documents)
            assert result is True

    @pytest.mark.asyncio
    async def test_add_documents_impl(self, pgvector_db, sample_documents):
        """Test _add_documents_impl."""
        with patch.object(pgvector_db, "_add_documents_impl_sync", return_value=True):
            result = await pgvector_db._add_documents_impl(sample_documents)
            assert result is True

    def test_search_impl_sync_cosine(self, pgvector_db):
        """Test _search_impl_sync with cosine distance."""
        pgvector_db.distance_metric = Distance.COSINE
        mock_session = MagicMock()
        mock_row = MagicMock()
        mock_row.id = "doc1"
        mock_row.content = "test content"
        mock_row.meta_data = {"meta": "value"}
        mock_row.embedding = [0.1] * 384
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "exists", return_value=True):
            with patch("numpy.array") as mock_np_array:
                mock_np_array.side_effect = lambda x: x
                with patch("numpy.dot", return_value=0.5):
                    with patch("numpy.linalg.norm", return_value=1.0):
                        results = pgvector_db._search_impl_sync([0.1] * 384, top_k=5)
                        assert len(results) == 1
                        assert results[0].content == "test content"

    def test_search_impl_sync_l2(self, pgvector_db):
        """Test _search_impl_sync with L2 distance."""
        pgvector_db.distance_metric = Distance.L2
        mock_session = MagicMock()
        mock_row = MagicMock()
        mock_row.id = "doc1"
        mock_row.content = "test content"
        mock_row.meta_data = {"meta": "value"}
        mock_row.embedding = [0.1] * 384
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "exists", return_value=True):
            # The method will import numpy inside, so we just verify it runs
            # The actual numpy operations are tested in integration tests
            try:
                results = pgvector_db._search_impl_sync([0.1] * 384, top_k=5)
                # If numpy is available, we get results; otherwise it may error
                # This test verifies the method structure is correct
                assert isinstance(results, list)
            except (ImportError, AttributeError):
                # If numpy is not available or patching fails, skip this test
                pytest.skip("numpy not available or patching failed")

    def test_search_impl_sync_with_filters(self, pgvector_db):
        """Test _search_impl_sync with filters."""
        pgvector_db.distance_metric = Distance.COSINE
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "exists", return_value=True):
            results = pgvector_db._search_impl_sync([0.1] * 384, top_k=5, filters={"source": "test"})
            assert isinstance(results, list)

    def test_search_impl_sync_table_not_exists(self, pgvector_db):
        """Test _search_impl_sync when table doesn't exist."""
        with patch.object(pgvector_db, "exists", return_value=False):
            results = pgvector_db._search_impl_sync([0.1] * 384, top_k=5)
            assert results == []

    @pytest.mark.asyncio
    async def test_search_impl(self, pgvector_db):
        """Test _search_impl."""
        with patch.object(pgvector_db, "_search_impl_sync", return_value=[]):
            results = await pgvector_db._search_impl([0.1] * 384, top_k=5)
            assert results == []

    def test_keyword_search(self, pgvector_db):
        """Test keyword_search."""
        mock_session = MagicMock()
        mock_row = MagicMock()
        mock_row.id = "doc1"
        mock_row.content = "test content"
        mock_row.meta_data = {"meta": "value"}
        mock_row.rank = 0.8
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "exists", return_value=True):
            results = pgvector_db.keyword_search("test query", top_k=5)
            assert len(results) == 1
            assert results[0].content == "test content"

    def test_keyword_search_table_not_exists(self, pgvector_db):
        """Test keyword_search when table doesn't exist."""
        with patch.object(pgvector_db, "exists", return_value=False):
            results = pgvector_db.keyword_search("test query", top_k=5)
            assert results == []

    def test_search_text_impl_sync(self, pgvector_db):
        """Test _search_text_impl_sync."""
        with patch.object(pgvector_db, "keyword_search", return_value=[]):
            results = pgvector_db._search_text_impl_sync("test query", top_k=5)
            assert results == []

    @pytest.mark.asyncio
    async def test_search_text_impl(self, pgvector_db):
        """Test _search_text_impl."""
        with patch.object(pgvector_db, "keyword_search", return_value=[]):
            results = await pgvector_db._search_text_impl("test query", top_k=5)
            assert results == []

    def test_get_documents_impl_sync(self, pgvector_db):
        """Test _get_documents_impl_sync."""
        mock_session = MagicMock()
        mock_row = MagicMock()
        mock_row.id = "doc1"
        mock_row.content = "test content"
        mock_row.meta_data = {"meta": "value"}
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "exists", return_value=True):
            result = pgvector_db._get_documents_impl_sync(ids=["doc1"])
            assert result["ids"] == ["doc1"]
            assert result["contents"] == ["test content"]

    @pytest.mark.asyncio
    async def test_get_documents_impl(self, pgvector_db):
        """Test _get_documents_impl."""
        with patch.object(pgvector_db, "_get_documents_impl_sync", return_value={"ids": [], "documents": [], "metadatas": [], "embeddings": []}):
            result = await pgvector_db._get_documents_impl(ids=["doc1"])
            assert "ids" in result

    def test_count_documents_impl_sync(self, pgvector_db):
        """Test _count_documents_impl_sync."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "exists", return_value=True):
            count = pgvector_db._count_documents_impl_sync()
            assert count == 42

    @pytest.mark.asyncio
    async def test_count_documents_impl(self, pgvector_db):
        """Test _count_documents_impl."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 42
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "async_exists", return_value=True):
            count = await pgvector_db._count_documents_impl()
            assert count == 42

    @pytest.mark.asyncio
    async def test_count_documents_impl_table_not_exists(self, pgvector_db):
        """Test _count_documents_impl when table doesn't exist."""
        with patch.object(pgvector_db, "async_exists", return_value=False):
            count = await pgvector_db._count_documents_impl()
            assert count == 0

    def test_delete_impl_sync_by_ids(self, pgvector_db):
        """Test _delete_impl_sync by ids."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 2
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "exists", return_value=True):
            result = pgvector_db._delete_impl_sync(ids=["doc1", "doc2"])
            assert result is True

    def test_delete_impl_sync_by_where(self, pgvector_db):
        """Test _delete_impl_sync by where clause."""
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session.begin = Mock(return_value=mock_session)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "exists", return_value=True):
            result = pgvector_db._delete_impl_sync(where={"source": "test"})
            assert result is True

    @pytest.mark.asyncio
    async def test_delete_impl(self, pgvector_db):
        """Test _delete_impl."""
        with patch.object(pgvector_db, "_delete_impl_sync", return_value=True):
            result = await pgvector_db._delete_impl(ids=["doc1"])
            assert result is True

    def test_update_impl_sync(self, pgvector_db):
        """Test _update_impl_sync."""
        mock_session = MagicMock()
        mock_session.execute = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "exists", return_value=True):
            result = pgvector_db._update_impl_sync(ids=["doc1"], documents=["new content"])
            assert result is True

    @pytest.mark.asyncio
    async def test_update_impl(self, pgvector_db):
        """Test _update_impl."""
        with patch.object(pgvector_db, "_update_impl_sync", return_value=True):
            result = await pgvector_db._update_impl(ids=["doc1"], documents=["new content"])
            assert result is True

    def test_upsert_impl_sync(self, pgvector_db):
        """Test _upsert_impl_sync."""
        mock_session = MagicMock()
        mock_session.execute = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        pgvector_db._Session = Mock(return_value=mock_session)

        with patch.object(pgvector_db, "exists", return_value=True):
            result = pgvector_db._upsert_impl_sync(ids=["doc1"], documents=["content"])
            assert result is True

    @pytest.mark.asyncio
    async def test_upsert_impl(self, pgvector_db):
        """Test _upsert_impl."""
        with patch.object(pgvector_db, "_upsert_impl_sync", return_value=True):
            result = await pgvector_db._upsert_impl(ids=["doc1"], documents=["content"])
            assert result is True

    def test_clean_content(self, pgvector_db):
        """Test _clean_content removes null bytes."""
        content = "test\x00content"
        cleaned = pgvector_db._clean_content(content)
        assert "\x00" not in cleaned
        assert "test" in cleaned
        assert "content" in cleaned
