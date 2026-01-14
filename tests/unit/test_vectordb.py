from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.vectordb.base import BaseVectorDB, VectorDocument, VectorSearchResult
from hypertic.vectordb.chroma.chroma import ChromaDB
from hypertic.vectordb.mongovector.mongovector import MongoDBAtlas
from hypertic.vectordb.pgvector.pgvector import PgVectorDB
from hypertic.vectordb.pinecone.pinecone import PineconeDB
from hypertic.vectordb.qdrant.qdrant import QdrantDB


class TestVectorDocument:
    def test_vector_document_creation(self, mock_embedding):
        doc = VectorDocument(
            id="doc_1",
            content="Test content",
            vector=mock_embedding,
            metadata={"source": "test"},
        )
        assert doc.id == "doc_1"
        assert doc.content == "Test content"
        assert doc.vector == mock_embedding
        assert doc.metadata == {"source": "test"}


class TestVectorSearchResult:
    def test_vector_search_result_creation(self, mock_embedding):
        result = VectorSearchResult(content="Test", metadata={}, score=0.95)
        assert result.content == "Test"
        assert result.score == 0.95


class TestChromaDB:
    @pytest.fixture
    def chroma_db(self, mock_embedder):
        return ChromaDB(embedder=mock_embedder, collection="test_collection")

    def test_chroma_db_creation(self, chroma_db):
        assert chroma_db is not None
        assert chroma_db.collection == "test_collection"

    @pytest.mark.asyncio
    async def test_chroma_db_initialize(self, chroma_db):
        # Mock the client and collection creation
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch.object(chroma_db, "_get_client_sync", return_value=mock_client):
            result = await chroma_db.initialize()
            assert result is True


class TestPineconeDB:
    @pytest.fixture
    def pinecone_db(self, mock_api_key, mock_embedder):
        return PineconeDB(api_key=mock_api_key, embedder=mock_embedder, collection="test_index")

    def test_pinecone_db_creation(self, pinecone_db):
        assert pinecone_db is not None
        assert pinecone_db.collection == "test_index"

    @pytest.mark.asyncio
    async def test_pinecone_db_initialize(self, pinecone_db):
        with patch.object(pinecone_db, "_get_client", return_value=MagicMock()):
            result = await pinecone_db.initialize()
            assert result is True


class TestQdrantDB:
    @pytest.fixture
    def qdrant_db(self, mock_embedder):
        mock_embedder.dimensions = 3
        return QdrantDB(embedder=mock_embedder, collection="test_collection")

    def test_qdrant_db_creation(self, qdrant_db):
        assert qdrant_db is not None
        assert qdrant_db.collection == "test_collection"

    @pytest.mark.asyncio
    @patch("qdrant_client.QdrantClient")
    async def test_qdrant_db_initialize(self, mock_client_class, qdrant_db):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        result = await qdrant_db.initialize()
        assert result is True


class TestPgVectorDB:
    @pytest.fixture
    def pgvector_db(self, mock_embedder):
        return PgVectorDB(
            embedder=mock_embedder,
            db_url="postgresql://localhost/test",
            collection="test_table",
        )

    def test_pgvector_db_creation(self, pgvector_db):
        assert pgvector_db is not None
        assert pgvector_db.collection == "test_table"

    @pytest.mark.asyncio
    async def test_pgvector_db_initialize(self, pgvector_db):
        with patch.object(pgvector_db, "_initialize_db", AsyncMock(return_value=True)):
            result = await pgvector_db.initialize()
            assert result is True


class TestMongoDBAtlas:
    @pytest.fixture
    def mongodb_atlas(self, mock_embedder):
        return MongoDBAtlas(
            embedder=mock_embedder,
            connection_string="mongodb://localhost:27017",
            database_name="test_db",
            collection="test_collection",
        )

    def test_mongodb_atlas_creation(self, mongodb_atlas):
        assert mongodb_atlas is not None
        assert mongodb_atlas.database_name == "test_db"
        assert mongodb_atlas.collection_name == "test_collection"

    @pytest.mark.asyncio
    async def test_mongodb_atlas_initialize(self, mongodb_atlas):
        with patch.object(mongodb_atlas, "_initialize_db", AsyncMock(return_value=True)):
            result = await mongodb_atlas.initialize()
            assert result is True


def create_concrete_vectordb_class():
    """Helper to create a complete concrete BaseVectorDB implementation."""

    class ConcreteVectorDB(BaseVectorDB):
        async def _initialize_db(self) -> bool:
            return True

        async def _add_documents_impl(self, documents: list[VectorDocument]) -> bool:
            return True

        async def _search_impl(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
            return []

        async def _search_text_impl(self, query: str, top_k: int) -> list[VectorSearchResult]:
            return []

        async def _delete_impl(self, ids: list[str] | None = None, where: dict | None = None) -> bool:
            return True

        async def _update_impl(
            self,
            ids: list[str],
            metadatas: list[dict] | None = None,
            documents: list[str] | None = None,
            embeddings: list[list[float]] | None = None,
        ) -> bool:
            return True

        async def _upsert_impl(
            self,
            ids: list[str],
            metadatas: list[dict] | None = None,
            documents: list[str] | None = None,
            embeddings: list[list[float]] | None = None,
        ) -> bool:
            return True

        async def _get_documents_impl(
            self,
            ids: list[str] | None = None,
            where: dict | None = None,
            limit: int | None = None,
            include: list[str] | None = None,
        ) -> dict:
            return {}

        def _get_documents_impl_sync(
            self,
            ids: list[str] | None = None,
            where: dict | None = None,
            limit: int | None = None,
            include: list[str] | None = None,
        ) -> dict:
            return {}

        async def _count_documents_impl(self) -> int:
            return 0

        def _count_documents_impl_sync(self) -> int:
            return 0

        def _initialize_db_sync(self) -> bool:
            return True

        def _add_documents_impl_sync(self, documents: list[VectorDocument]) -> bool:
            return True

        def _search_impl_sync(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
            return []

        def _search_text_impl_sync(self, query: str, top_k: int) -> list[VectorSearchResult]:
            return []

        def _delete_impl_sync(self, ids: list[str] | None = None, where: dict | None = None) -> bool:
            return True

        def _update_impl_sync(
            self,
            ids: list[str],
            metadatas: list[dict] | None = None,
            documents: list[str] | None = None,
            embeddings: list[list[float]] | None = None,
        ) -> bool:
            return True

        def _upsert_impl_sync(
            self,
            ids: list[str],
            metadatas: list[dict] | None = None,
            documents: list[str] | None = None,
            embeddings: list[list[float]] | None = None,
        ) -> bool:
            return True

        def search(self, query: str, top_k: int = 5) -> list[VectorSearchResult]:
            return []

    return ConcreteVectorDB


class TestBaseVectorDB:
    """Test BaseVectorDB abstract class methods"""

    def test_base_vectordb_is_abstract(self):
        """Test that BaseVectorDB is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseVectorDB()  # type: ignore[abstract]

    def test_base_vectordb_initialization(self, mock_embedder):
        """Test BaseVectorDB can be subclassed and initialized."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806
        db = ConcreteVectorDB(embedder=mock_embedder)
        assert db.embedder == mock_embedder
        assert db.initialized is False
        assert db._document_loader is not None

    @pytest.mark.asyncio
    async def test_base_vectordb_initialize_with_embedder(self, mock_embedder):
        """Test initialize method with embedder."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806
        mock_embedder.initialize = AsyncMock(return_value=True)
        db = ConcreteVectorDB(embedder=mock_embedder)
        result = await db.initialize()
        assert result is True
        mock_embedder.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_base_vectordb_initialize_without_embedder(self):
        """Test initialize method without embedder."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806
        db = ConcreteVectorDB()
        result = await db.initialize()
        assert result is True

    @pytest.mark.asyncio
    async def test_base_vectordb_add_documents_with_embedder(self, mock_embedder, mock_embedding):
        """Test add_documents with embedder."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806
        mock_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 100, [0.2] * 100])
        db = ConcreteVectorDB(embedder=mock_embedder)
        docs = [
            VectorDocument(id="doc1", content="test1", vector=None),
            VectorDocument(id="doc2", content="test2", vector=None),
        ]
        result = await db.add_documents(docs)
        assert result is True
        assert docs[0].vector == [0.1] * 100
        assert docs[1].vector == [0.2] * 100
        mock_embedder.embed_batch.assert_called_once_with(["test1", "test2"])

    @pytest.mark.asyncio
    async def test_base_vectordb_add_documents_with_existing_vectors(self, mock_embedder):
        """Test add_documents with documents that already have vectors."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806
        db = ConcreteVectorDB(embedder=mock_embedder)
        existing_vector = [0.5] * 100
        docs = [
            VectorDocument(id="doc1", content="test1", vector=existing_vector),
        ]
        result = await db.add_documents(docs)
        assert result is True
        assert docs[0].vector == existing_vector
        # embed_batch should not be called since vector already exists
        mock_embedder.embed_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_base_vectordb_async_search_impl_with_embedder(self, mock_embedder, mock_embedding):
        """Test _async_search_impl with embedder."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806

        # Override _search_impl for this test
        original_class = ConcreteVectorDB

        class TestVectorDB(original_class):
            async def _search_impl(self, query_vector: list[float], top_k: int) -> list[VectorSearchResult]:
                return [VectorSearchResult(content="result", metadata={}, score=0.9)]

        ConcreteVectorDB = TestVectorDB  # noqa: N806
        mock_embedder.embed = AsyncMock(return_value=mock_embedding)
        db = ConcreteVectorDB(embedder=mock_embedder)
        results = await db._async_search_impl("test query", top_k=5)
        assert len(results) == 1
        assert results[0].content == "result"
        mock_embedder.embed.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_base_vectordb_async_search_impl_without_embedder(self):
        """Test _async_search_impl without embedder."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806

        # Override _search_text_impl for this test
        original_class = ConcreteVectorDB

        class TestVectorDB(original_class):
            async def _search_text_impl(self, query: str, top_k: int) -> list[VectorSearchResult]:
                return [VectorSearchResult(content="text result", metadata={}, score=0.8)]

        ConcreteVectorDB = TestVectorDB  # noqa: N806
        db = ConcreteVectorDB()
        results = await db._async_search_impl("test query", top_k=5)
        assert len(results) == 1
        assert results[0].content == "text result"

    @pytest.mark.asyncio
    async def test_base_vectordb_async_delete(self):
        """Test async_delete method."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806
        db = ConcreteVectorDB()
        result = await db.async_delete(ids=["doc1", "doc2"])
        assert result is True

    @pytest.mark.asyncio
    async def test_base_vectordb_async_get(self):
        """Test async_get method."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806

        # Override _get_documents_impl for this test
        original_class = ConcreteVectorDB

        class TestVectorDB(original_class):
            async def _get_documents_impl(
                self,
                ids: list[str] | None = None,
                where: dict | None = None,
                limit: int | None = None,
                include: list[str] | None = None,
            ) -> dict:
                return {"ids": ids or [], "data": []}

        ConcreteVectorDB = TestVectorDB  # noqa: N806
        db = ConcreteVectorDB()
        result = await db.async_get(ids=["doc1"], limit=10, include=["metadata"])
        assert result == {"ids": ["doc1"], "data": []}

    def test_base_vectordb_get_sync(self):
        """Test get method (sync)."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806

        # Override _get_documents_impl_sync for this test
        original_class = ConcreteVectorDB

        class TestVectorDB(original_class):
            def _get_documents_impl_sync(
                self,
                ids: list[str] | None = None,
                where: dict | None = None,
                limit: int | None = None,
                include: list[str] | None = None,
            ) -> dict:
                return {"ids": ids or [], "data": []}

        ConcreteVectorDB = TestVectorDB  # noqa: N806
        db = ConcreteVectorDB()
        result = db.get(ids=["doc1"], where={"key": "value"})
        assert result == {"ids": ["doc1"], "data": []}

    @pytest.mark.asyncio
    async def test_base_vectordb_async_add_with_texts(self, mock_embedder, mock_embedding):
        """Test async_add with texts parameter."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806
        mock_embedder.embed_batch = AsyncMock(return_value=[[0.1] * 100, [0.2] * 100])
        db = ConcreteVectorDB(embedder=mock_embedder)

        with patch.object(db, "add_documents", new_callable=AsyncMock, return_value=True) as mock_add_docs:
            result = await db.async_add(texts=["Hello world", "Another text"])
            assert result is True
            mock_add_docs.assert_called_once()
            # Verify documents were created correctly
            call_args = mock_add_docs.call_args[0][0]
            assert len(call_args) == 2
            assert call_args[0].content == "Hello world"
            assert call_args[1].content == "Another text"
            # IDs should be MD5 hashes
            assert call_args[0].id == "3e25960a79dbc69b674cd4ec67a72c62"  # MD5 of "Hello world"
            assert call_args[0].metadata == {}
            assert call_args[1].metadata == {}

    @pytest.mark.asyncio
    async def test_base_vectordb_async_add_with_texts_and_metadata(self, mock_embedder):
        """Test async_add with texts and custom metadata/ids."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806
        db = ConcreteVectorDB(embedder=mock_embedder)

        with patch.object(db, "add_documents", new_callable=AsyncMock, return_value=True) as mock_add_docs:
            result = await db.async_add(
                texts=["Text 1", "Text 2"],
                metadatas=[{"source": "api"}, {"source": "user"}],
                ids=["custom-id-1", "custom-id-2"],
            )
            assert result is True
            call_args = mock_add_docs.call_args[0][0]
            assert len(call_args) == 2
            assert call_args[0].id == "custom-id-1"
            assert call_args[0].metadata == {"source": "api"}
            assert call_args[1].id == "custom-id-2"
            assert call_args[1].metadata == {"source": "user"}

    @pytest.mark.asyncio
    async def test_base_vectordb_async_add_validation_errors(self):
        """Test async_add validation - neither files nor texts provided."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806
        db = ConcreteVectorDB()

        with pytest.raises(ValueError, match="Either 'files' or 'texts' must be provided"):
            await db.async_add()

    @pytest.mark.asyncio
    async def test_base_vectordb_async_add_both_provided_error(self):
        """Test async_add validation - both files and texts provided."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806
        db = ConcreteVectorDB()

        with pytest.raises(ValueError, match="Cannot provide both 'files' and 'texts'"):
            await db.async_add(files=["file.txt"], texts=["text"])

    def test_base_vectordb_add_with_texts_sync(self, mock_embedder):
        """Test add (sync) with texts parameter."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806
        mock_embedder.embed_batch_sync = MagicMock(return_value=[[0.1] * 100, [0.2] * 100])
        db = ConcreteVectorDB(embedder=mock_embedder)

        with patch.object(db, "add_documents_sync", return_value=True) as mock_add_docs:
            result = db.add(texts=["Hello world", "Another text"])
            assert result is True
            mock_add_docs.assert_called_once()
            call_args = mock_add_docs.call_args[0][0]
            assert len(call_args) == 2
            assert call_args[0].content == "Hello world"
            assert call_args[1].content == "Another text"

    def test_base_vectordb_add_validation_errors_sync(self):
        """Test add (sync) validation - neither files nor texts provided."""
        ConcreteVectorDB = create_concrete_vectordb_class()  # noqa: N806
        db = ConcreteVectorDB()

        with pytest.raises(ValueError, match="Either 'files' or 'texts' must be provided"):
            db.add()
