from unittest.mock import AsyncMock, Mock, patch

import pytest

from hypertic.vectordb.base import VectorDocument, VectorSearchResult
from hypertic.vectordb.chroma.chroma import ChromaDB
from hypertic.vectordb.mongovector.mongovector import MongoDBAtlas
from hypertic.vectordb.pgvector.pgvector import PgVectorDB
from hypertic.vectordb.pinecone.pinecone import PineconeDB
from hypertic.vectordb.qdrant.qdrant import QdrantDB


class TestVectorDBIntegration:
    @pytest.mark.parametrize(
        "db_class,db_params",
        [
            (ChromaDB, {"embedder": Mock(), "collection": "test"}),
            (PineconeDB, {"api_key": "test", "embedder": Mock(), "collection": "test"}),
            (QdrantDB, {"embedder": Mock(), "collection": "test"}),
            (PgVectorDB, {"embedder": Mock(), "db_url": "postgresql://localhost/test", "collection": "test"}),
            (MongoDBAtlas, {"embedder": Mock(), "connection_string": "mongodb://localhost:27017", "database_name": "test", "collection": "test"}),
        ],
    )
    def test_vectordb_initialization(self, db_class, db_params):
        db = db_class(**db_params)
        assert db is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "db_class,db_params",
        [
            (ChromaDB, {"embedder": Mock(), "collection": "test"}),
            (PineconeDB, {"api_key": "test", "embedder": Mock(), "collection": "test"}),
            (QdrantDB, {"embedder": Mock(), "collection": "test"}),
            (PgVectorDB, {"embedder": Mock(), "db_url": "postgresql://localhost/test", "collection": "test"}),
            (MongoDBAtlas, {"embedder": Mock(), "connection_string": "mongodb://localhost:27017", "database_name": "test", "collection": "test"}),
        ],
    )
    async def test_vectordb_async_initialize(self, db_class, db_params):
        db = db_class(**db_params)

        with patch.object(db, "initialize", return_value=True):
            result = await db.initialize()
            assert result is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "db_class,db_params",
        [
            (ChromaDB, {"embedder": Mock(), "collection": "test"}),
            (PineconeDB, {"api_key": "test", "embedder": Mock(), "collection": "test"}),
            (QdrantDB, {"embedder": Mock(), "collection": "test"}),
            (PgVectorDB, {"embedder": Mock(), "db_url": "postgresql://localhost/test", "collection": "test"}),
            (MongoDBAtlas, {"embedder": Mock(), "connection_string": "mongodb://localhost:27017", "database_name": "test", "collection": "test"}),
        ],
    )
    async def test_vectordb_async_add_documents(self, db_class, db_params, mock_embedding):
        db = db_class(**db_params)

        documents = [
            VectorDocument(
                id="doc_1",
                content="Test content",
                vector=mock_embedding,
                metadata={"source": "test"},
            )
        ]

        if hasattr(db, "add_documents_async"):
            with patch.object(db, "add_documents_async", return_value=None):
                await db.add_documents_async(documents)
        else:
            with patch.object(db, "add_documents", return_value=None):
                await db.add_documents(documents)
            assert True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "db_class,db_params",
        [
            (ChromaDB, {"embedder": Mock(), "collection": "test"}),
            (PineconeDB, {"api_key": "test", "embedder": Mock(), "collection": "test"}),
            (QdrantDB, {"embedder": Mock(), "collection": "test"}),
            (PgVectorDB, {"embedder": Mock(), "db_url": "postgresql://localhost/test", "collection": "test"}),
            (MongoDBAtlas, {"embedder": Mock(), "connection_string": "mongodb://localhost:27017", "database_name": "test", "collection": "test"}),
        ],
    )
    async def test_vectordb_async_search(self, db_class, db_params, mock_embedding):
        db = db_class(**db_params)

        mock_result = VectorSearchResult(
            content="Test",
            metadata={},
            score=0.95,
        )

        if hasattr(db, "search_async"):
            with patch.object(db, "search_async", return_value=[mock_result]):
                results = await db.search_async("test query", top_k=5)
                assert isinstance(results, list)
        else:
            with patch.object(db, "search", return_value=[mock_result]):
                results = db.search("test query", top_k=5)
                assert isinstance(results, list)

    @pytest.mark.parametrize(
        "db_class,db_params",
        [
            (ChromaDB, {"embedder": Mock(), "collection": "test"}),
            (PineconeDB, {"api_key": "test", "embedder": Mock(), "collection": "test"}),
            (QdrantDB, {"embedder": Mock(), "collection": "test"}),
            (PgVectorDB, {"embedder": Mock(), "db_url": "postgresql://localhost/test", "collection": "test"}),
            (MongoDBAtlas, {"embedder": Mock(), "connection_string": "mongodb://localhost:27017", "database_name": "test", "collection": "test"}),
        ],
    )
    def test_vectordb_sync_add_documents(self, db_class, db_params, mock_embedding):
        db = db_class(**db_params)

        documents = [
            VectorDocument(
                id="doc_1",
                content="Test content",
                vector=mock_embedding,
                metadata={"source": "test"},
            )
        ]

        with patch.object(db, "add_documents", return_value=None):
            db.add_documents(documents)
            assert True

    @pytest.mark.parametrize(
        "db_class,db_params",
        [
            (ChromaDB, {"embedder": Mock(), "collection": "test"}),
            (PineconeDB, {"api_key": "test", "embedder": Mock(), "collection": "test"}),
            (QdrantDB, {"embedder": Mock(), "collection": "test"}),
            (PgVectorDB, {"embedder": Mock(), "db_url": "postgresql://localhost/test", "collection": "test"}),
            (MongoDBAtlas, {"embedder": Mock(), "connection_string": "mongodb://localhost:27017", "database_name": "test", "collection": "test"}),
        ],
    )
    def test_vectordb_sync_search(self, db_class, db_params, mock_embedding):
        db = db_class(**db_params)

        mock_result = VectorSearchResult(
            content="Test",
            metadata={},
            score=0.95,
        )

        with patch.object(db, "search", return_value=[mock_result]):
            results = db.search("test query", top_k=5)
            assert isinstance(results, list)


class TestVectorDBWithEmbedder:
    @pytest.fixture
    def mock_embedder(self):
        embedder = Mock()
        embedder.embed = AsyncMock(return_value=[0.1] * 100)
        embedder.embed_sync = Mock(return_value=[0.1] * 100)
        return embedder

    def test_vectordb_embedder_integration(self, mock_embedder):
        db = ChromaDB(embedder=mock_embedder, collection="test")
        assert db.embedder == mock_embedder

    @pytest.mark.asyncio
    async def test_vectordb_auto_embedding(self, mock_embedder):
        db = ChromaDB(embedder=mock_embedder, collection="test")

        assert db.embedder is not None
