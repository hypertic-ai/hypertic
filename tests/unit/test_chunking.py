import pytest

from hypertic.vectordb.chunking.base import Chunk, ChunkingStrategy
from hypertic.vectordb.chunking.strategies import DocumentChunker


class TestChunkingStrategy:
    def test_chunking_strategy_is_abstract(self):
        with pytest.raises(TypeError):
            ChunkingStrategy()  # type: ignore[abstract]


class TestDocumentChunker:
    @pytest.fixture
    def chunker(self):
        return DocumentChunker(chunk_size=100, chunk_overlap=20)

    def test_document_chunker_creation(self, chunker):
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20

    def test_document_chunker_default_separators(self, chunker):
        assert chunker.separators is not None
        assert len(chunker.separators) > 0

    def test_document_chunker_chunk_sync(self, chunker):
        text = "This is a test. " * 50
        chunks = chunker.chunk_sync(text)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    @pytest.mark.asyncio
    async def test_document_chunker_chunk_async(self, chunker):
        text = "This is a test. " * 50
        chunks = await chunker.chunk(text)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_document_chunker_chunk_with_overlap(self, chunker):
        text = "This is a test. " * 50
        chunks = chunker.chunk_sync(text)
        if len(chunks) > 1:
            assert chunks[0].overlap_with_next is True
            assert chunks[-1].overlap_with_previous is True


class TestChunk:
    def test_chunk_creation(self):
        chunk = Chunk(
            content="Test content",
            metadata={"source": "test"},
            chunk_type="document",
        )
        assert chunk.content == "Test content"
        assert chunk.metadata == {"source": "test"}
        assert chunk.chunk_type == "document"
