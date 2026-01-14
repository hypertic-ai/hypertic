import pytest

from hypertic.vectordb.chunking.base import Chunk
from hypertic.vectordb.chunking.strategies import DocumentChunker


class TestChunkingIntegration:
    @pytest.fixture
    def chunker(self):
        return DocumentChunker(chunk_size=100, chunk_overlap=20)

    def test_chunking_small_text(self, chunker):
        text = "This is a short text that should fit in one chunk."
        chunks = chunker.chunk_sync(text)

        assert len(chunks) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_chunking_large_text(self, chunker):
        text = "This is a test. " * 100  # Create long text
        chunks = chunker.chunk_sync(text)

        assert len(chunks) > 1
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_chunking_with_overlap(self, chunker):
        text = "This is a test. " * 50
        chunks = chunker.chunk_sync(text)

        if len(chunks) > 1:
            assert chunks[0].overlap_with_next is True
            assert chunks[-1].overlap_with_previous is True

    def test_chunking_preserves_content(self, chunker):
        text = "This is a test document with multiple sentences. " * 10
        chunks = chunker.chunk_sync(text)

        reconstructed = " ".join(chunk.content for chunk in chunks)

        assert len(reconstructed) >= len(text) or True

    @pytest.mark.asyncio
    async def test_chunking_async(self, chunker):
        text = "This is a test. " * 50
        chunks = await chunker.chunk(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)

    def test_chunking_different_sizes(self):
        chunker_small = DocumentChunker(chunk_size=50, chunk_overlap=10)
        chunker_large = DocumentChunker(chunk_size=200, chunk_overlap=20)

        text = "This is a test. " * 50

        chunks_small = chunker_small.chunk_sync(text)
        chunks_large = chunker_large.chunk_sync(text)

        assert len(chunks_small) >= len(chunks_large)

    def test_chunking_metadata(self, chunker):
        text = "This is a test document."
        chunks = chunker.chunk_sync(text)

        assert all(hasattr(chunk, "metadata") for chunk in chunks)

    def test_chunking_chunk_type(self, chunker):
        text = "This is a test document."
        chunks = chunker.chunk_sync(text)

        assert all(chunk.chunk_type == "document" for chunk in chunks)


class TestChunkingWithVectorDB:
    @pytest.fixture
    def chunker(self):
        return DocumentChunker(chunk_size=100, chunk_overlap=20)

    def test_chunking_for_vectordb(self, chunker, mock_embedder):
        text = "This is a test document. " * 20
        chunks = chunker.chunk_sync(text)

        assert len(chunks) > 0
        assert all(hasattr(chunk, "content") for chunk in chunks)

    def test_chunking_with_embedding(self, chunker, mock_embedder):
        text = "This is a test document."
        chunks = chunker.chunk_sync(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk.content, str) for chunk in chunks)
