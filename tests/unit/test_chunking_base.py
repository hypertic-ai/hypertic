import pytest

from hypertic.vectordb.chunking.base import Chunk, ChunkingStrategy


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

    def test_chunk_with_overlap(self):
        chunk = Chunk(
            content="Test",
            metadata={},
            chunk_type="document",
            overlap_with_previous=True,
            overlap_with_next=True,
        )
        assert chunk.overlap_with_previous is True
        assert chunk.overlap_with_next is True

    def test_chunk_with_start_index(self):
        chunk = Chunk(
            content="Test",
            metadata={"start_index": 10},
            chunk_type="document",
        )
        assert chunk.metadata is not None
        assert chunk.metadata.get("start_index") == 10


class ConcreteChunkingStrategy(ChunkingStrategy):
    """Concrete implementation for testing base class methods."""

    def __init__(
        self, chunk_size=100, chunk_overlap=20, separators=None, keep_separator=False, strip_headers=False, threshold=0.5, min_sentences_per_chunk=2
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " "]
        self.keep_separator = keep_separator
        self.strip_headers = strip_headers
        self.threshold = threshold
        self.min_sentences_per_chunk = min_sentences_per_chunk

    def _get_chunk_type(self) -> str:
        return "test"

    def _chunk_impl_async(self, text: str, params: dict) -> list:
        return self._chunk_impl_sync(text, params)

    def _chunk_impl_sync(self, text: str, params: dict) -> list:
        # Simple chunking implementation for testing
        chunk_size = params["chunk_size"]
        chunk_overlap = params["chunk_overlap"]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append(chunk_text)
            start = end - chunk_overlap
            if start >= len(text):
                break
        return self._create_chunks_from_texts(chunks, self._get_chunk_type(), chunk_size, chunk_overlap, params.get("add_start_index", False))


class TestChunkingStrategy:
    def test_chunking_strategy_is_abstract(self):
        with pytest.raises(TypeError):
            ChunkingStrategy()  # type: ignore[abstract]

    def test_chunking_strategy_has_required_methods(self):
        assert hasattr(ChunkingStrategy, "chunk")
        assert hasattr(ChunkingStrategy, "chunk_sync")

    def test_extract_parameters_with_strip_headers(self):
        """Test _extract_parameters with strip_headers attribute."""
        strategy = ConcreteChunkingStrategy(strip_headers=True)
        params = strategy._extract_parameters({"strip_headers": False}, "test")
        assert "strip_headers" in params
        assert params["strip_headers"] is False

    def test_extract_parameters_with_threshold(self):
        """Test _extract_parameters with threshold attribute."""
        strategy = ConcreteChunkingStrategy(threshold=0.7)
        params = strategy._extract_parameters({"threshold": 0.8}, "test")
        assert "threshold" in params
        assert params["threshold"] == 0.8

    def test_extract_parameters_with_min_sentences_per_chunk(self):
        """Test _extract_parameters with min_sentences_per_chunk attribute."""
        strategy = ConcreteChunkingStrategy(min_sentences_per_chunk=3)
        params = strategy._extract_parameters({"min_sentences_per_chunk": 4}, "test")
        assert "min_sentences_per_chunk" in params
        assert params["min_sentences_per_chunk"] == 4

    def test_create_chunks_from_texts_with_start_index(self):
        """Test _create_chunks_from_texts with add_start_index=True."""
        strategy = ConcreteChunkingStrategy()
        chunks = strategy._create_chunks_from_texts(["chunk1", "chunk2", "chunk3"], "test", chunk_size=100, chunk_overlap=20, add_start_index=True)
        assert len(chunks) == 3
        assert chunks[0].metadata.get("start_index") == 0
        assert chunks[1].metadata.get("start_index") == 80  # 100 - 20
        assert chunks[2].metadata.get("start_index") == 160  # 2 * (100 - 20)

    def test_create_chunks_from_texts_without_start_index(self):
        """Test _create_chunks_from_texts with add_start_index=False."""
        strategy = ConcreteChunkingStrategy()
        chunks = strategy._create_chunks_from_texts(["chunk1", "chunk2"], "test", chunk_size=100, chunk_overlap=20, add_start_index=False)
        assert len(chunks) == 2
        assert "start_index" not in chunks[0].metadata

    def test_create_chunks_from_tuples(self):
        """Test _create_chunks_from_tuples method."""
        strategy = ConcreteChunkingStrategy()
        chunk_tuples = [
            ("chunk1", {"source": "doc1"}),
            ("chunk2", {"source": "doc2"}),
            ("chunk3", {"source": "doc3"}),
        ]
        chunks = strategy._create_chunks_from_tuples(chunk_tuples, "test", chunk_size=100, chunk_overlap=20, add_start_index=True)
        assert len(chunks) == 3
        assert chunks[0].content == "chunk1"
        assert chunks[0].metadata["source"] == "doc1"
        assert chunks[0].metadata["start_index"] == 0
        assert chunks[1].content == "chunk2"
        assert chunks[1].metadata["source"] == "doc2"
        assert chunks[1].metadata["start_index"] == 80
        assert chunks[2].content == "chunk3"
        assert chunks[2].metadata["source"] == "doc3"
        assert chunks[2].metadata["start_index"] == 160

    def test_create_chunks_from_tuples_without_start_index(self):
        """Test _create_chunks_from_tuples with add_start_index=False."""
        strategy = ConcreteChunkingStrategy()
        chunk_tuples = [
            ("chunk1", {"source": "doc1"}),
            ("chunk2", {"source": "doc2"}),
        ]
        chunks = strategy._create_chunks_from_tuples(chunk_tuples, "test", chunk_size=100, chunk_overlap=20, add_start_index=False)
        assert len(chunks) == 2
        assert "start_index" not in chunks[0].metadata
        assert chunks[0].metadata["source"] == "doc1"

    def test_chunk_unified_small_text(self):
        """Test chunk_unified with text smaller than chunk_size."""
        strategy = ConcreteChunkingStrategy(chunk_size=100)
        chunks = strategy.chunk_unified("short text", is_async=False)
        assert len(chunks) == 1
        assert chunks[0].content == "short text"

    def test_chunk_unified_async(self):
        """Test chunk_unified with is_async=True."""
        strategy = ConcreteChunkingStrategy(chunk_size=10, chunk_overlap=2)
        chunks = strategy.chunk_unified("This is a longer text that needs chunking", is_async=True)
        assert len(chunks) > 1

    def test_chunk_unified_sync(self):
        """Test chunk_unified with is_async=False."""
        strategy = ConcreteChunkingStrategy(chunk_size=10, chunk_overlap=2)
        chunks = strategy.chunk_unified("This is a longer text that needs chunking", is_async=False)
        assert len(chunks) > 1

    def test_chunk_post_init_with_none_metadata(self):
        """Test Chunk __post_init__ with None metadata."""
        chunk = Chunk(content="test", metadata=None)
        assert chunk.metadata == {}

    def test_chunk_post_init_with_metadata(self):
        """Test Chunk __post_init__ with existing metadata."""
        chunk = Chunk(content="test", metadata={"key": "value"})
        assert chunk.metadata == {"key": "value"}
