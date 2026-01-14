import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.vectordb.document_loader import DocumentLoader


@pytest.mark.unit
class TestDocumentLoader:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def loader(self):
        return DocumentLoader()

    def test_document_loader_creation(self, loader):
        """Test DocumentLoader initialization."""
        assert loader.processor is not None

    def test_document_loader_with_custom_processor(self):
        """Test DocumentLoader with custom processor."""
        mock_processor = MagicMock()
        loader = DocumentLoader(processor=mock_processor)
        assert loader.processor == mock_processor

    def test_get_file_type_local_file(self, loader):
        """Test _get_file_type for local file."""
        file_type = loader._get_file_type("/path/to/file.txt")
        assert file_type == ".txt"

    def test_get_file_type_url(self, loader):
        """Test _get_file_type for URL."""
        file_type = loader._get_file_type("https://example.com/document.pdf")
        assert file_type == ".pdf"

    def test_get_file_type_url_with_query(self, loader):
        """Test _get_file_type for URL with query parameters."""
        file_type = loader._get_file_type("https://example.com/file.docx?param=value")
        assert file_type == ".docx"

    def test_prepare_file_metadata_no_metadatas_no_ids(self, loader):
        """Test _prepare_file_metadata with no metadatas or ids."""
        files = ["file1.txt", "file2.txt"]
        metadatas, ids = loader._prepare_file_metadata(files, None, None)
        assert len(metadatas) == 2
        assert len(ids) == 2
        assert metadatas[0] == {}
        assert metadatas[1] == {}

    def test_prepare_file_metadata_with_metadatas(self, loader):
        """Test _prepare_file_metadata with provided metadatas."""
        files = ["file1.txt", "file2.txt"]
        metadatas = [{"key1": "value1"}, {"key2": "value2"}]
        _, ids = loader._prepare_file_metadata(files, metadatas, None)
        assert len(ids) == 2

    def test_prepare_file_metadata_with_ids(self, loader):
        """Test _prepare_file_metadata with provided ids."""
        files = ["file1.txt", "file2.txt"]
        ids = ["id1", "id2"]
        metadatas, result_ids = loader._prepare_file_metadata(files, None, ids)
        assert result_ids == ids

    def test_prepare_file_metadata_url_file(self, loader):
        """Test _prepare_file_metadata generates ID for URL file."""
        files = ["https://example.com/file.txt"]
        _, ids = loader._prepare_file_metadata(files, None, None)
        assert len(ids) == 1
        assert ids[0] is not None

    def test_prepare_file_metadata_local_file(self, loader, temp_dir):
        """Test _prepare_file_metadata generates ID for local file."""
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        files = [test_file]
        _, ids = loader._prepare_file_metadata(files, None, None)
        assert len(ids) == 1
        assert ids[0] is not None

    def test_prepare_file_metadata_file_stat_error(self, loader):
        """Test _prepare_file_metadata handles file stat error."""
        files = ["/nonexistent/file.txt"]
        _, ids = loader._prepare_file_metadata(files, None, None)
        assert len(ids) == 1
        # Should generate UUID on error
        assert ids[0] is not None

    def test_create_document_from_chunk_single_chunk(self, loader):
        """Test _create_document_from_chunk with single chunk."""
        from hypertic.vectordb.chunking.base import Chunk

        chunk = Chunk(content="test content", metadata={"key": "value"})
        doc = loader._create_document_from_chunk(chunk, doc_id="doc1", metadata={"meta": "data"}, file_path="file.txt", chunk_index=0, total_chunks=1)
        assert doc.id == "doc1"
        assert doc.content == "test content"
        assert doc.metadata["source"] == "file.txt"
        assert doc.metadata["chunk_index"] == 0
        assert doc.metadata["total_chunks"] == 1

    def test_create_document_from_chunk_multiple_chunks(self, loader):
        """Test _create_document_from_chunk with multiple chunks."""
        from hypertic.vectordb.chunking.base import Chunk

        chunk = Chunk(content="chunk content", metadata={"chunk_meta": "value"})
        doc = loader._create_document_from_chunk(chunk, doc_id="doc1", metadata={"meta": "data"}, file_path="file.txt", chunk_index=1, total_chunks=3)
        assert doc.id != "doc1"  # Should generate UUID5 for chunk
        assert doc.metadata["chunk_index"] == 1
        assert doc.metadata["total_chunks"] == 3
        assert doc.metadata["original_doc_id"] == "doc1"
        assert doc.metadata["chunk_meta"] == "value"

    def test_create_document_from_content(self, loader):
        """Test _create_document_from_content."""
        doc = loader._create_document_from_content("test content", doc_id="doc1", metadata={"key": "value"}, file_path="file.txt")
        assert doc.id == "doc1"
        assert doc.content == "test content"
        assert doc.metadata["source"] == "file.txt"
        assert doc.metadata["key"] == "value"
        assert doc.metadata["file_type"] == ".txt"

    @pytest.mark.asyncio
    async def test_load_files_no_chunking(self, loader):
        """Test load_files without chunking."""
        mock_processor = MagicMock()
        mock_processor.process_file = AsyncMock(return_value="file content")
        loader.processor = mock_processor

        documents = await loader.load_files(["file1.txt"], chunking_strategy="none")
        assert len(documents) == 1
        assert documents[0].content == "file content"

    @pytest.mark.asyncio
    async def test_load_files_with_chunking(self, loader):
        """Test load_files with chunking."""
        from hypertic.vectordb.chunking.base import Chunk

        mock_processor = MagicMock()
        mock_processor.process_file = AsyncMock(return_value="file content")
        loader.processor = mock_processor

        mock_chunker = MagicMock()
        mock_chunker.chunk = AsyncMock(return_value=[Chunk(content="chunk1"), Chunk(content="chunk2")])
        with patch("hypertic.vectordb.document_loader.get_chunker", return_value=mock_chunker):
            documents = await loader.load_files(["file1.txt"], chunking_strategy="document")
            assert len(documents) == 2
            assert documents[0].content == "chunk1"
            assert documents[1].content == "chunk2"

    @pytest.mark.asyncio
    async def test_load_files_multiple_files(self, loader):
        """Test load_files with multiple files."""
        mock_processor = MagicMock()
        mock_processor.process_file = AsyncMock(side_effect=["content1", "content2"])
        loader.processor = mock_processor

        documents = await loader.load_files(["file1.txt", "file2.txt"], chunking_strategy="none")
        assert len(documents) == 2

    @pytest.mark.asyncio
    async def test_load_files_with_metadatas(self, loader):
        """Test load_files with metadatas."""
        mock_processor = MagicMock()
        mock_processor.process_file = AsyncMock(return_value="content")
        loader.processor = mock_processor

        documents = await loader.load_files(["file1.txt"], metadatas=[{"key": "value"}], chunking_strategy="none")
        assert len(documents) == 1
        assert documents[0].metadata["key"] == "value"

    @pytest.mark.asyncio
    async def test_load_files_with_ids(self, loader):
        """Test load_files with provided ids."""
        mock_processor = MagicMock()
        mock_processor.process_file = AsyncMock(return_value="content")
        loader.processor = mock_processor

        documents = await loader.load_files(["file1.txt"], ids=["custom_id"], chunking_strategy="none")
        assert len(documents) == 1
        assert documents[0].id == "custom_id"

    @pytest.mark.asyncio
    async def test_load_files_error_handling(self, loader):
        """Test load_files handles errors gracefully."""
        mock_processor = MagicMock()
        mock_processor.process_file = AsyncMock(side_effect=Exception("Processing error"))
        loader.processor = mock_processor

        documents = await loader.load_files(["file1.txt"], chunking_strategy="none")
        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_load_files_with_chunking_params(self, loader):
        """Test load_files with chunking parameters."""
        from hypertic.vectordb.chunking.base import Chunk

        mock_processor = MagicMock()
        mock_processor.process_file = AsyncMock(return_value="content")
        loader.processor = mock_processor

        mock_chunker = MagicMock()
        mock_chunker.chunk = AsyncMock(return_value=[Chunk(content="chunk")])
        with patch("hypertic.vectordb.document_loader.get_chunker", return_value=mock_chunker):
            documents = await loader.load_files(["file1.txt"], chunking_strategy="document", chunking_params={"chunk_size": 100})
            assert len(documents) == 1

    def test_load_files_sync_no_chunking(self, loader):
        """Test load_files_sync without chunking."""
        mock_processor = MagicMock()
        mock_processor.process_file_sync = Mock(return_value="file content")
        loader.processor = mock_processor

        documents = loader.load_files_sync(["file1.txt"], chunking_strategy="none")
        assert len(documents) == 1
        assert documents[0].content == "file content"

    def test_load_files_sync_with_chunking(self, loader):
        """Test load_files_sync with chunking."""
        from hypertic.vectordb.chunking.base import Chunk

        mock_processor = MagicMock()
        mock_processor.process_file_sync = Mock(return_value="file content")
        loader.processor = mock_processor

        mock_chunker = MagicMock()
        mock_chunker.chunk_sync = Mock(return_value=[Chunk(content="chunk1"), Chunk(content="chunk2")])
        with patch("hypertic.vectordb.document_loader.get_chunker_sync", return_value=mock_chunker):
            documents = loader.load_files_sync(["file1.txt"], chunking_strategy="document")
            assert len(documents) == 2

    def test_load_files_sync_multiple_files(self, loader):
        """Test load_files_sync with multiple files."""
        mock_processor = MagicMock()
        mock_processor.process_file_sync = Mock(side_effect=["content1", "content2"])
        loader.processor = mock_processor

        documents = loader.load_files_sync(["file1.txt", "file2.txt"], chunking_strategy="none")
        assert len(documents) == 2

    def test_load_files_sync_with_metadatas(self, loader):
        """Test load_files_sync with metadatas."""
        mock_processor = MagicMock()
        mock_processor.process_file_sync = Mock(return_value="content")
        loader.processor = mock_processor

        documents = loader.load_files_sync(["file1.txt"], metadatas=[{"key": "value"}], chunking_strategy="none")
        assert len(documents) == 1
        assert documents[0].metadata["key"] == "value"

    def test_load_files_sync_with_ids(self, loader):
        """Test load_files_sync with provided ids."""
        mock_processor = MagicMock()
        mock_processor.process_file_sync = Mock(return_value="content")
        loader.processor = mock_processor

        documents = loader.load_files_sync(["file1.txt"], ids=["custom_id"], chunking_strategy="none")
        assert len(documents) == 1
        assert documents[0].id == "custom_id"

    def test_load_files_sync_error_handling(self, loader):
        """Test load_files_sync handles errors gracefully."""
        mock_processor = MagicMock()
        mock_processor.process_file_sync = Mock(side_effect=Exception("Processing error"))
        loader.processor = mock_processor

        documents = loader.load_files_sync(["file1.txt"], chunking_strategy="none")
        assert len(documents) == 0

    def test_load_files_sync_with_chunking_params(self, loader):
        """Test load_files_sync with chunking parameters."""
        from hypertic.vectordb.chunking.base import Chunk

        mock_processor = MagicMock()
        mock_processor.process_file_sync = Mock(return_value="content")
        loader.processor = mock_processor

        mock_chunker = MagicMock()
        mock_chunker.chunk_sync = Mock(return_value=[Chunk(content="chunk")])
        with patch("hypertic.vectordb.document_loader.get_chunker_sync", return_value=mock_chunker):
            documents = loader.load_files_sync(["file1.txt"], chunking_strategy="document", chunking_params={"chunk_size": 100})
            assert len(documents) == 1
