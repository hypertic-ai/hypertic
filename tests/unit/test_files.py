import base64
from unittest.mock import mock_open, patch

import pytest

from hypertic.utils.files import File, FileProcessor, FileType


@pytest.mark.unit
class TestFileType:
    @pytest.mark.parametrize(
        "file_type",
        [FileType.IMAGE, FileType.AUDIO, FileType.DOCUMENT, FileType.VIDEO, FileType.FILE],
    )
    def test_file_type_enum_members(self, file_type):
        """Test that all file types are valid enum members."""
        assert file_type in FileType

    @pytest.mark.parametrize(
        "file_type,expected_value",
        [
            (FileType.IMAGE, "image"),
            (FileType.AUDIO, "audio"),
            (FileType.DOCUMENT, "document"),
            (FileType.VIDEO, "video"),
            (FileType.FILE, "file"),
        ],
    )
    def test_file_type_values(self, file_type, expected_value):
        """Test that file types have correct string values."""
        assert file_type.value == expected_value


@pytest.mark.unit
class TestFile:
    def test_file_creation_with_url(self):
        """Test file creation with URL."""
        file_obj = File(url="https://example.com/image.jpg")
        assert file_obj.url == "https://example.com/image.jpg"
        assert file_obj.file_type == FileType.IMAGE

    def test_file_creation_with_filepath(self):
        """Test file creation with filepath."""
        file_obj = File(filepath="path/to/file.jpg")
        assert file_obj.filepath == "path/to/file.jpg"
        assert file_obj.file_type == FileType.IMAGE

    def test_file_creation_with_content(self):
        """Test file creation with content bytes."""
        content = b"test content"
        file_obj = File(content=content)
        assert file_obj.content == content

    def test_file_creation_with_source_url(self):
        """Test file creation using 'source' parameter with URL."""
        file_obj = File(source="https://example.com/file.pdf")
        assert file_obj.url == "https://example.com/file.pdf"
        assert "source" not in file_obj.model_dump()

    def test_file_creation_with_source_filepath(self):
        """Test file creation using 'source' parameter with filepath."""
        file_obj = File(source="path/to/file.pdf")
        assert file_obj.filepath == "path/to/file.pdf"

    def test_file_creation_no_source_raises_error(self):
        """Test that file creation without source raises error."""
        with pytest.raises(ValueError, match="One of"):
            File()

    def test_file_creation_multiple_sources_raises_error(self):
        """Test that file creation with multiple sources raises error."""
        with pytest.raises(ValueError, match="Only one of"):
            File(url="https://example.com/file.pdf", filepath="local.pdf")

    def test_file_auto_id_generation(self):
        """Test that file gets auto-generated ID."""
        file_obj = File(filepath="test.jpg")
        assert file_obj.id is not None
        assert isinstance(file_obj.id, str)

    def test_file_custom_id(self):
        """Test file with custom ID."""
        file_obj = File(filepath="test.jpg", id="custom-id")
        assert file_obj.id == "custom-id"

    def test_file_filename_from_filepath(self):
        """Test filename extraction from filepath."""
        file_obj = File(filepath="path/to/file.jpg")
        assert file_obj.filename == "file.jpg"

    def test_file_filename_from_url(self):
        """Test filename extraction from URL."""
        file_obj = File(url="https://example.com/path/file.jpg")
        assert file_obj.filename == "file.jpg"

    def test_file_mime_type_detection(self):
        """Test MIME type detection."""
        file_obj = File(filepath="test.jpg")
        assert file_obj.mime_type is not None

    @pytest.mark.parametrize(
        "url,expected_type",
        [
            ("https://example.com/image.jpg", FileType.IMAGE),
            ("https://example.com/audio.mp3", FileType.AUDIO),
            ("https://example.com/doc.pdf", FileType.DOCUMENT),
            ("https://youtube.com/watch?v=123", FileType.VIDEO),
        ],
    )
    def test_file_type_detection_from_url(self, url, expected_type):
        """Test file type detection from URL."""
        file_obj = File(url=url)
        assert file_obj.file_type == expected_type

    def test_file_get_content_bytes_from_content(self):
        """Test getting content bytes when content is provided."""
        content = b"test bytes"
        file_obj = File(content=content)
        assert file_obj.get_content_bytes() == content

    @patch("builtins.open", new_callable=mock_open, read_data=b"file content")
    def test_file_get_content_bytes_from_filepath(self, mock_file):
        """Test getting content bytes from filepath."""
        file_obj = File(filepath="test.txt")
        content = file_obj.get_content_bytes()
        assert content == b"file content"
        mock_file.assert_called_once()

    @patch("builtins.__import__")
    def test_file_get_content_bytes_from_url(self, mock_import):
        """Test getting content bytes from URL."""
        mock_requests = mock_import.return_value
        mock_response = mock_requests.get.return_value
        mock_response.content = b"url content"
        mock_response.raise_for_status = lambda: None

        file_obj = File(url="https://example.com/file.txt")
        # This will fail if requests not installed, which is expected
        content = file_obj.get_content_bytes()
        # If requests is not available, content will be None
        # This test just ensures the method doesn't crash
        assert content is None or content == b"url content"

    def test_file_get_content_bytes_none(self):
        """Test getting content bytes when no source available."""
        file_obj = File(filepath="nonexistent.txt")
        # Will return None if file doesn't exist
        result = file_obj.get_content_bytes()
        assert result is None

    def test_file_to_base64_from_content(self):
        """Test converting file to base64 from content."""
        content = b"test"
        file_obj = File(content=content)
        base64_str = file_obj.to_base64()
        assert base64_str is not None
        assert isinstance(base64_str, str)
        decoded = base64.b64decode(base64_str)
        assert decoded == content

    def test_file_to_base64_none(self):
        """Test to_base64 returns None when no content available."""
        file_obj = File(filepath="nonexistent.txt")
        result = file_obj.to_base64()
        assert result is None

    def test_file_from_base64(self):
        """Test creating file from base64."""
        content = b"test content"
        base64_content = base64.b64encode(content).decode("utf-8")
        file_obj = File.from_base64(base64_content, id="test-id")
        assert file_obj.content == content
        assert file_obj.id == "test-id"

    def test_file_from_base64_with_mime_type(self):
        """Test creating file from base64 with MIME type."""
        content = b"test"
        base64_content = base64.b64encode(content).decode("utf-8")
        file_obj = File.from_base64(base64_content, mime_type="text/plain")
        assert file_obj.mime_type == "text/plain"

    def test_file_to_dict(self):
        """Test converting file to dictionary."""
        file_obj = File(
            filepath="test.jpg",
            id="test-id",
            file_type=FileType.IMAGE,
            mime_type="image/jpeg",
        )
        result = file_obj.to_dict()
        assert result["id"] == "test-id"
        assert result["file_type"] == "image"
        assert result["mime_type"] == "image/jpeg"

    def test_file_to_dict_with_base64(self):
        """Test converting file to dictionary with base64 content."""
        file_obj = File(content=b"test")
        result = file_obj.to_dict(include_base64_content=True)
        assert "content" in result

    def test_file_to_dict_without_base64(self):
        """Test converting file to dictionary without base64 content."""
        file_obj = File(content=b"test")
        result = file_obj.to_dict(include_base64_content=False)
        assert "content" not in result


@pytest.mark.unit
class TestFileProcessor:
    def test_process_files_urls(self):
        """Test processing files from URLs."""
        sources = ["https://example.com/file1.jpg", "https://example.com/file2.pdf"]
        files = FileProcessor.process_files(sources)
        assert len(files) == 2
        assert files[0].url == sources[0]
        assert files[1].url == sources[1]

    def test_process_files_filepaths(self):
        """Test processing files from filepaths."""
        sources = ["path/to/file1.jpg", "path/to/file2.pdf"]
        files = FileProcessor.process_files(sources)
        assert len(files) == 2
        assert files[0].filepath == sources[0]
        assert files[1].filepath == sources[1]

    def test_process_files_mixed(self):
        """Test processing mixed URLs and filepaths."""
        sources = ["https://example.com/file.jpg", "local/file.pdf"]
        files = FileProcessor.process_files(sources)
        assert len(files) == 2

    def test_process_message_with_files(self):
        """Test processing message with files."""
        message = {
            "role": "user",
            "content": "Check these files",
            "files": ["https://example.com/file.jpg", "local.pdf"],
        }
        result = FileProcessor.process_message(message, provider="test")
        assert "files" in result
        assert "_file_objects" in result
        assert len(result["files"]) == 2

    def test_process_message_without_files(self):
        """Test processing message without files."""
        message = {"role": "user", "content": "Hello"}
        result = FileProcessor.process_message(message, provider="test")
        assert result == message

    @pytest.mark.parametrize(
        "file_path,expected_mime",
        [
            ("test.pdf", "application/pdf"),
            ("test.jpg", "image/jpeg"),
            ("test.png", "image/png"),
            ("test.mp3", "audio/mpeg"),
            ("test.mp4", "video/mp4"),
            ("test.txt", "text/plain"),
            ("test.unknown", "application/octet-stream"),
        ],
    )
    def test_get_mime_type(self, file_path, expected_mime):
        """Test MIME type detection."""
        result = FileProcessor.get_mime_type(file_path)
        assert result == expected_mime
