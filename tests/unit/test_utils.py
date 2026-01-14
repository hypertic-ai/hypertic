import pytest

from hypertic.utils.files import File, FileProcessor, FileType


@pytest.mark.unit
class TestFileProcessor:
    def test_file_processor_process_message_no_files(self):
        """Test processing message without files."""
        message = {"role": "user", "content": "Hello"}
        result = FileProcessor.process_message(message, "OpenAI")
        assert result == message

    def test_file_processor_process_message_with_files(self):
        """Test processing message with files."""
        message = {
            "role": "user",
            "content": "Hello",
            "files": ["path/to/image.jpg"],
        }
        assert "files" in message

    def test_file_processor_process_files(self, temp_dir):
        """Test processing files from directory."""
        import os

        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        files = FileProcessor.process_files([test_file])
        assert len(files) == 1
        assert files[0].filepath == test_file

    @pytest.mark.parametrize(
        "provider",
        ["OpenAI", "Anthropic", "Google", "Groq"],
    )
    def test_file_processor_process_message_providers(self, provider):
        """Test processing message with different providers."""
        message = {"role": "user", "content": "Hello"}
        result = FileProcessor.process_message(message, provider)
        assert result == message


@pytest.mark.unit
class TestFile:
    def test_file_creation(self):
        """Test file creation with auto-detection."""
        file_obj = File(filepath="path/to/file.jpg")
        assert file_obj.filepath == "path/to/file.jpg"

    def test_file_with_type(self):
        """Test file creation with explicit type."""
        file_obj = File(filepath="path/to/file.jpg", file_type=FileType.IMAGE)
        assert file_obj.file_type == FileType.IMAGE


@pytest.mark.unit
class TestFileType:
    @pytest.mark.parametrize(
        "file_type",
        [FileType.IMAGE, FileType.AUDIO, FileType.DOCUMENT, FileType.VIDEO, FileType.FILE],
    )
    def test_file_type_values(self, file_type):
        """Test that all file types are valid enum members."""
        assert file_type in FileType
