import pytest

from hypertic.utils.files import File, FileType


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
    def test_file_creation(self):
        """Test file creation with auto-detection."""
        file_obj = File(filepath="path/to/file.jpg")
        assert file_obj.filepath == "path/to/file.jpg"
        assert file_obj.file_type == FileType.IMAGE

    def test_file_with_type(self):
        """Test file creation with explicit type."""
        file_obj = File(filepath="path/to/file.jpg", file_type=FileType.IMAGE)
        assert file_obj.filepath == "path/to/file.jpg"
        assert file_obj.file_type == FileType.IMAGE

    @pytest.mark.parametrize(
        "filepath,file_type",
        [
            ("image.png", FileType.IMAGE),
            ("audio.mp3", FileType.AUDIO),
            ("doc.pdf", FileType.DOCUMENT),
            ("video.mp4", FileType.VIDEO),
        ],
    )
    def test_file_with_explicit_type(self, filepath, file_type):
        """Test file creation with explicit type override."""
        file_obj = File(filepath=filepath, file_type=file_type)
        assert file_obj.file_type == file_type
