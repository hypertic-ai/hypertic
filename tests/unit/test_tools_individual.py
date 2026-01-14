import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from hypertic.tools import (  # type: ignore[attr-defined]
    DalleTools,
    DuckDuckGoTools,
    FileSystemTools,
)


class TestFileSystemTools:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def fs_tools(self, temp_dir):
        return FileSystemTools(root_dir=temp_dir)

    def test_filesystem_tools_creation(self, fs_tools):
        assert fs_tools.root_dir is not None

    def test_filesystem_tools_no_root_dir(self):
        """Test FileSystemTools without root_dir."""
        fs_tools = FileSystemTools()
        assert fs_tools.root_dir is None

    def test_get_path_with_root_dir(self, fs_tools, temp_dir):
        """Test _get_path with root_dir."""
        path = fs_tools._get_path("test.txt")
        assert str(path) == os.path.join(temp_dir, "test.txt")

    def test_get_path_without_root_dir(self, temp_dir):
        """Test _get_path without root_dir."""
        fs_tools = FileSystemTools()
        path = fs_tools._get_path(str(temp_dir))
        assert str(path) == str(Path(temp_dir).resolve())

    def test_get_path_outside_root(self, fs_tools):
        """Test _get_path with path outside root_dir."""
        with pytest.raises(ValueError, match="outside allowed directory"):
            fs_tools._get_path("../../etc/passwd")

    def test_copy_file(self, fs_tools, temp_dir):
        """Test copy_file method."""
        source = os.path.join(temp_dir, "source.txt")
        dest = os.path.join(temp_dir, "dest.txt")
        with open(source, "w") as f:
            f.write("test content")
        result = fs_tools.copy_file(source_path="source.txt", destination_path="dest.txt")
        result_dict = json.loads(result)
        assert result_dict["success"] is True
        assert os.path.exists(dest)

    def test_copy_file_source_not_exists(self, fs_tools):
        """Test copy_file with non-existent source."""
        with pytest.raises(ValueError, match="Source file does not exist"):
            fs_tools.copy_file(source_path="nonexistent.txt", destination_path="dest.txt")

    def test_delete_file(self, fs_tools, temp_dir):
        """Test delete_file method."""
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        result = fs_tools.delete_file(file_path="test.txt")
        result_dict = json.loads(result)
        assert result_dict["success"] is True
        assert not os.path.exists(test_file)

    def test_search_files(self, fs_tools, temp_dir):
        """Test search_files method."""
        test_file = os.path.join(temp_dir, "test.py")
        with open(test_file, "w") as f:
            f.write("test")
        result = fs_tools.search_files(pattern="*.py", dir_path=".")
        result_dict = json.loads(result)
        assert len(result_dict["matches"]) > 0

    def test_move_file(self, fs_tools, temp_dir):
        """Test move_file method."""
        source = os.path.join(temp_dir, "source.txt")
        dest = os.path.join(temp_dir, "dest.txt")
        with open(source, "w") as f:
            f.write("test content")
        result = fs_tools.move_file(source_path="source.txt", destination_path="dest.txt")
        result_dict = json.loads(result)
        assert result_dict["success"] is True
        assert not os.path.exists(source)
        assert os.path.exists(dest)

    def test_read_file(self, fs_tools, temp_dir):
        """Test read_file method."""
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        result = fs_tools.read_file(file_path="test.txt")
        result_dict = json.loads(result)
        assert result_dict["content"] == "test content"

    def test_write_file(self, fs_tools, temp_dir):
        """Test write_file method."""
        result = fs_tools.write_file(file_path="test.txt", text="test content")
        result_dict = json.loads(result)
        assert result_dict["success"] is True
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file) as f:
            assert f.read() == "test content"

    def test_write_file_append(self, fs_tools, temp_dir):
        """Test write_file with append mode."""
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("original")
        result = fs_tools.write_file(file_path="test.txt", text=" appended", append=True)
        result_dict = json.loads(result)
        assert result_dict["success"] is True
        with open(test_file) as f:
            assert f.read() == "original appended"

    def test_list_directory(self, fs_tools, temp_dir):
        """Test list_directory method."""
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        result = fs_tools.list_directory(dir_path=".")
        assert "test.txt" in result


class TestDuckDuckGoTools:
    @pytest.fixture
    def ddg_tools(self):
        return DuckDuckGoTools()

    def test_duckduckgo_tools_creation(self, ddg_tools):
        assert ddg_tools is not None
        assert ddg_tools.max_results == 5

    def test_duckduckgo_tools_with_params(self):
        """Test DuckDuckGoTools with custom parameters."""
        tools = DuckDuckGoTools(modifier="site:example.com", max_results=10, region="uk-en")
        assert tools.modifier == "site:example.com"
        assert tools.max_results == 10
        assert tools.region == "uk-en"

    def test_duckduckgo_search_exists(self, ddg_tools):
        assert hasattr(ddg_tools, "search")
        assert callable(getattr(ddg_tools, "search", None))

    @patch("ddgs.DDGS")
    def test_search(self, mock_ddgs_class, ddg_tools):
        """Test search method."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = Mock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = Mock(return_value=None)
        mock_ddgs.text.return_value = [{"title": "Result 1", "body": "Content 1"}]
        mock_ddgs_class.return_value = mock_ddgs

        result = ddg_tools.search(query="test query")
        result_list = json.loads(result)
        assert len(result_list) == 1

    @patch("ddgs.DDGS")
    def test_search_with_modifier(self, mock_ddgs_class, ddg_tools):
        """Test search with modifier."""
        ddg_tools.modifier = "site:example.com"
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = Mock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = Mock(return_value=None)
        mock_ddgs.text.return_value = []
        mock_ddgs_class.return_value = mock_ddgs

        ddg_tools.search(query="test")
        call_args = mock_ddgs.text.call_args[1]
        assert "site:example.com test" in call_args["query"]

    @patch("ddgs.DDGS")
    def test_news(self, mock_ddgs_class, ddg_tools):
        """Test news method."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = Mock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = Mock(return_value=None)
        mock_ddgs.news.return_value = [{"title": "News 1"}]
        mock_ddgs_class.return_value = mock_ddgs

        result = ddg_tools.news(query="test")
        result_list = json.loads(result)
        assert len(result_list) == 1

    @patch("ddgs.DDGS")
    def test_images(self, mock_ddgs_class, ddg_tools):
        """Test images method."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = Mock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = Mock(return_value=None)
        mock_ddgs.images = Mock(return_value=iter([{"title": "Image 1", "image": "http://example.com/img.jpg"}]))
        mock_ddgs_class.return_value = mock_ddgs

        result = ddg_tools.images(query="test")
        result_list = json.loads(result)
        assert len(result_list) == 1


class TestDalleTools:
    @pytest.fixture
    def dalle_tools(self):
        return DalleTools(api_key="test")

    def test_dalle_tools_creation(self, dalle_tools):
        assert dalle_tools is not None
        assert dalle_tools.api_key == "test"
        assert dalle_tools.model == "dall-e-3"

    @patch("hypertic.tools.dalle.dalle.getenv")
    def test_dalle_tools_with_env_api_key(self, mock_getenv):
        """Test DalleTools with API key from environment."""
        mock_getenv.return_value = "env_key"
        tools = DalleTools()
        assert tools.api_key == "env_key"

    def test_dalle_tools_with_custom_params(self):
        """Test DalleTools with custom parameters."""
        tools = DalleTools(api_key="test", model="dall-e-2", size="512x512", quality="hd", style="natural")
        assert tools.model == "dall-e-2"
        assert tools.size == "512x512"
        assert tools.quality == "hd"
        assert tools.style == "natural"

    @patch("hypertic.tools.dalle.dalle.OpenAI")
    def test_generate_dalle3(self, mock_openai_class, dalle_tools):
        """Test generate with DALL-E 3."""
        mock_client = MagicMock()
        mock_image = MagicMock()
        mock_image.url = "http://example.com/image.jpg"
        mock_image.revised_prompt = "revised prompt"
        mock_response = MagicMock()
        mock_response.data = [mock_image]
        mock_client.images.generate.return_value = mock_response
        mock_openai_class.return_value = mock_client

        result = dalle_tools.generate(query="test image")
        result_dict = json.loads(result)
        assert result_dict["model"] == "dall-e-3"
        assert len(result_dict["images"]) == 1
        assert result_dict["images"][0]["url"] == "http://example.com/image.jpg"

    @patch("hypertic.tools.dalle.dalle.OpenAI")
    def test_generate_dalle2(self, mock_openai_class):
        """Test generate with DALL-E 2."""
        tools = DalleTools(api_key="test", model="dall-e-2", n=3)
        mock_client = MagicMock()

        # Create a simple object that can be JSON serialized
        class MockImage:
            def __init__(self):
                self.url = "http://example.com/image.jpg"

        mock_image = MockImage()
        mock_response = MagicMock()
        mock_response.data = [mock_image]
        mock_client.images.generate.return_value = mock_response
        mock_openai_class.return_value = mock_client

        result = tools.generate(query="test image")
        result_dict = json.loads(result)
        assert result_dict["model"] == "dall-e-2"
        assert len(result_dict["images"]) == 1

    def test_generate_no_api_key(self):
        """Test generate without API key."""
        tools = DalleTools()
        with pytest.raises(ValueError, match="API key is required"):
            tools.generate(query="test")
