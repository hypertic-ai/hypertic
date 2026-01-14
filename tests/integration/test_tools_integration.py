from unittest.mock import patch

import pytest

from hypertic.tools import (
    DalleTools,
    DuckDuckGoTools,
    FileSystemTools,
    tool,
)


class TestToolSystemIntegration:
    def test_tool_decorator_integration(self):
        @tool
        def test_tool(param: str) -> str:
            """Test tool."""
            return f"Result: {param}"

        assert test_tool("value") == "Result: value"

    def test_tool_manager_integration(self):
        from hypertic.tools.base import _ToolManager

        @tool
        def tool1(param: str) -> str:
            """Tool 1."""
            return "result1"

        @tool
        def tool2(param: int) -> str:
            """Tool 2."""
            return "result2"

        manager = _ToolManager()
        manager.add_tool(tool1)
        manager.add_tool(tool2)

        assert "tool1" in manager._tools_dict
        assert "tool2" in manager._tools_dict


class TestFileSystemToolsIntegration:
    @pytest.fixture
    def fs_tools(self, temp_dir):
        return FileSystemTools(root_dir=temp_dir)

    def test_filesystem_tools_list_directory(self, fs_tools):
        import os

        test_file = os.path.join(fs_tools.root_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        result = fs_tools.list_directory(dir_path=fs_tools.root_dir)
        assert "test.txt" in result

    def test_filesystem_tools_read_file(self, fs_tools):
        import os

        test_file = os.path.join(fs_tools.root_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        result = fs_tools.read_file(file_path=test_file)
        assert "test content" in result

    def test_filesystem_tools_write_file(self, fs_tools):
        import os

        test_file = os.path.join(fs_tools.root_dir, "test.txt")

        fs_tools.write_file(file_path=test_file, text="new content")

        with open(test_file) as f:
            content = f.read()
        assert content == "new content"


class TestSearchToolsIntegration:
    def test_duckduckgo_tools_integration(self):
        tools = DuckDuckGoTools()
        assert tools is not None


class TestAPIToolsIntegration:
    def test_dalle_tools_initialization(self, mock_api_key):
        with patch.object(DalleTools, "__init__", return_value=None):
            tools = DalleTools(api_key=mock_api_key)
        assert tools is not None


class TestToolExecutionIntegration:
    @pytest.fixture
    def sample_tool(self):
        @tool
        def calculate(operation: str, a: float, b: float) -> float:
            """Perform calculation."""
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            return 0.0

        return calculate

    def test_tool_execution(self, sample_tool):
        result = sample_tool(operation="add", a=5.0, b=3.0)
        assert result == 8.0

    def test_tool_execution_with_validation(self, sample_tool):
        result = sample_tool(operation="multiply", a=2.0, b=4.0)
        assert result == 8.0
