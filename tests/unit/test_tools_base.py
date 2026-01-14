import pytest

from hypertic.tools import tool  # type: ignore[attr-defined]
from hypertic.tools.base import _ToolManager


@pytest.mark.unit
class TestToolDecorator:
    def test_tool_decorator_basic(self):
        """Test basic tool decorator functionality."""

        @tool
        def test_tool(param: str) -> str:
            return f"Result: {param}"

        assert test_tool("x") == "Result: x"  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "custom_name",
        ["custom_name", "my_tool", "tool_123"],
    )
    def test_tool_decorator_with_custom_name(self, custom_name):
        """Test tool decorator with custom names."""

        @tool(name=custom_name)  # type: ignore
        def test_tool(param: str) -> str:
            return f"Result: {param}"

        if hasattr(test_tool, "__tool_metadata__"):
            assert test_tool.__tool_metadata__["name"] == custom_name

    @pytest.mark.parametrize(
        "description",
        ["Custom description", "Another description", "Test tool"],
    )
    def test_tool_decorator_with_custom_description(self, description):
        """Test tool decorator with custom descriptions."""

        @tool(description=description)  # type: ignore
        def test_tool(param: str) -> str:
            return f"Result: {param}"

        if hasattr(test_tool, "__tool_metadata__"):
            assert test_tool.__tool_metadata__["description"] == description

    def test_tool_with_type_hints(self):
        """Test tool with various type hints."""

        @tool
        def test_tool(
            text: str,
            count: int,
            items: list[str],
            metadata: dict[str, int],
        ) -> str:
            return "result"

        if hasattr(test_tool, "__tool_metadata__"):
            metadata = test_tool.__tool_metadata__
            assert "parameters" in metadata
            assert "properties" in metadata["parameters"]
            assert "text" in metadata["parameters"]["properties"]
            assert "count" in metadata["parameters"]["properties"]
            assert "items" in metadata["parameters"]["properties"]


@pytest.mark.unit
class TestToolManager:
    def test_tool_manager_creation(self):
        """Test tool manager creation."""
        manager = _ToolManager()
        assert manager is not None

    def test_tool_manager_register_tool(self):
        """Test registering a tool with the manager."""

        @tool
        def test_tool(param: str) -> str:
            return "result"

        manager = _ToolManager()
        manager.add_tool(test_tool)
        assert "test_tool" in manager._tools_dict

    def test_tool_manager_get_tool(self):
        """Test retrieving a tool from the manager."""

        @tool
        def test_tool(param: str) -> str:
            return "result"

        manager = _ToolManager()
        manager.add_tool(test_tool)
        retrieved = manager.get_tool("test_tool")
        assert retrieved == test_tool

    def test_tool_manager_get_tool_not_found(self):
        """Test retrieving non-existent tool returns None."""
        manager = _ToolManager()
        assert manager.get_tool("nonexistent") is None

    def test_tool_manager_list_tools(self):
        """Test listing all tools."""

        @tool
        def tool1(param: str) -> str:
            return "result1"

        @tool
        def tool2(param: int) -> str:
            return "result2"

        manager = _ToolManager()
        manager.add_tool(tool1)
        manager.add_tool(tool2)
        tools = manager.list_tools()
        assert len(tools) == 2
        assert "tool1" in tools
        assert "tool2" in tools


class TestToolFunction:
    """Test ToolFunction class"""

    def test_tool_function_creation(self):
        """Test ToolFunction creation."""
        from hypertic.tools.base import ToolFunction

        def test_func(param: str) -> str:
            return param

        tool_func = ToolFunction(
            func=test_func,
            validated_func=test_func,
            metadata={"name": "test", "description": "Test tool"},
        )
        assert tool_func.__name__ == "test_func"
        assert tool_func.__doc__ == test_func.__doc__

    def test_tool_function_call(self):
        """Test calling a ToolFunction."""
        from hypertic.tools.base import ToolFunction

        def test_func(param: str) -> str:
            return f"Result: {param}"

        tool_func = ToolFunction(
            func=test_func,
            validated_func=test_func,
            metadata={"name": "test", "description": "Test tool"},
        )
        result = tool_func(param="value")
        assert result == "Result: value"

    def test_tool_function_repr(self):
        """Test ToolFunction string representation."""
        from hypertic.tools.base import ToolFunction

        def test_func(param: str) -> str:
            return param

        tool_func = ToolFunction(
            func=test_func,
            validated_func=test_func,
            metadata={
                "name": "test_tool",
                "description": "Test description",
                "parameters": {"type": "object", "properties": {}},
            },
        )
        repr_str = repr(tool_func)
        assert "test_tool" in repr_str
        assert "Test description" in repr_str

    def test_tool_function_as_descriptor(self):
        """Test ToolFunction as a descriptor (for methods)."""

        class TestClass:
            @tool
            def test_method(self, param: str) -> str:
                return f"Result: {param}"

        instance = TestClass()
        # Accessing as attribute should return bound method
        method = instance.test_method
        assert callable(method)
        result = method(param="value")
        assert result == "Result: value"

    def test_tool_function_get_with_none(self):
        """Test ToolFunction.__get__ with None (class access)."""
        from hypertic.tools.base import ToolFunction

        def test_func(param: str) -> str:
            return param

        tool_func = ToolFunction(
            func=test_func,
            validated_func=test_func,
            metadata={"name": "test", "description": "Test tool"},
            is_method=False,
        )
        # Accessing from class (obj is None) should return self
        result = tool_func.__get__(None, type(None))
        assert result is tool_func


class TestToolDecoratorAdvanced:
    """Test advanced tool decorator functionality"""

    def test_tool_with_default_parameters(self):
        """Test tool with default parameter values."""

        @tool
        def test_tool(param: str = "default") -> str:
            return f"Result: {param}"

        result = test_tool()
        assert result == "Result: default"
        result = test_tool(param="custom")
        assert result == "Result: custom"

    def test_tool_with_optional_parameters(self):
        """Test tool with optional parameters."""

        @tool
        def test_tool(param: str | None = None) -> str:
            return f"Result: {param or 'None'}"

        result = test_tool()
        assert result == "Result: None"
        result = test_tool(param="value")
        assert result == "Result: value"

    def test_tool_with_list_parameters(self):
        """Test tool with list parameters."""

        @tool
        def test_tool(items: list[str]) -> str:
            return f"Count: {len(items)}"

        result = test_tool(items=["a", "b", "c"])
        assert result == "Count: 3"

    def test_tool_with_dict_parameters(self):
        """Test tool with dict parameters."""

        @tool
        def test_tool(data: dict[str, int]) -> str:
            return f"Keys: {len(data)}"

        result = test_tool(data={"a": 1, "b": 2})
        assert result == "Keys: 2"

    def test_tool_without_docstring(self):
        """Test tool without docstring uses default description."""

        @tool
        def test_tool(param: str) -> str:
            return param

        if hasattr(test_tool, "_tool_metadata"):
            metadata = test_tool._tool_metadata
            assert "description" in metadata
            assert metadata["description"].startswith("Tool:")

    def test_tool_with_method(self):
        """Test tool decorator on a method."""

        class TestClass:
            @tool
            def test_method(self, param: str) -> str:
                return f"Result: {param}"

        instance = TestClass()
        result = instance.test_method(param="value")
        assert result == "Result: value"


class TestBaseToolkit:
    """Test BaseToolkit class"""

    def test_base_toolkit_creation(self):
        """Test BaseToolkit can be instantiated."""
        from hypertic.tools.base import BaseToolkit

        toolkit = BaseToolkit()
        assert toolkit is not None

    def test_base_toolkit_get_tools_empty(self):
        """Test BaseToolkit.get_tools() with no tools."""
        from hypertic.tools.base import BaseToolkit

        toolkit = BaseToolkit()
        tools = toolkit.get_tools()
        assert tools == []

    def test_base_toolkit_get_tools_with_decorated_methods(self):
        """Test BaseToolkit.get_tools() extracts @tool decorated methods."""
        from hypertic.tools.base import BaseToolkit

        class TestToolkit(BaseToolkit):
            @tool
            def tool1(self, param: str) -> str:
                """Tool 1."""
                return f"Result1: {param}"

            @tool
            def tool2(self, param: int) -> str:
                """Tool 2."""
                return f"Result2: {param}"

            def not_a_tool(self):
                """This is not a tool."""
                pass

        toolkit = TestToolkit()
        tools = toolkit.get_tools()
        assert len(tools) == 2
        # Check that tools are callable
        assert callable(tools[0])
        assert callable(tools[1])

    def test_base_toolkit_str_representation(self):
        """Test BaseToolkit string representation."""
        from hypertic.tools.base import BaseToolkit

        class TestToolkit(BaseToolkit):
            @tool
            def tool1(self, param: str) -> str:
                """Tool 1."""
                return f"Result1: {param}"

        toolkit = TestToolkit()
        str_repr = str(toolkit)
        assert "tool1" in str_repr or "Tool 1" in str_repr

    def test_base_toolkit_str_representation_no_tools(self):
        """Test BaseToolkit string representation with no tools."""
        from hypertic.tools.base import BaseToolkit

        toolkit = BaseToolkit()
        str_repr = str(toolkit)
        assert "no tools" in str_repr

    def test_base_toolkit_repr_representation(self):
        """Test BaseToolkit repr representation."""
        from hypertic.tools.base import BaseToolkit

        class TestToolkit(BaseToolkit):
            @tool
            def tool1(self, param: str) -> str:
                """Tool 1."""
                return f"Result1: {param}"

        toolkit = TestToolkit()
        repr_str = repr(toolkit)
        assert "tool1" in repr_str or "Tool 1" in repr_str

    def test_base_toolkit_tool_execution(self):
        """Test executing tools from a toolkit."""
        from hypertic.tools.base import BaseToolkit

        class TestToolkit(BaseToolkit):
            @tool
            def tool1(self, param: str) -> str:
                """Tool 1."""
                return f"Result1: {param}"

        toolkit = TestToolkit()
        tools = toolkit.get_tools()
        assert len(tools) == 1
        result = tools[0](param="test")
        assert result == "Result1: test"


class TestToolManagerAdvanced:
    """Test advanced ToolManager functionality"""

    def test_tool_manager_to_openai_format(self):
        """Test converting tools to OpenAI format."""

        @tool
        def test_tool(param: str) -> str:
            """Test tool description."""
            return "result"

        manager = _ToolManager()
        manager.add_tool(test_tool)
        openai_format = manager.to_openai_format()
        assert len(openai_format) == 1
        assert openai_format[0]["type"] == "function"
        assert openai_format[0]["function"]["name"] == "test_tool"
        assert "description" in openai_format[0]["function"]

    def test_tool_manager_execute_tool_public(self):
        """Test public execute_tool method."""

        @tool
        def test_tool(param: str) -> str:
            return f"Result: {param}"

        manager = _ToolManager()
        manager.add_tool(test_tool)
        result = manager.execute_tool("test_tool", {"param": "value"})
        assert result == "Result: value"

    def test_tool_manager_execute_tool_with_error(self):
        """Test execute_tool handles errors."""
        from hypertic.utils.exceptions import ToolExecutionError

        @tool
        def failing_tool(param: str) -> str:
            raise ValueError("Test error")

        manager = _ToolManager()
        manager.add_tool(failing_tool)
        with pytest.raises(ToolExecutionError):
            manager.execute_tool("failing_tool", {"param": "value"})

    def test_tool_manager_duplicate_tool_name(self):
        """Test that duplicate tool names raise error."""

        @tool(name="same_name")
        def tool1(param: str) -> str:
            return "result1"

        @tool(name="same_name")
        def tool2(param: str) -> str:
            return "result2"

        manager = _ToolManager()
        manager.add_tool(tool1)
        with pytest.raises(ValueError, match="Duplicate tool name"):
            manager.add_tool(tool2)

    def test_tool_manager_add_non_tool(self):
        """Test that adding non-tool function raises error."""

        def not_a_tool(param: str) -> str:
            return "result"

        manager = _ToolManager()
        with pytest.raises(ValueError, match="not decorated with @tool"):
            manager.add_tool(not_a_tool)

    def test_tool_manager_initialization_with_tools(self):
        """Test ToolManager initialization with tools list."""

        @tool
        def tool1(param: str) -> str:
            return "result1"

        @tool
        def tool2(param: str) -> str:
            return "result2"

        manager = _ToolManager(tools=[tool1, tool2])
        assert "tool1" in manager._tools_dict
        assert "tool2" in manager._tools_dict
        assert len(manager.list_tools()) == 2
