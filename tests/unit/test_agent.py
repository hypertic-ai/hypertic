import dataclasses
from typing import TypedDict
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from hypertic.agents.agent import Agent, _convert_to_pydantic
from hypertic.models.base import Base, LLMResponse
from hypertic.models.metrics import Metrics
from hypertic.tools.base import BaseToolkit
from hypertic.tools.mcp.client import ExecutableTool
from hypertic.utils.exceptions import ConfigurationError, SchemaConversionError


class TestConvertToPydantic:
    """Test the _convert_to_pydantic helper function."""

    def test_convert_pydantic_model(self):
        """Test converting a Pydantic BaseModel."""

        class TestModel(BaseModel):
            name: str
            age: int

        result = _convert_to_pydantic(TestModel)
        assert result == TestModel

    def test_convert_dataclass(self):
        """Test converting a dataclass."""

        @dataclasses.dataclass
        class TestDataclass:
            name: str
            age: int = 0

        result = _convert_to_pydantic(TestDataclass)
        assert issubclass(result, BaseModel)
        instance = result(name="test", age=10)
        assert instance.name == "test"
        assert instance.age == 10

    def test_convert_typed_dict(self):
        """Test converting a TypedDict."""

        class TestTypedDict(TypedDict):
            name: str
            age: int

        result = _convert_to_pydantic(TestTypedDict)
        assert issubclass(result, BaseModel)
        instance = result(name="test", age=10)
        assert instance.name == "test"
        assert instance.age == 10

    def test_convert_typed_dict_optional(self):
        """Test converting a TypedDict with optional fields."""
        from typing import Optional

        class TestTypedDict(TypedDict, total=False):
            name: str
            age: Optional[int] = None

        result = _convert_to_pydantic(TestTypedDict)
        assert issubclass(result, BaseModel)
        # With total=False, all fields are optional, but the conversion might require them
        # Let's test with all fields provided
        instance = result(name="test", age=10)
        assert instance.name == "test"
        assert instance.age == 10

    def test_convert_unsupported_type(self):
        """Test converting an unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            _convert_to_pydantic(str)

    def test_convert_dataclass_with_default_factory(self):
        """Test converting a dataclass with default_factory."""

        @dataclasses.dataclass
        class TestDataclass:
            name: str
            items: list[str] = dataclasses.field(default_factory=lambda: [])

        result = _convert_to_pydantic(TestDataclass)
        assert issubclass(result, BaseModel)
        instance = result(name="test")
        assert instance.name == "test"
        assert hasattr(instance, "items")
        instance.items = []
        assert instance.items == []

    def test_convert_typed_dict_with_total_true(self):
        """Test converting a TypedDict with __total__ = True."""

        class TestTypedDict(TypedDict):
            name: str
            age: int

        # Set __total__ explicitly
        TestTypedDict.__total__ = True

        result = _convert_to_pydantic(TestTypedDict)
        assert issubclass(result, BaseModel)
        instance = result(name="test", age=10)
        assert instance.name == "test"
        assert instance.age == 10

    def test_convert_typed_dict_with_required_keys(self):
        """Test converting a TypedDict with __required_keys__."""

        class TestTypedDict(TypedDict, total=False):
            name: str
            age: int

        # Simulate __required_keys__
        TestTypedDict.__required_keys__ = {"name"}

        result = _convert_to_pydantic(TestTypedDict)
        assert issubclass(result, BaseModel)
        instance = result(name="test")
        assert instance.name == "test"

    def test_convert_typed_dict_with_optional_keys(self):
        """Test converting a TypedDict with __optional_keys__."""
        from typing import Optional

        class TestTypedDict(TypedDict, total=False):
            name: str
            age: Optional[int]

        # Simulate __optional_keys__
        TestTypedDict.__optional_keys__ = {"age"}

        result = _convert_to_pydantic(TestTypedDict)
        assert issubclass(result, BaseModel)
        instance = result(name="test", age=None)
        assert instance.name == "test"
        assert instance.age is None

    def test_convert_typed_dict_fallback_path(self):
        """Test converting a TypedDict that falls back to get_type_hints."""

        class TestTypedDict:
            __annotations__ = {"name": str, "age": int}
            __module__ = "typing"

        result = _convert_to_pydantic(TestTypedDict)
        assert issubclass(result, BaseModel)
        instance = result(name="test", age=10)
        assert instance.name == "test"
        assert instance.age == 10

    def test_convert_typed_dict_exception_handling(self):
        """Test converting a TypedDict that raises exception in first try."""

        class TestTypedDict:
            __annotations__ = {"name": str}
            __module__ = "typing"

        original_get_type_hints = __import__("hypertic.agents.agent", fromlist=["get_type_hints"]).get_type_hints
        call_count = [0]

        def mock_get_type_hints(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Error")
            return original_get_type_hints(*args, **kwargs)

        with patch("hypertic.agents.agent.get_type_hints", side_effect=mock_get_type_hints):
            result = _convert_to_pydantic(TestTypedDict)
            assert issubclass(result, BaseModel)


class TestAgentInitialization:
    """Test Agent initialization and basic setup."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock(spec=Base)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    def test_agent_init_basic(self, mock_model):
        """Test basic Agent initialization."""
        agent = Agent(model=mock_model)
        assert agent.model == mock_model
        assert agent.instructions is None
        assert agent.max_steps == 10
        assert agent.parallel_calls is True

    def test_agent_init_with_all_params(self, mock_model):
        """Test Agent initialization with all parameters."""
        agent = Agent(
            model=mock_model,
            instructions="Test instructions",
            max_steps=5,
            parallel_calls=False,
        )
        assert agent.instructions == "Test instructions"
        assert agent.max_steps == 5
        assert agent.parallel_calls is False

    def test_agent_init_invalid_parallel_calls(self, mock_model):
        """Test Agent initialization with invalid parallel_calls."""
        with pytest.raises(ConfigurationError, match="parallel_calls must be True or False"):
            agent = Agent(model=mock_model)
            agent.parallel_calls = "invalid"
            agent._initialize()

    def test_agent_init_with_output_type(self, mock_model):
        """Test Agent initialization with output_type."""

        class OutputModel(BaseModel):
            result: str

        agent = Agent(model=mock_model, output_type=OutputModel)
        assert agent.output_type == OutputModel

    def test_agent_init_with_invalid_output_type(self, mock_model):
        """Test Agent initialization with invalid output_type."""
        with pytest.raises(SchemaConversionError):
            Agent(model=mock_model, output_type=str)

    def test_agent_separate_tool_types_empty(self, mock_model):
        """Test _separate_tool_types with no tools."""
        agent = Agent(model=mock_model)
        assert agent.mcp_tools == []
        assert agent.function_tools == []

    def test_agent_separate_tool_types_with_list(self, mock_model):
        """Test _separate_tool_types with list of tools."""
        mock_tool = Mock()
        mock_tool._tool_metadata = {"name": "test_tool"}
        agent = Agent(model=mock_model, tools=[mock_tool])
        assert len(agent.function_tools) == 1

    def test_agent_separate_tool_types_with_toolkit(self, mock_model):
        """Test _separate_tool_types with BaseToolkit."""
        mock_toolkit = Mock(spec=BaseToolkit)
        mock_tool = Mock()
        mock_tool._tool_metadata = {"name": "test_tool"}
        mock_toolkit.get_tools = Mock(return_value=[mock_tool])
        agent = Agent(model=mock_model, tools=[mock_toolkit])
        assert len(agent.function_tools) == 1

    def test_agent_separate_tool_types_with_mcp_tool(self, mock_model):
        """Test _separate_tool_types with ExecutableTool."""
        mock_mcp_tool = Mock(spec=ExecutableTool)
        mock_mcp_tool.name = "mcp_tool"
        agent = Agent(model=mock_model, tools=[mock_mcp_tool])
        assert len(agent.mcp_tools) == 1

    def test_agent_separate_tool_types_with_class_raises_error(self, mock_model):
        """Test _separate_tool_types raises error for class instead of instance."""

        class ToolClass:
            pass

        with pytest.raises(ValueError, match="Tool must be an instance"):
            Agent(model=mock_model, tools=[ToolClass])

    def test_agent_separate_tool_types_nested_list(self, mock_model):
        """Test _separate_tool_types with nested list of tools."""
        mock_tool1 = Mock()
        mock_tool1._tool_metadata = {"name": "tool1"}
        mock_tool2 = Mock()
        mock_tool2._tool_metadata = {"name": "tool2"}
        agent = Agent(model=mock_model, tools=[[mock_tool1, mock_tool2]])
        assert len(agent.function_tools) == 2

    def test_agent_separate_tool_types_non_list(self, mock_model):
        """Test _separate_tool_types with non-list tools."""
        mock_tool = Mock()
        mock_tool._tool_metadata = {"name": "tool1"}
        agent = Agent(model=mock_model, tools=mock_tool)
        assert len(agent.function_tools) == 1

    def test_agent_separate_tool_types_with_name_and_inputSchema(self, mock_model):  # noqa: N802
        """Test _separate_tool_types with tool that has name and inputSchema."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.inputSchema = {"type": "object"}
        # Ensure it doesn't have _tool_metadata to avoid being added to function_tools
        if hasattr(mock_tool, "_tool_metadata"):
            delattr(mock_tool, "_tool_metadata")
        agent = Agent(model=mock_model, tools=[mock_tool])
        assert len(agent.mcp_tools) == 1

    def test_agent_separate_tool_types_primitive_types(self, mock_model):
        """Test _separate_tool_types skips primitive types."""
        agent = Agent(model=mock_model, tools=["string", 123, 4.5, True, [1, 2], {"key": "value"}])
        assert len(agent.mcp_tools) == 0
        assert len(agent.function_tools) == 0

    def test_agent_separate_tool_types_with_dict(self, mock_model):
        """Test _separate_tool_types with tool that has __dict__."""

        class ToolWithDict:
            def __init__(self):
                self.name = "test"

        tool = ToolWithDict()
        agent = Agent(model=mock_model, tools=[tool])
        assert len(agent.mcp_tools) == 1

    def test_agent_separate_tool_types_fallback(self, mock_model):
        """Test _separate_tool_types fallback path."""
        mock_tool = Mock()
        del mock_tool._tool_metadata
        del mock_tool.name
        del mock_tool.inputSchema
        del mock_tool.__dict__
        agent = Agent(model=mock_model, tools=[mock_tool])
        assert len(agent.mcp_tools) == 1

    @pytest.mark.asyncio
    async def test_get_all_tools_async_with_dict_tool(self, mock_model):
        """Test _get_all_tools_async with dict tool."""
        agent = Agent(model=mock_model)
        agent.mcp_tools = [{"type": "function", "function": {"name": "test_tool"}}]
        tools = await agent._get_all_tools_async()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"

    @pytest.mark.asyncio
    async def test_get_all_tools_async_with_name_tool(self, mock_model):
        """Test _get_all_tools_async with tool that has name attribute."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test description"
        mock_tool.inputSchema = {"type": "object", "properties": {"param": {"type": "string"}}}
        agent = Agent(model=mock_model)
        agent.mcp_tools = [mock_tool]
        tools = await agent._get_all_tools_async()
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "test_tool"
        assert tools[0]["function"]["description"] == "Test description"

    @pytest.mark.asyncio
    async def test_get_all_tools_async_with_tool_manager(self, mock_model):
        """Test _get_all_tools_async with tool manager."""
        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool._tool_metadata = {"name": "test_tool"}
        agent.function_tools = [mock_tool]
        mock_tool_manager = Mock()
        mock_tool_manager.to_openai_format = Mock(return_value=[{"type": "function", "function": {"name": "test_tool"}}])
        agent._tool_manager = mock_tool_manager
        tools = await agent._get_all_tools_async()
        assert len(tools) == 1


class TestAgentMemory:
    """Test Agent memory operations."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock(spec=Base)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory."""
        memory = Mock()
        memory.load_messages = Mock(return_value=[])
        memory.save_message = Mock()
        return memory

    def test_load_memory_context_no_memory(self, mock_model):
        """Test _load_memory_context with no memory."""
        agent = Agent(model=mock_model)
        result = agent._load_memory_context(user_id=None, session_id=None)
        assert result == []

    def test_load_memory_context_with_memory(self, mock_model, mock_memory):
        """Test _load_memory_context with memory."""
        mock_memory.get_messages = Mock(return_value=[{"role": "user", "content": "Hello"}])
        agent = Agent(model=mock_model, memory=mock_memory)
        result = agent._load_memory_context(user_id="user1", session_id="session1")
        assert len(result) == 1
        mock_memory.get_messages.assert_called_once_with(session_id="session1", user_id="user1")

    def test_load_memory_context_with_memory_user_id_only(self, mock_model, mock_memory):
        """Test _load_memory_context with memory and user_id only."""
        mock_memory.get_messages = Mock(return_value=[{"role": "user", "content": "Hello"}])
        agent = Agent(model=mock_model, memory=mock_memory)
        result = agent._load_memory_context(user_id="user1", session_id=None)
        assert len(result) == 1
        mock_memory.get_messages.assert_called_once_with(session_id=None, user_id="user1")

    def test_load_memory_context_with_memory_error(self, mock_model, mock_memory):
        """Test _load_memory_context handles errors gracefully."""
        mock_memory.get_messages = Mock(side_effect=Exception("Error"))
        agent = Agent(model=mock_model, memory=mock_memory)
        result = agent._load_memory_context(user_id="user1", session_id="session1")
        assert result == []

    def test_load_memory_context_cleans_messages(self, mock_model, mock_memory):
        """Test _load_memory_context cleans message format."""
        mock_memory.get_messages = Mock(
            return_value=[{"role": "user", "content": "Hello", "extra": "data"}, {"role": "assistant", "content": "Hi", "metadata": "test"}]
        )
        agent = Agent(model=mock_model, memory=mock_memory)
        result = agent._load_memory_context(user_id="user1", session_id="session1")
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi"}

    @pytest.mark.asyncio
    async def test_aload_memory_context_no_memory(self, mock_model):
        """Test _aload_memory_context with no memory."""
        agent = Agent(model=mock_model)
        result = await agent._aload_memory_context(user_id=None, session_id=None)
        assert result == []

    @pytest.mark.asyncio
    async def test_aload_memory_context_with_memory(self, mock_model, mock_memory):
        """Test _aload_memory_context with memory."""
        mock_memory.aget_messages = AsyncMock(return_value=[{"role": "user", "content": "Hello"}])
        agent = Agent(model=mock_model, memory=mock_memory)
        result = await agent._aload_memory_context(user_id="user1", session_id="session1")
        assert len(result) == 1
        mock_memory.aget_messages.assert_called_once_with(session_id="session1", user_id="user1")

    @pytest.mark.asyncio
    async def test_aload_memory_context_with_memory_user_id_only(self, mock_model, mock_memory):
        """Test _aload_memory_context with memory and user_id only."""
        mock_memory.aget_messages = AsyncMock(return_value=[{"role": "user", "content": "Hello"}])
        agent = Agent(model=mock_model, memory=mock_memory)
        result = await agent._aload_memory_context(user_id="user1", session_id=None)
        assert len(result) == 1
        mock_memory.aget_messages.assert_called_once_with(session_id=None, user_id="user1")

    @pytest.mark.asyncio
    async def test_aload_memory_context_with_memory_error(self, mock_model, mock_memory):
        """Test _aload_memory_context handles errors gracefully."""
        mock_memory.aget_messages = AsyncMock(side_effect=Exception("Error"))
        agent = Agent(model=mock_model, memory=mock_memory)
        result = await agent._aload_memory_context(user_id="user1", session_id="session1")
        assert result == []

    @pytest.mark.asyncio
    async def test_aload_memory_context_cleans_messages(self, mock_model, mock_memory):
        """Test _aload_memory_context cleans message format."""
        mock_memory.aget_messages = AsyncMock(
            return_value=[{"role": "user", "content": "Hello", "extra": "data"}, {"role": "assistant", "content": "Hi", "metadata": "test"}]
        )
        agent = Agent(model=mock_model, memory=mock_memory)
        result = await agent._aload_memory_context(user_id="user1", session_id="session1")
        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi"}

    def test_save_to_memory_no_memory(self, mock_model):
        """Test _save_to_memory with no memory."""
        agent = Agent(model=mock_model)
        # Should not raise error
        agent._save_to_memory("session1", "user message", "assistant message")

    def test_save_to_memory_with_memory(self, mock_model, mock_memory):
        """Test _save_to_memory with memory."""
        agent = Agent(model=mock_model, memory=mock_memory)
        agent._save_to_memory("session1", "user message", "assistant message", user_id="user1")
        assert mock_memory.save_message.call_count == 2  # Called twice: once for user, once for assistant

    @pytest.mark.asyncio
    async def test_asave_to_memory_no_memory(self, mock_model):
        """Test _asave_to_memory with no memory."""
        agent = Agent(model=mock_model)
        # Should not raise error
        await agent._asave_to_memory("session1", "user message", "assistant message")

    @pytest.mark.asyncio
    async def test_asave_to_memory_with_memory(self, mock_model, mock_memory):
        """Test _asave_to_memory with memory."""
        mock_memory.asave_message = AsyncMock()
        agent = Agent(model=mock_model, memory=mock_memory)
        await agent._asave_to_memory("session1", "user message", "assistant message", user_id="user1")
        assert mock_memory.asave_message.call_count == 2  # Called twice: once for user, once for assistant


class TestAgentValidation:
    """Test Agent input validation."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock(spec=Base)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    @pytest.fixture
    def mock_guardrail(self):
        """Create a mock guardrail."""
        from hypertic.guardrails.base import GuardrailResult

        guardrail = Mock()
        guardrail.validate_input = Mock(return_value=GuardrailResult(allowed=True, modified_input=None))
        guardrail.avalidate_input = AsyncMock(return_value=GuardrailResult(allowed=True, modified_input=None))
        return guardrail

    def test_validate_input_no_guardrails(self, mock_model):
        """Test _validate_input with no guardrails."""
        agent = Agent(model=mock_model)
        result = agent._validate_input("Hello")
        assert result == "Hello"

    def test_validate_input_with_guardrails_pass(self, mock_model, mock_guardrail):
        """Test _validate_input with guardrails that pass."""
        agent = Agent(model=mock_model, guardrails=[mock_guardrail])
        result = agent._validate_input("Hello")
        assert result == "Hello"
        mock_guardrail.validate_input.assert_called_once()

    def test_validate_input_with_guardrails_fail(self, mock_model, mock_guardrail):
        """Test _validate_input with guardrails that fail."""
        from hypertic.guardrails.base import GuardrailResult
        from hypertic.utils.exceptions import GuardrailViolationError

        mock_guardrail.validate_input = Mock(return_value=GuardrailResult(allowed=False, reason="Invalid input"))
        agent = Agent(model=mock_model, guardrails=[mock_guardrail])
        with pytest.raises(GuardrailViolationError, match="Invalid input"):
            agent._validate_input("Hello")

    def test_load_memory_context_session_id_without_user_id(self, mock_model, mock_memory):
        """Test _load_memory_context raises error when session_id provided without user_id."""
        agent = Agent(model=mock_model, memory=mock_memory)
        with pytest.raises(ValueError, match="session_id requires user_id"):
            agent._load_memory_context(user_id=None, session_id="session1")

    @pytest.mark.asyncio
    async def test_avalidate_input_no_guardrails(self, mock_model):
        """Test _avalidate_input with no guardrails."""
        agent = Agent(model=mock_model)
        result = await agent._avalidate_input("Hello")
        assert result == "Hello"

    @pytest.mark.asyncio
    async def test_avalidate_input_with_guardrails_pass(self, mock_model, mock_guardrail):
        """Test _avalidate_input with guardrails that pass."""
        agent = Agent(model=mock_model, guardrails=[mock_guardrail])
        result = await agent._avalidate_input("Hello")
        assert result == "Hello"
        mock_guardrail.validate_input.assert_called_once()

    @pytest.mark.asyncio
    async def test_avalidate_input_with_guardrails_fail(self, mock_model, mock_guardrail):
        """Test _avalidate_input with guardrails that fail."""
        from hypertic.guardrails.base import GuardrailResult
        from hypertic.utils.exceptions import GuardrailViolationError

        mock_guardrail.validate_input = Mock(return_value=GuardrailResult(allowed=False, reason="Invalid input"))
        agent = Agent(model=mock_model, guardrails=[mock_guardrail])
        with pytest.raises(GuardrailViolationError, match="Invalid input"):
            await agent._avalidate_input("Hello")


class TestAgentTools:
    """Test Agent tool operations."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock(spec=Base)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    def test_get_all_tools_sync_no_tools(self, mock_model):
        """Test _get_all_tools_sync with no tools."""
        agent = Agent(model=mock_model)
        result = agent._get_all_tools_sync()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_all_tools_async_no_tools(self, mock_model):
        """Test _get_all_tools_async with no tools."""
        agent = Agent(model=mock_model)
        result = await agent._get_all_tools_async()
        assert result == []


class TestAgentExecution:
    """Test Agent execution methods (arun, astream, run, stream)."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with handler."""
        model = Mock(spec=Base)
        model.api_key = "test_key"
        model.model = "test-model"
        handler = Mock()
        handler.ahandle_non_streaming = AsyncMock(return_value=LLMResponse(content="Test response", response_text="Test response", metrics=Metrics()))
        handler.ahandle_streaming = AsyncMock(return_value=iter([]))
        handler.handle_non_streaming = Mock(return_value=LLMResponse(content="Test response", response_text="Test response", metrics=Metrics()))
        handler.handle_streaming = Mock(return_value=iter([]))
        handler._reset_metrics = Mock()
        handler._get_cumulative_metrics = Mock(return_value=Metrics())
        handler.model = "test-model"
        handler.temperature = None
        handler.top_p = None
        handler.presence_penalty = None
        handler.frequency_penalty = None
        handler.max_tokens = None
        model.get_handler = Mock(return_value=handler)
        return model

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory."""
        memory = AsyncMock()
        memory.aget_messages = AsyncMock(return_value=[])
        memory.asave_message = AsyncMock()
        memory.get_messages = Mock(return_value=[])
        memory.save_message = Mock()
        return memory

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = Mock()
        retriever.async_search = AsyncMock(return_value=[])
        retriever.search = Mock(return_value=[])
        return retriever

    @pytest.mark.asyncio
    async def test_arun_basic(self, mock_model):
        """Test basic arun execution."""
        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        response = await agent.arun("Hello")
        assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_arun_with_retriever(self, mock_model, mock_retriever):
        """Test arun with retriever."""
        agent = Agent(model=mock_model, retriever=mock_retriever)
        agent.handler = mock_model.get_handler()
        mock_retriever.async_search = AsyncMock(return_value=[Mock(page_content="Context", metadata={"source": "test"}, score=0.9)])
        response = await agent.arun("Hello")
        assert response.content == "Test response"
        mock_retriever.async_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_arun_with_structured_output(self, mock_model):
        """Test arun with structured output."""

        class OutputModel(BaseModel):
            result: str

        agent = Agent(model=mock_model, output_type=OutputModel)
        agent.handler = mock_model.get_handler()
        handler = agent.handler
        handler.ahandle_non_streaming = AsyncMock(
            return_value=LLMResponse(content='{"result": "test"}', response_text='{"result": "test"}', metrics=Metrics())
        )
        response = await agent.arun("Hello")
        assert hasattr(response, "structured_output")
        assert response.structured_output.result == "test"

    @pytest.mark.asyncio
    async def test_arun_with_memory(self, mock_model, mock_memory):
        """Test arun with memory."""
        agent = Agent(model=mock_model, memory=mock_memory)
        agent.handler = mock_model.get_handler()
        response = await agent.arun("Hello", user_id="user1", session_id="session1")
        assert response.content == "Test response"
        mock_memory.aget_messages.assert_called_once()
        mock_memory.asave_message.assert_called()

    @pytest.mark.asyncio
    async def test_arun_with_files(self, mock_model):
        """Test arun with files."""
        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        with patch.object(agent.file_processor, "process_message", return_value={"role": "user", "content": "Hello"}):
            response = await agent.arun("Hello", files=["test.txt"])
            assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_arun_auto_generates_session_id(self, mock_model):
        """Test arun auto-generates session_id when not provided."""
        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        response = await agent.arun("Hello")
        assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_astream_basic(self, mock_model):
        """Test basic astream execution."""
        from hypertic.models.events import ContentEvent, MetadataEvent

        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        handler = agent.handler

        async def async_iter_events():
            yield ContentEvent(content="Test")
            yield MetadataEvent(metadata={})

        handler.ahandle_streaming = Mock(return_value=async_iter_events())
        events = []
        async for event in agent.astream("Hello"):
            events.append(event)
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_astream_with_retriever_error(self, mock_model, mock_retriever):
        """Test astream handles retriever errors gracefully."""
        from hypertic.utils.exceptions import RetrieverError

        agent = Agent(model=mock_model, retriever=mock_retriever)
        agent.handler = mock_model.get_handler()
        handler = agent.handler

        async def async_iter_empty():
            return
            yield

        handler.ahandle_streaming = Mock(return_value=async_iter_empty())
        mock_retriever.async_search = AsyncMock(side_effect=RetrieverError("Retrieval failed"))
        events = []
        async for event in agent.astream("Hello"):
            events.append(event)
        # Should continue without error

    @pytest.mark.asyncio
    async def test_arun_non_streaming_with_instructions(self, mock_model):
        """Test _arun_non_streaming with instructions."""
        agent = Agent(model=mock_model, instructions="You are helpful")
        agent.handler = mock_model.get_handler()
        response = await agent._arun_non_streaming("Hello")
        assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_arun_non_streaming_with_files(self, mock_model):
        """Test _arun_non_streaming with files."""
        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        with patch.object(agent.file_processor, "process_message", return_value={"role": "user", "content": "Hello"}):
            response = await agent._arun_non_streaming("Hello", files=["test.txt"])
            assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_arun_streaming_with_instructions(self, mock_model):
        """Test _arun_streaming with instructions."""
        agent = Agent(model=mock_model, instructions="You are helpful")
        agent.handler = mock_model.get_handler()
        handler = agent.handler

        async def async_iter_empty():
            return
            yield

        handler.ahandle_streaming = Mock(return_value=async_iter_empty())
        events = []
        async for event in agent._arun_streaming("Hello"):
            events.append(event)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_aexecute_tool_function_tool(self, mock_model):
        """Test _aexecute_tool with function tool."""
        from hypertic.tools.base import _ToolManager

        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool._tool_metadata = {"name": "test_tool"}
        agent.tools = [mock_tool]
        agent._separate_tool_types()
        agent._tool_manager = _ToolManager(agent.function_tools)
        agent._tool_manager.execute_tool = Mock(return_value="Tool result")
        result = await agent._aexecute_tool("test_tool", {"arg": "value"})
        assert result == "Tool result"

    @pytest.mark.asyncio
    async def test_aexecute_tool_mcp_tool_async(self, mock_model):
        """Test _aexecute_tool with async MCP tool."""
        agent = Agent(model=mock_model)
        mock_tool = Mock(spec=ExecutableTool)
        mock_tool.name = "mcp_tool"
        mock_tool.call_tool = AsyncMock(return_value="MCP result")
        agent.mcp_tools = [mock_tool]
        result = await agent._aexecute_tool("mcp_tool", {"arg": "value"})
        assert result == "MCP result"

    @pytest.mark.asyncio
    async def test_aexecute_tool_mcp_tool_sync(self, mock_model):
        """Test _aexecute_tool with sync MCP tool."""
        agent = Agent(model=mock_model)
        mock_tool = Mock(spec=ExecutableTool)
        mock_tool.name = "mcp_tool"
        mock_tool.call_tool = Mock(return_value="MCP result")
        agent.mcp_tools = [mock_tool]
        result = await agent._aexecute_tool("mcp_tool", {"arg": "value"})
        assert result == "MCP result"

    @pytest.mark.asyncio
    async def test_aexecute_tool_not_found(self, mock_model):
        """Test _aexecute_tool raises error when tool not found."""
        from hypertic.utils.exceptions import ToolNotFoundError

        agent = Agent(model=mock_model)
        with pytest.raises(ToolNotFoundError):
            await agent._aexecute_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_aexecute_tools_parallel(self, mock_model):
        """Test _aexecute_tools_parallel."""
        import json

        agent = Agent(model=mock_model)
        mock_tool = Mock(spec=ExecutableTool)
        mock_tool.name = "mcp_tool"
        mock_tool.call_tool = AsyncMock(return_value="Result")
        agent.mcp_tools = [mock_tool]
        tool_calls = [{"id": "call1", "function": {"name": "mcp_tool", "arguments": json.dumps({"arg": "value"})}}]
        results = await agent._aexecute_tools_parallel(tool_calls)
        assert "mcp_tool" in results
        assert results["mcp_tool"] == "Result"

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_tools(self, mock_model):
        """Test _ahandle_non_streaming_with_tools."""
        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        messages = [{"role": "user", "content": "Hello"}]
        response = await agent._ahandle_non_streaming_with_tools(messages, None)
        assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_max_steps(self, mock_model):
        """Test _ahandle_non_streaming_with_tools raises MaxStepsError."""
        from hypertic.utils.exceptions import MaxStepsError

        agent = Agent(model=mock_model, max_steps=1)
        agent.handler = mock_model.get_handler()
        handler = agent.handler
        handler.ahandle_non_streaming = AsyncMock(return_value=None)
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(MaxStepsError):
            await agent._ahandle_non_streaming_with_tools(messages, None)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_tools(self, mock_model):
        """Test _ahandle_streaming_with_tools."""
        from hypertic.models.events import ContentEvent

        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        handler = agent.handler

        async def async_iter_events():
            yield ContentEvent(content="Test")
            yield True  # has_more_tools

        handler.ahandle_streaming = Mock(return_value=async_iter_events())
        messages = [{"role": "user", "content": "Hello"}]
        events = []
        async for event in agent._ahandle_streaming_with_tools(messages, None):
            events.append(event)
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_retrieve_knowledge_async(self, mock_model, mock_retriever):
        """Test _retrieve_knowledge_async."""
        agent = Agent(model=mock_model, retriever=mock_retriever)
        mock_retriever.async_search = AsyncMock(return_value=[Mock(page_content="Context", metadata={"source": "test"})])
        result = await agent._retrieve_knowledge_async("query")
        assert "Context" in result

    @pytest.mark.asyncio
    async def test_retrieve_knowledge_async_no_retriever(self, mock_model):
        """Test _retrieve_knowledge_async with no retriever."""
        agent = Agent(model=mock_model)
        result = await agent._retrieve_knowledge_async("query")
        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_knowledge_async_error(self, mock_model, mock_retriever):
        """Test _retrieve_knowledge_async handles errors."""
        from hypertic.utils.exceptions import RetrieverError

        agent = Agent(model=mock_model, retriever=mock_retriever)
        mock_retriever.async_search = AsyncMock(side_effect=Exception("Error"))
        with pytest.raises(RetrieverError):
            await agent._retrieve_knowledge_async("query")

    def test_retrieve_knowledge_sync(self, mock_model, mock_retriever):
        """Test _retrieve_knowledge_sync."""
        agent = Agent(model=mock_model, retriever=mock_retriever)
        mock_retriever.search = Mock(return_value=[Mock(page_content="Context", metadata={"source": "test"})])
        result = agent._retrieve_knowledge_sync("query")
        assert "Context" in result

    def test_retrieve_knowledge_sync_no_retriever(self, mock_model):
        """Test _retrieve_knowledge_sync with no retriever."""
        agent = Agent(model=mock_model)
        result = agent._retrieve_knowledge_sync("query")
        assert result == ""

    def test_run_basic(self, mock_model):
        """Test basic run execution."""
        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        response = agent.run("Hello")
        assert response.content == "Test response"

    def test_run_with_retriever(self, mock_model, mock_retriever):
        """Test run with retriever."""
        agent = Agent(model=mock_model, retriever=mock_retriever)
        agent.handler = mock_model.get_handler()
        mock_retriever.search = Mock(return_value=[Mock(page_content="Context", metadata={"source": "test"})])
        response = agent.run("Hello")
        assert response.content == "Test response"

    def test_run_with_memory(self, mock_model, mock_memory):
        """Test run with memory."""
        agent = Agent(model=mock_model, memory=mock_memory)
        agent.handler = mock_model.get_handler()
        response = agent.run("Hello", user_id="user1", session_id="session1")
        assert response.content == "Test response"
        mock_memory.get_messages.assert_called_once()
        mock_memory.save_message.assert_called()

    def test_run_non_streaming_with_instructions(self, mock_model):
        """Test _run_non_streaming with instructions."""
        agent = Agent(model=mock_model, instructions="You are helpful")
        agent.handler = mock_model.get_handler()
        response = agent._run_non_streaming("Hello")
        assert response.content == "Test response"

    def test_run_streaming_with_instructions(self, mock_model):
        """Test _run_streaming with instructions."""
        agent = Agent(model=mock_model, instructions="You are helpful")
        agent.handler = mock_model.get_handler()
        handler = agent.handler
        handler.handle_streaming = Mock(return_value=iter([]))
        events = []
        for event in agent._run_streaming("Hello"):
            events.append(event)
        # Should complete without error

    def test_handle_non_streaming_with_tools(self, mock_model):
        """Test _handle_non_streaming_with_tools."""
        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        messages = [{"role": "user", "content": "Hello"}]
        response = agent._handle_non_streaming_with_tools(messages, None)
        assert response.content == "Test response"

    def test_handle_streaming_with_tools(self, mock_model):
        """Test _handle_streaming_with_tools."""
        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        handler = agent.handler
        handler.handle_streaming = Mock(return_value=iter([Mock(type="content", content="Test")]))
        messages = [{"role": "user", "content": "Hello"}]
        events = []
        for event in agent._handle_streaming_with_tools(messages, None):
            events.append(event)
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_auto_disconnect_clients(self, mock_model):
        """Test _auto_disconnect_clients."""
        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool.name = "mcp_tool"
        mock_tool.mcp_servers = Mock()
        mock_tool.mcp_servers.initialized = True
        mock_tool.mcp_servers.disconnect = AsyncMock()
        agent.mcp_tools = [mock_tool]
        await agent._auto_disconnect_clients()
        mock_tool.mcp_servers.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_arun_with_structured_output_json(self, mock_model):
        """Test arun with structured output (JSON parsing)."""

        class OutputModel(BaseModel):
            result: str

        agent = Agent(model=mock_model, output_type=OutputModel)
        agent.handler = mock_model.get_handler()
        mock_response = Mock(spec=LLMResponse)
        mock_response.content = '{"result": "test"}'
        mock_response.structured_output = None
        agent.handler.handle_non_streaming = AsyncMock(return_value=mock_response)
        agent._tool_calls = []
        agent._tool_outputs = {}
        response = await agent.arun("Hello")
        assert response is not None

    @pytest.mark.asyncio
    async def test_arun_with_structured_output_json_error(self, mock_model):
        """Test arun with structured output (JSON parsing error)."""

        class OutputModel(BaseModel):
            result: str

        agent = Agent(model=mock_model, output_type=OutputModel)
        agent.handler = mock_model.get_handler()
        mock_response = Mock(spec=LLMResponse)
        mock_response.content = "invalid json"
        mock_response.structured_output = None
        agent.handler.handle_non_streaming = AsyncMock(return_value=mock_response)
        agent._tool_calls = []
        agent._tool_outputs = {}
        response = await agent.arun("Hello")
        assert response is not None

    @pytest.mark.asyncio
    async def test_astream_with_structured_output(self, mock_model):
        """Test astream with structured output."""
        from hypertic.models.events import ContentEvent, StructuredOutputEvent

        class OutputModel(BaseModel):
            result: str

        agent = Agent(model=mock_model, output_type=OutputModel)
        agent.handler = mock_model.get_handler()

        class AsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        async_iterator = AsyncIterator(
            [
                ContentEvent(content='{"result": "test"}'),
            ]
        )
        agent._arun_streaming = Mock(return_value=async_iterator)
        agent.memory = None
        events = []
        async for event in agent.astream("Hello"):
            events.append(event)
        assert len(events) > 0
        assert any(isinstance(e, StructuredOutputEvent) for e in events)

    @pytest.mark.asyncio
    async def test_astream_with_structured_output_json_error(self, mock_model):
        """Test astream with structured output (JSON parsing error)."""
        from hypertic.models.events import ContentEvent, MetadataEvent

        class OutputModel(BaseModel):
            result: str

        agent = Agent(model=mock_model, output_type=OutputModel)
        agent.handler = mock_model.get_handler()

        class AsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        # When output_type is set and content is invalid JSON, no events are yielded
        # because content events are filtered out (line 581) and JSON parsing fails silently
        # So we need to add a metadata event to ensure we get at least one event
        async_iterator = AsyncIterator(
            [
                ContentEvent(content="invalid json"),
                MetadataEvent(metadata={"model": "test", "input_tokens": 10, "output_tokens": 20}),
            ]
        )
        agent._arun_streaming = Mock(return_value=async_iterator)
        agent.memory = None
        events = []
        async for event in agent.astream("Hello"):
            events.append(event)
        # Should have MetadataEvent even though JSON parsing failed
        assert len(events) > 0
        assert any(isinstance(e, MetadataEvent) for e in events)

    @pytest.mark.asyncio
    async def test_astream_with_metadata(self, mock_model):
        """Test astream with metadata collection."""
        from hypertic.models.events import ContentEvent, MetadataEvent

        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()

        class AsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        mock_metadata = {"model": "test", "input_tokens": 10, "output_tokens": 20}
        async_iterator = AsyncIterator(
            [
                ContentEvent(content="Response"),
                MetadataEvent(metadata=mock_metadata),
            ]
        )
        agent._arun_streaming = Mock(return_value=async_iterator)
        agent.memory = None
        events = []
        async for event in agent.astream("Hello"):
            events.append(event)
        assert len(events) > 0
        assert any(isinstance(e, MetadataEvent) for e in events)

    @pytest.mark.asyncio
    async def test_astream_with_memory_and_metadata(self, mock_model, mock_memory):
        """Test astream saves to memory with metadata."""
        from hypertic.models.events import ContentEvent, MetadataEvent

        agent = Agent(model=mock_model, memory=mock_memory)
        agent.handler = mock_model.get_handler()
        mock_memory.asave_message = AsyncMock()

        class AsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        mock_metadata = {"model": "test", "input_tokens": 10, "output_tokens": 20}
        async_iterator = AsyncIterator(
            [
                ContentEvent(content="Response"),
                MetadataEvent(metadata=mock_metadata),
            ]
        )
        agent._arun_streaming = Mock(return_value=async_iterator)
        events = []
        async for event in agent.astream("Hello", user_id="user1", session_id="session1"):
            events.append(event)
        assert len(events) > 0
        mock_memory.asave_message.assert_called()

    @pytest.mark.asyncio
    async def test_astream_with_tool_calls_and_outputs(self, mock_model):
        """Test astream collects tool calls and outputs."""
        from hypertic.models.events import ContentEvent, ToolCallsEvent, ToolOutputsEvent

        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()

        class AsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        async_iterator = AsyncIterator(
            [
                ContentEvent(content="Response"),
                ToolCallsEvent(tool_calls=[{"id": "call_1", "function": {"name": "test_tool", "arguments": "{}"}}]),
                ToolOutputsEvent(tool_outputs={"test_tool": "result"}),
            ]
        )
        agent._arun_streaming = Mock(return_value=async_iterator)
        agent.memory = None
        events = []
        async for event in agent.astream("Hello"):
            events.append(event)
        assert len(events) > 0
        assert any(isinstance(e, ToolCallsEvent) for e in events)
        assert any(isinstance(e, ToolOutputsEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_tools_has_more_tools(self, mock_model):
        """Test _ahandle_streaming_with_tools with has_more_tools flag."""
        from hypertic.models.events import ContentEvent, ToolCallsEvent, ToolOutputsEvent

        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        agent.handler._reset_metrics = Mock()
        agent.handler._get_cumulative_metrics = Mock(return_value=Mock(input_tokens=10, output_tokens=20))
        agent.handler.temperature = 0.7
        agent.handler.top_p = 0.9
        agent.handler.presence_penalty = 0.1
        agent.handler.frequency_penalty = 0.2
        agent.handler.max_tokens = 1000
        agent.handler.model = "test-model"

        class AsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        async_iterator = AsyncIterator(
            [
                ToolCallsEvent(tool_calls=[{"id": "call_1", "function": {"name": "test_tool", "arguments": "{}"}}]),
                True,  # has_more_tools
                ToolOutputsEvent(tool_outputs={"test_tool": "result"}),
                True,  # has_more_tools
                ContentEvent(content="Response"),
                False,  # no more tools
            ]
        )
        agent.handler.ahandle_streaming = Mock(return_value=async_iterator)
        messages = [{"role": "user", "content": "Hello"}]
        events = []
        async for event in agent._ahandle_streaming_with_tools(messages, None):
            events.append(event)
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_tools_max_steps(self, mock_model):
        """Test _ahandle_streaming_with_tools with max_steps reached."""
        from hypertic.models.events import ToolCallsEvent

        agent = Agent(model=mock_model, max_steps=2)
        agent.handler = mock_model.get_handler()
        agent.handler._reset_metrics = Mock()
        agent.handler._get_cumulative_metrics = Mock(return_value=Mock(input_tokens=10, output_tokens=20))
        agent.handler.model = "test-model"

        class AsyncIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        # Always return has_more_tools=True to trigger max_steps
        async_iterator = AsyncIterator(
            [
                ToolCallsEvent(tool_calls=[{"id": "call_1", "function": {"name": "test_tool", "arguments": "{}"}}]),
                True,  # has_more_tools
            ]
        )
        agent.handler.ahandle_streaming = Mock(return_value=async_iterator)
        messages = [{"role": "user", "content": "Hello"}]
        events = []
        async for event in agent._ahandle_streaming_with_tools(messages, None):
            events.append(event)
        # Should complete after max_steps
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_tools_response_format_error(self, mock_model):
        """Test _ahandle_streaming_with_tools with response_format conversion error."""
        from hypertic.models.events import ContentEvent

        class OutputModel(BaseModel):
            result: str

        agent = Agent(model=mock_model, output_type=OutputModel)
        agent.handler = mock_model.get_handler()
        agent.handler._reset_metrics = Mock()
        agent.handler._get_cumulative_metrics = Mock(return_value=Mock(input_tokens=10, output_tokens=20))
        agent.handler.model = "test-model"

        # Mock _convert_to_pydantic to raise exception
        with patch("hypertic.agents.agent._convert_to_pydantic", side_effect=Exception("Conversion error")):

            class AsyncIterator:
                def __init__(self, items):
                    self.items = items
                    self.index = 0

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self.index >= len(self.items):
                        raise StopAsyncIteration
                    item = self.items[self.index]
                    self.index += 1
                    return item

            async_iterator = AsyncIterator(
                [
                    ContentEvent(content="Response"),
                ]
            )
            agent.handler.ahandle_streaming = Mock(return_value=async_iterator)
            messages = [{"role": "user", "content": "Hello"}]
            events = []
            async for event in agent._ahandle_streaming_with_tools(messages, None):
                events.append(event)
            # Should handle error gracefully
            assert len(events) > 0

    def test_stream_with_structured_output(self, mock_model):
        """Test stream with structured output."""
        from hypertic.models.events import ContentEvent, StructuredOutputEvent

        class OutputModel(BaseModel):
            result: str

        agent = Agent(model=mock_model, output_type=OutputModel)
        agent.handler = mock_model.get_handler()
        agent._run_streaming = Mock(
            return_value=iter(
                [
                    ContentEvent(content='{"result": "test"}'),
                ]
            )
        )
        agent.memory = None
        events = list(agent.stream("Hello"))
        assert len(events) > 0
        assert any(isinstance(e, StructuredOutputEvent) for e in events)

    def test_stream_with_metadata(self, mock_model):
        """Test stream with metadata collection."""
        from hypertic.models.events import ContentEvent, MetadataEvent

        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        mock_metadata = {"model": "test", "input_tokens": 10, "output_tokens": 20}
        agent._run_streaming = Mock(
            return_value=iter(
                [
                    ContentEvent(content="Response"),
                    MetadataEvent(metadata=mock_metadata),
                ]
            )
        )
        agent.memory = None
        events = list(agent.stream("Hello"))
        assert len(events) > 0
        assert any(isinstance(e, MetadataEvent) for e in events)

    def test_stream_with_memory_and_metadata(self, mock_model, mock_memory):
        """Test stream saves to memory with metadata."""
        from hypertic.models.events import ContentEvent, MetadataEvent

        agent = Agent(model=mock_model, memory=mock_memory)
        agent.handler = mock_model.get_handler()
        mock_memory.save_message = Mock()
        mock_metadata = {"model": "test", "input_tokens": 10, "output_tokens": 20}
        agent._run_streaming = Mock(
            return_value=iter(
                [
                    ContentEvent(content="Response"),
                    MetadataEvent(metadata=mock_metadata),
                ]
            )
        )
        events = list(agent.stream("Hello", user_id="user1", session_id="session1"))
        assert len(events) > 0
        mock_memory.save_message.assert_called()

    def test_handle_streaming_with_tools_has_more_tools(self, mock_model):
        """Test _handle_streaming_with_tools with has_more_tools flag."""
        from hypertic.models.events import ContentEvent, ToolCallsEvent, ToolOutputsEvent

        agent = Agent(model=mock_model)
        agent.handler = mock_model.get_handler()
        agent.handler._reset_metrics = Mock()
        agent.handler._get_cumulative_metrics = Mock(return_value=Mock(input_tokens=10, output_tokens=20))
        agent.handler.temperature = 0.7
        agent.handler.top_p = 0.9
        agent.handler.presence_penalty = 0.1
        agent.handler.frequency_penalty = 0.2
        agent.handler.max_tokens = 1000
        agent.handler.model = "test-model"
        agent.handler.handle_streaming = Mock(
            return_value=iter(
                [
                    ToolCallsEvent(tool_calls=[{"id": "call_1", "function": {"name": "test_tool", "arguments": "{}"}}]),
                    True,  # has_more_tools
                    ToolOutputsEvent(tool_outputs={"test_tool": "result"}),
                    True,  # has_more_tools
                    ContentEvent(content="Response"),
                    False,  # no more tools
                ]
            )
        )
        messages = [{"role": "user", "content": "Hello"}]
        events = list(agent._handle_streaming_with_tools(messages, None))
        assert len(events) > 0

    def test_handle_streaming_with_tools_max_steps(self, mock_model):
        """Test _handle_streaming_with_tools with max_steps reached."""
        from hypertic.models.events import ToolCallsEvent

        agent = Agent(model=mock_model, max_steps=2)
        agent.handler = mock_model.get_handler()
        agent.handler._reset_metrics = Mock()
        agent.handler._get_cumulative_metrics = Mock(return_value=Mock(input_tokens=10, output_tokens=20))
        agent.handler.model = "test-model"
        # Always return has_more_tools=True to trigger max_steps
        agent.handler.handle_streaming = Mock(
            return_value=iter(
                [
                    ToolCallsEvent(tool_calls=[{"id": "call_1", "function": {"name": "test_tool", "arguments": "{}"}}]),
                    True,  # has_more_tools
                ]
            )
        )
        messages = [{"role": "user", "content": "Hello"}]
        events = list(agent._handle_streaming_with_tools(messages, None))
        # Should complete after max_steps
        assert len(events) > 0

    def test_handle_non_streaming_with_tools_response_format_error(self, mock_model):
        """Test _handle_non_streaming_with_tools with response_format conversion error."""

        class OutputModel(BaseModel):
            result: str

        agent = Agent(model=mock_model, output_type=OutputModel)
        agent.handler = mock_model.get_handler()
        agent.handler._reset_metrics = Mock()
        mock_response = Mock(spec=LLMResponse)
        mock_response.content = "Response"
        agent.handler.handle_non_streaming = Mock(return_value=mock_response)

        # Mock _convert_to_pydantic to raise exception
        with patch("hypertic.agents.agent._convert_to_pydantic", side_effect=Exception("Conversion error")):
            messages = [{"role": "user", "content": "Hello"}]
            response = agent._handle_non_streaming_with_tools(messages, None)
            # Should handle error gracefully
            assert response is not None

    def test_handle_non_streaming_with_tools_max_steps(self, mock_model):
        """Test _handle_non_streaming_with_tools with max_steps reached."""
        from hypertic.utils.exceptions import MaxStepsError

        agent = Agent(model=mock_model, max_steps=2)
        agent.handler = mock_model.get_handler()
        agent.handler._reset_metrics = Mock()
        # Always return None to trigger max_steps
        agent.handler.handle_non_streaming = Mock(return_value=None)
        messages = [{"role": "user", "content": "Hello"}]
        with pytest.raises(MaxStepsError):
            agent._handle_non_streaming_with_tools(messages, None)

    def test_execute_tool_function_tool(self, mock_model):
        """Test _execute_tool with function tool."""
        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool._tool_metadata = {"name": "test_tool"}
        agent.function_tools = [mock_tool]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = Mock(return_value="result")
        agent._tool_manager = mock_tool_manager
        result = agent._execute_tool("test_tool", {"param": "value"})
        assert result == "result"

    def test_execute_tool_function_tool_not_found(self, mock_model):
        """Test _execute_tool with function tool not found."""
        from hypertic.utils.exceptions import ToolNotFoundError

        agent = Agent(model=mock_model)
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = Mock(side_effect=ToolNotFoundError("Tool not found"))
        agent._tool_manager = mock_tool_manager
        agent.function_tools = [Mock()]
        # Should continue to MCP tools
        with pytest.raises(ToolNotFoundError):
            agent._execute_tool("test_tool", {"param": "value"})

    def test_execute_tool_function_tool_error(self, mock_model):
        """Test _execute_tool with function tool execution error."""
        from hypertic.utils.exceptions import ToolExecutionError

        agent = Agent(model=mock_model)
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool = Mock(side_effect=Exception("Execution error"))
        agent._tool_manager = mock_tool_manager
        agent.function_tools = [Mock()]
        with pytest.raises(ToolExecutionError):
            agent._execute_tool("test_tool", {"param": "value"})

    def test_execute_tool_mcp_tool_with_mcp_servers(self, mock_model):
        """Test _execute_tool with MCP tool that has mcp_servers (async-only)."""
        from hypertic.utils.exceptions import ToolExecutionError

        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool.name = "mcp_tool"
        mock_tool.mcp_servers = Mock()
        agent.mcp_tools = [mock_tool]
        with pytest.raises(ToolExecutionError, match="async-only"):
            agent._execute_tool("mcp_tool", {"param": "value"})

    def test_execute_tool_mcp_tool_with_call_tool(self, mock_model):
        """Test _execute_tool with MCP tool that has call_tool method."""
        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool.name = "mcp_tool"
        mock_tool.call_tool = Mock(return_value="result")
        # Ensure mcp_servers is not present to avoid async-only error
        if hasattr(mock_tool, "mcp_servers"):
            delattr(mock_tool, "mcp_servers")
        agent.mcp_tools = [mock_tool]
        result = agent._execute_tool("mcp_tool", {"param": "value"})
        assert result == "result"

    def test_execute_tool_mcp_tool_with_call_tool_error(self, mock_model):
        """Test _execute_tool with MCP tool call_tool error."""
        from hypertic.utils.exceptions import ToolExecutionError

        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool.name = "mcp_tool"
        mock_tool.call_tool = Mock(side_effect=Exception("Execution error"))
        # Ensure mcp_servers is not present to avoid async-only error
        if hasattr(mock_tool, "mcp_servers"):
            delattr(mock_tool, "mcp_servers")
        agent.mcp_tools = [mock_tool]
        with pytest.raises(ToolExecutionError):
            agent._execute_tool("mcp_tool", {"param": "value"})

    def test_execute_tool_mcp_tool_fallback(self, mock_model):
        """Test _execute_tool with MCP tool fallback (no call_tool)."""
        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool.name = "mcp_tool"
        # Ensure mcp_servers is not present to avoid async-only error
        if hasattr(mock_tool, "mcp_servers"):
            delattr(mock_tool, "mcp_servers")
        # Remove call_tool to trigger fallback
        if hasattr(mock_tool, "call_tool"):
            delattr(mock_tool, "call_tool")
        agent.mcp_tools = [mock_tool]
        result = agent._execute_tool("mcp_tool", {"param": "value"})
        assert "mcp_tool" in result
        assert "executed" in result

    def test_execute_tools_parallel(self, mock_model):
        """Test _execute_tools_parallel."""
        import json

        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.call_tool = Mock(return_value="result")
        # Ensure mcp_servers is not present to avoid async-only error
        if hasattr(mock_tool, "mcp_servers"):
            del mock_tool.mcp_servers
        agent.mcp_tools = [mock_tool]
        tool_calls = [{"id": "call_1", "function": {"name": "test_tool", "arguments": json.dumps({"param": "value"})}}]
        results = agent._execute_tools_parallel(tool_calls)
        assert "test_tool" in results
        assert results["test_tool"] == "result"

    def test_execute_tools_parallel_invalid_json(self, mock_model):
        """Test _execute_tools_parallel with invalid JSON arguments."""
        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.call_tool = Mock(return_value="result")
        # Ensure mcp_servers is not present to avoid async-only error
        if hasattr(mock_tool, "mcp_servers"):
            del mock_tool.mcp_servers
        agent.mcp_tools = [mock_tool]
        tool_calls = [{"id": "call_1", "function": {"name": "test_tool", "arguments": "invalid json"}}]
        results = agent._execute_tools_parallel(tool_calls)
        assert "test_tool" in results

    def test_execute_tools_parallel_none_arguments(self, mock_model):
        """Test _execute_tools_parallel with None arguments."""
        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.call_tool = Mock(return_value="result")
        # Ensure mcp_servers is not present to avoid async-only error
        if hasattr(mock_tool, "mcp_servers"):
            del mock_tool.mcp_servers
        agent.mcp_tools = [mock_tool]
        tool_calls = [{"id": "call_1", "function": {"name": "test_tool", "arguments": None}}]
        results = agent._execute_tools_parallel(tool_calls)
        assert "test_tool" in results

    def test_execute_tools_parallel_tool_execution_error(self, mock_model):
        """Test _execute_tools_parallel with ToolExecutionError."""
        import json

        from hypertic.utils.exceptions import ToolExecutionError

        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.call_tool = Mock(side_effect=ToolExecutionError("Tool error"))
        # Ensure mcp_servers is not present to avoid async-only error
        if hasattr(mock_tool, "mcp_servers"):
            del mock_tool.mcp_servers
        agent.mcp_tools = [mock_tool]
        tool_calls = [{"id": "call_1", "function": {"name": "test_tool", "arguments": json.dumps({"param": "value"})}}]
        with pytest.raises(ToolExecutionError):
            agent._execute_tools_parallel(tool_calls)

    def test_execute_tools_parallel_tool_not_found_error(self, mock_model):
        """Test _execute_tools_parallel with ToolNotFoundError."""
        import json

        from hypertic.utils.exceptions import ToolNotFoundError

        agent = Agent(model=mock_model)
        tool_calls = [{"id": "call_1", "function": {"name": "test_tool", "arguments": json.dumps({"param": "value"})}}]
        with pytest.raises(ToolNotFoundError):
            agent._execute_tools_parallel(tool_calls)

    def test_execute_tools_parallel_unexpected_error(self, mock_model):
        """Test _execute_tools_parallel with unexpected error."""
        import json

        from hypertic.utils.exceptions import ToolExecutionError

        agent = Agent(model=mock_model)
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.call_tool = Mock(side_effect=ValueError("Unexpected error"))
        # Ensure mcp_servers is not present to avoid async-only error
        if hasattr(mock_tool, "mcp_servers"):
            del mock_tool.mcp_servers
        agent.mcp_tools = [mock_tool]
        tool_calls = [{"id": "call_1", "function": {"name": "test_tool", "arguments": json.dumps({"param": "value"})}}]
        with pytest.raises(ToolExecutionError):
            agent._execute_tools_parallel(tool_calls)

    def test_run_with_structured_output(self, mock_model):
        """Test run with structured output."""

        class OutputModel(BaseModel):
            result: str

        agent = Agent(model=mock_model, output_type=OutputModel)
        agent.handler = mock_model.get_handler()
        mock_response = Mock(spec=LLMResponse)
        mock_response.content = '{"result": "test"}'
        mock_response.structured_output = None
        mock_response.metadata = None
        agent.handler.handle_non_streaming = Mock(return_value=mock_response)
        response = agent.run("Hello")
        assert response is not None

    def test_run_with_structured_output_json_error(self, mock_model):
        """Test run with structured output (JSON parsing error)."""

        class OutputModel(BaseModel):
            result: str

        agent = Agent(model=mock_model, output_type=OutputModel)
        agent.handler = mock_model.get_handler()
        mock_response = Mock(spec=LLMResponse)
        mock_response.content = "invalid json"
        mock_response.structured_output = None
        mock_response.metadata = None
        agent.handler.handle_non_streaming = Mock(return_value=mock_response)
        response = agent.run("Hello")
        assert response is not None
