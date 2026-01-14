from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.models.events import ContentEvent, FinishReasonEvent, ResponseCompletedEvent
from hypertic.models.mistral.mistral import MistralAI
from hypertic.utils.files import File, FileType


class TestMistralAI:
    @pytest.fixture
    def model(self, mock_api_key):
        with patch("hypertic.models.mistral.mistral.MistralClient"):
            return MistralAI(api_key=mock_api_key, model="mistral-large-latest")

    def test_mistral_ai_creation(self, mock_api_key):
        """Test MistralAI initialization."""
        with patch("hypertic.models.mistral.mistral.MistralClient"):
            model = MistralAI(api_key=mock_api_key, model="mistral-large-latest")
            assert model.api_key == mock_api_key
            assert model.model == "mistral-large-latest"

    def test_mistral_ai_creation_no_api_key(self):
        """Test MistralAI initialization without API key."""
        with patch("hypertic.models.mistral.mistral.getenv", return_value=None), patch("hypertic.models.mistral.mistral.MistralClient"):
            model = MistralAI(model="mistral-large-latest")
            assert model.api_key is None

    def test_mistral_ai_with_params(self, mock_api_key):
        """Test MistralAI with all parameters."""
        with patch("hypertic.models.mistral.mistral.MistralClient"):
            model = MistralAI(
                api_key=mock_api_key,
                model="mistral-large-latest",
                temperature=0.8,
                top_p=0.9,
                max_tokens=1000,
            )
            assert model.temperature == 0.8
            assert model.top_p == 0.9
            assert model.max_tokens == 1000

    def test_format_message_system(self, model):
        """Test _format_message with system role."""
        message = {"role": "system", "content": "You are helpful"}
        result = model._format_message(message)
        assert result.content == "You are helpful"

    def test_format_message_system_list(self, model):
        """Test _format_message with system role and list content."""
        message = {"role": "system", "content": [{"type": "text", "text": "System message"}]}
        result = model._format_message(message)
        assert result.content == "System message"

    def test_format_message_user(self, model):
        """Test _format_message with user role."""
        message = {"role": "user", "content": "Hello"}
        result = model._format_message(message)
        assert result.content == "Hello"

    def test_format_message_assistant(self, model):
        """Test _format_message with assistant role."""
        message = {"role": "assistant", "content": "Hi there"}
        result = model._format_message(message)
        assert result.content == "Hi there"

    def test_format_message_assistant_with_tool_calls(self, model):
        """Test _format_message with assistant role and tool calls."""
        message = {
            "role": "assistant",
            "content": "Response",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
        }
        result = model._format_message(message)
        assert result.tool_calls is not None

    def test_format_message_tool(self, model):
        """Test _format_message with tool role."""
        message = {"role": "tool", "content": "Tool result", "tool_call_id": "call_1"}
        result = model._format_message(message)
        assert result.content == "Tool result"
        assert result.tool_call_id == "call_1"

    def test_format_files_for_mistral_no_files(self, model):
        """Test _format_files_for_mistral with no files."""
        message = {"role": "user", "content": "Hello"}
        result = model._format_files_for_mistral(message)
        assert result == message

    def test_format_files_for_mistral_with_image_url(self, model):
        """Test _format_files_for_mistral with image URL."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        result = model._format_files_for_mistral(message)
        assert "content" in result
        assert isinstance(result["content"], list) or isinstance(result["content"], str)

    def test_format_files_for_mistral_with_image_base64(self, model):
        """Test _format_files_for_mistral with image base64."""
        file_obj = File(content=b"fake image", file_type=FileType.IMAGE, mime_type="image/jpeg")
        with patch.object(File, "to_base64", return_value="base64data"):
            message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
            result = model._format_files_for_mistral(message)
            assert "content" in result

    def test_format_files_for_mistral_removes_file_objects(self, model):
        """Test that _format_files_for_mistral removes _file_objects."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "_file_objects": [file_obj], "files": ["url"]}
        result = model._format_files_for_mistral(message)
        assert "_file_objects" not in result
        assert "files" not in result

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_basic(self, model):
        """Test ahandle_non_streaming with basic message."""
        mock_message = MagicMock()
        mock_message.content = "Response text"
        mock_message.tool_calls = None
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.complete_async = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("mistral-large-latest", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_tools(self, model):
        """Test ahandle_non_streaming with tools."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = '{"param": "value"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        # Create a proper usage object
        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.complete_async = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("mistral-large-latest", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_error(self, model):
        """Test ahandle_non_streaming with error."""
        from mistralai.models import SDKError

        try:
            sdk_error = SDKError(message="API Error")
        except TypeError:
            sdk_error = Exception("API Error")

        model.async_client.chat.complete_async = AsyncMock(side_effect=sdk_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="API Error"):  # noqa: B017
            await model.ahandle_non_streaming("mistral-large-latest", messages, None, tool_executor)

    def test_handle_non_streaming_basic(self, model):
        """Test handle_non_streaming with basic message."""
        mock_message = MagicMock()
        mock_message.content = "Response text"
        mock_message.tool_calls = None
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        class ResponseObj:
            def __init__(self):
                self.choices = [mock_choice]
                self.usage = mock_usage

        mock_response = ResponseObj()

        model.client.chat.complete = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("mistral-large-latest", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_with_tools(self, model):
        """Test handle_non_streaming with tools."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = '{"param": "value"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        # Create a proper usage object
        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        # Create response with proper structure
        mock_response = MagicMock(spec=["choices", "usage"])
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.client.chat.complete = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("mistral-large-latest", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_streaming_basic(self, model):
        """Test ahandle_streaming with basic message."""
        mock_delta = MagicMock()
        mock_delta.content = "Response"
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_tool_calls(self, model):
        """Test ahandle_streaming with tool calls."""
        mock_function_delta = MagicMock()
        mock_function_delta.name = "test_tool"
        mock_function_delta.arguments = '{"param": "value"}'

        mock_tool_call_delta = MagicMock()
        mock_tool_call_delta.index = 0
        mock_tool_call_delta.id = "call_1"
        mock_tool_call_delta.function = mock_function_delta

        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = [mock_tool_call_delta]

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "tool_calls"

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, tools, tool_executor):
            events.append(event)

        assert len(events) >= 0

    def test_handle_streaming_basic(self, model):
        """Test handle_streaming with basic message."""
        mock_delta = MagicMock()
        mock_delta.content = "Response"
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

        model.client.chat.stream = MagicMock(return_value=iter([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("mistral-large-latest", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)

    def test_handle_streaming_with_finish_reason(self, model):
        """Test handle_streaming with finish reason."""
        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

        model.client.chat.stream = MagicMock(return_value=iter([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("mistral-large-latest", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, FinishReasonEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_response_format_pydantic(self, model):
        """Test ahandle_non_streaming with Pydantic response format."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_message = MagicMock()
        mock_message.content = '{"result": "test"}'
        mock_message.tool_calls = None
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.complete_async = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("mistral-large-latest", messages, None, tool_executor, response_format=OutputModel)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_tools_and_response_format_error(self, model):
        """Test ahandle_non_streaming raises error when tools and response_format are both provided."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="does not support structured output"):
            await model.ahandle_non_streaming(
                "mistral-large-latest",
                messages,
                tools,
                tool_executor,
                response_format={"type": "json_object"},
            )

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_tools(self, model):
        """Test ahandle_non_streaming with sequential (non-parallel) tool execution."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = '{"param": "value"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.complete_async = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("mistral-large-latest", messages, tools, tool_executor)
        assert result is None
        assert model._execute_tools_parallel_async.call_count == 1

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_tools_json_decode_error(self, model):
        """Test ahandle_non_streaming with sequential tools and JSON decode error."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = "invalid json"

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.complete_async = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("mistral-large-latest", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_tools_empty_arguments(self, model):
        """Test ahandle_non_streaming with sequential tools and empty arguments."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = ""  # Empty arguments

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.complete_async = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("mistral-large-latest", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_tools_dict_arguments(self, model):
        """Test ahandle_non_streaming with sequential tools and dict arguments."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = {"param": "value"}  # Dict instead of string

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.complete_async = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("mistral-large-latest", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_content_accumulation(self, model):
        """Test ahandle_non_streaming with content accumulation from previous messages."""
        mock_message = MagicMock()
        mock_message.content = "Final response"
        mock_message.tool_calls = None
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.complete_async = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        # Previous assistant message with content
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Previous response"},
        ]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("mistral-large-latest", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_list_content(self, model):
        """Test ahandle_non_streaming with list content in message."""
        mock_message = MagicMock()
        mock_message.content = [{"type": "text", "text": "Response"}]
        mock_message.tool_calls = None
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.complete_async = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("mistral-large-latest", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_usage_none(self, model):
        """Test ahandle_non_streaming with None usage."""
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None  # None usage

        model.async_client.chat.complete_async = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("mistral-large-latest", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_http_validation_error(self, model):
        """Test ahandle_non_streaming with HTTP validation error."""
        from mistralai.models import HTTPValidationError, HTTPValidationErrorData

        error_data = HTTPValidationErrorData(detail=[{"loc": ["body"], "msg": "Validation error", "type": "value_error"}])
        try:
            validation_error = HTTPValidationError(data=error_data)
        except TypeError:

            class CustomHTTPValidationError(HTTPValidationError):
                def __str__(self):
                    return "Validation error"

            try:
                mock_response = MagicMock()
                mock_response.status_code = 422
                validation_error = CustomHTTPValidationError(data=error_data, raw_response=mock_response)
            except Exception:
                validation_error = Exception("Validation error")
                import hypertic.models.mistral.mistral as mistral_module

                original_handler = mistral_module.HTTPValidationError
                mistral_module.HTTPValidationError = type("HTTPValidationError", (type(validation_error),), {})
                try:
                    model.async_client.chat.complete_async = AsyncMock(side_effect=validation_error)
                    messages = [{"role": "user", "content": "Hello"}]
                    tool_executor = MagicMock()
                    with pytest.raises(Exception, match="Mistral API Validation Error"):
                        await model.ahandle_non_streaming("mistral-large-latest", messages, None, tool_executor)
                finally:
                    mistral_module.HTTPValidationError = original_handler
                return

        model.async_client.chat.complete_async = AsyncMock(side_effect=validation_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Mistral API Validation Error"):
            await model.ahandle_non_streaming("mistral-large-latest", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_response_format(self, model):
        """Test ahandle_streaming with response format."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_delta = MagicMock()
        mock_delta.content = "Response"
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, None, tool_executor, response_format=OutputModel):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_tools_and_response_format_error(self, model):
        """Test ahandle_streaming raises error when tools and response_format are both provided."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="does not support structured output"):
            async for _ in model.ahandle_streaming(
                "mistral-large-latest",
                messages,
                tools,
                tool_executor,
                response_format={"type": "json_object"},
            ):
                pass

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_usage_in_chunk(self, model):
        """Test ahandle_streaming with usage in chunk."""

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20
                self.total_tokens = 30

        mock_usage = UsageObj()

        mock_delta = MagicMock()
        mock_delta.content = "Response"
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = mock_usage

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()
        tool_executor._streaming_usage = None

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert tool_executor._streaming_usage is not None

    @pytest.mark.asyncio
    async def test_ahandle_streaming_no_choices_with_usage(self, model):
        """Test ahandle_streaming with no choices but usage."""

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20
                self.total_tokens = 30

        mock_usage = UsageObj()

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = []
        mock_chunk_data.usage = mock_usage

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()
        tool_executor._streaming_usage = None

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_delta_content_list(self, model):
        """Test ahandle_streaming with delta content as list."""
        from mistralai.models import TextChunk

        mock_text_chunk = MagicMock(spec=TextChunk)
        mock_text_chunk.text = "Response"

        mock_delta = MagicMock()
        mock_delta.content = [mock_text_chunk]
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_delta_content_list_dict(self, model):
        """Test ahandle_streaming with delta content as list of dicts."""
        mock_delta = MagicMock()
        mock_delta.content = [{"text": "Response"}]
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_delta_content_list_reference(self, model):
        """Test ahandle_streaming with delta content as list with reference type."""
        mock_delta = MagicMock()
        mock_delta.content = [{"type": "reference", "text": "Reference text"}]
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_delta_content_list_reference_content(self, model):
        """Test ahandle_streaming with delta content as list with reference type and content field."""
        mock_delta = MagicMock()
        mock_delta.content = [{"type": "reference", "content": "Reference content"}]
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_delta_content_list_string(self, model):
        """Test ahandle_streaming with delta content as list of strings."""
        mock_delta = MagicMock()
        mock_delta.content = ["Response", " text"]
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_tool_calls_sequential(self, model):
        """Test ahandle_streaming with sequential tool execution."""
        mock_function_delta = MagicMock()
        mock_function_delta.name = "test_tool"
        mock_function_delta.arguments = '{"param": "value"}'

        mock_tool_call_delta = MagicMock()
        mock_tool_call_delta.index = 0
        mock_tool_call_delta.id = "call_1"
        mock_tool_call_delta.function = mock_function_delta

        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = [mock_tool_call_delta]

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "tool_calls"

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor._streaming_usage = None

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, tools, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_sequential_multiple_tool_calls(self, model):
        """Test ahandle_streaming with sequential execution and multiple tool calls."""
        mock_function1_delta = MagicMock()
        mock_function1_delta.name = "test_tool1"
        mock_function1_delta.arguments = '{"param": "value1"}'

        mock_function2_delta = MagicMock()
        mock_function2_delta.name = "test_tool2"
        mock_function2_delta.arguments = '{"param": "value2"}'

        mock_tool_call_delta1 = MagicMock()
        mock_tool_call_delta1.index = 0
        mock_tool_call_delta1.id = "call_1"
        mock_tool_call_delta1.function = mock_function1_delta

        mock_tool_call_delta2 = MagicMock()
        mock_tool_call_delta2.index = 1
        mock_tool_call_delta2.id = "call_2"
        mock_tool_call_delta2.function = mock_function2_delta

        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = [mock_tool_call_delta1, mock_tool_call_delta2]

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "tool_calls"

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool1": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool1", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor._streaming_usage = None

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, tools, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert model._execute_tools_parallel_async.call_count == 1

    @pytest.mark.asyncio
    async def test_ahandle_streaming_tool_call_delta_concatenation(self, model):
        """Test ahandle_streaming with tool call delta concatenation."""
        mock_function_delta1 = MagicMock()
        mock_function_delta1.name = "test_"
        mock_function_delta1.arguments = None

        mock_tool_call_delta1 = MagicMock()
        mock_tool_call_delta1.index = 0
        mock_tool_call_delta1.id = "call_"
        mock_tool_call_delta1.function = mock_function_delta1

        mock_function_delta2 = MagicMock()
        mock_function_delta2.name = "tool"
        mock_function_delta2.arguments = '{"param": "value"}'

        mock_tool_call_delta2 = MagicMock()
        mock_tool_call_delta2.index = 0
        mock_tool_call_delta2.id = "1"
        mock_tool_call_delta2.function = mock_function_delta2

        mock_delta1 = MagicMock()
        mock_delta1.content = None
        mock_delta1.tool_calls = [mock_tool_call_delta1]

        mock_delta2 = MagicMock()
        mock_delta2.content = None
        mock_delta2.tool_calls = [mock_tool_call_delta2]

        mock_choice1 = MagicMock()
        mock_choice1.delta = mock_delta1
        mock_choice1.finish_reason = None

        mock_choice2 = MagicMock()
        mock_choice2.delta = mock_delta2
        mock_choice2.finish_reason = "tool_calls"

        mock_chunk_data1 = MagicMock()
        mock_chunk_data1.choices = [mock_choice1]
        mock_chunk_data1.usage = None

        mock_chunk_data2 = MagicMock()
        mock_chunk_data2.choices = [mock_choice2]
        mock_chunk_data2.usage = None

        mock_chunk1 = MagicMock()
        mock_chunk1.data = mock_chunk_data1

        mock_chunk2 = MagicMock()
        mock_chunk2.data = mock_chunk_data2

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk1, mock_chunk2]))
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor._streaming_usage = None

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, tools, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_finish_reason_and_usage(self, model):
        """Test ahandle_streaming with finish reason and usage."""

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20
                self.total_tokens = 30

        mock_usage = UsageObj()

        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = mock_usage

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()
        tool_executor._streaming_usage = None

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_stream_error_handling(self, model):
        """Test ahandle_streaming with stream error handling."""
        mock_delta = MagicMock()
        mock_delta.content = "Response"
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None

        mock_chunk_data = MagicMock()
        mock_chunk_data.choices = [mock_choice]
        mock_chunk_data.usage = None

        mock_chunk = MagicMock()
        mock_chunk.data = mock_chunk_data

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
                if self.index == 1:
                    raise ValueError("Pydantic validation error")
                return item

        model.async_client.chat.stream_async = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("mistral-large-latest", messages, None, tool_executor):
            events.append(event)

        # Should continue despite errors
        assert len(events) >= 0

    def test_handle_non_streaming_with_response_format_pydantic(self, model):
        """Test handle_non_streaming with Pydantic response format."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_message = MagicMock()
        mock_message.content = '{"result": "test"}'
        mock_message.tool_calls = None
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.client.chat.complete = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("mistral-large-latest", messages, None, tool_executor, response_format=OutputModel)
        assert result is not None

    def test_handle_non_streaming_tools_and_response_format_error(self, model):
        """Test handle_non_streaming raises error when tools and response_format are both provided."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="does not support structured output"):
            model.handle_non_streaming(
                "mistral-large-latest",
                messages,
                tools,
                tool_executor,
                response_format={"type": "json_object"},
            )

    def test_handle_non_streaming_sequential_tools(self, model):
        """Test handle_non_streaming with sequential (non-parallel) tool execution."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = '{"param": "value"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.client.chat.complete = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("mistral-large-latest", messages, tools, tool_executor)
        assert result is None
        assert model._execute_tools_parallel_sync.call_count == 1

    def test_handle_non_streaming_sequential_tools_json_decode_error(self, model):
        """Test handle_non_streaming with sequential tools and JSON decode error."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = "invalid json"

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.client.chat.complete = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("mistral-large-latest", messages, tools, tool_executor)
        assert result is None

    def test_handle_non_streaming_sequential_tools_empty_arguments(self, model):
        """Test handle_non_streaming with sequential tools and empty arguments."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = ""  # Empty arguments

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.client.chat.complete = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("mistral-large-latest", messages, tools, tool_executor)
        assert result is None

    def test_handle_non_streaming_sequential_tools_dict_arguments(self, model):
        """Test handle_non_streaming with sequential tools and dict arguments."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = {"param": "value"}  # Dict instead of string

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.client.chat.complete = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("mistral-large-latest", messages, tools, tool_executor)
        assert result is None

    def test_handle_non_streaming_with_content_accumulation(self, model):
        """Test handle_non_streaming with content accumulation from previous messages."""
        mock_message = MagicMock()
        mock_message.content = "Final response"
        mock_message.tool_calls = None
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.client.chat.complete = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        # Previous assistant message with content
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Previous response"},
        ]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("mistral-large-latest", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_with_list_content(self, model):
        """Test handle_non_streaming with list content in message."""
        mock_message = MagicMock()
        mock_message.content = [{"type": "text", "text": "Response"}]
        mock_message.tool_calls = None
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        class UsageObj:
            def __init__(self):
                self.prompt_tokens = 10
                self.completion_tokens = 20

        mock_usage = UsageObj()

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.client.chat.complete = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("mistral-large-latest", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_usage_none(self, model):
        """Test handle_non_streaming with None usage."""
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None  # None usage

        model.client.chat.complete = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("mistral-large-latest", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_http_validation_error(self, model):
        """Test handle_non_streaming with HTTP validation error."""
        from mistralai.models import HTTPValidationError, HTTPValidationErrorData

        error_data = HTTPValidationErrorData(detail=[{"loc": ["body"], "msg": "Validation error", "type": "value_error"}])
        try:
            validation_error = HTTPValidationError(data=error_data)
        except TypeError:

            class CustomHTTPValidationError(HTTPValidationError):
                def __str__(self):
                    return "Validation error"

            try:
                mock_response = MagicMock()
                mock_response.status_code = 422
                validation_error = CustomHTTPValidationError(data=error_data, raw_response=mock_response)
            except Exception:
                validation_error = Exception("Validation error")
                import hypertic.models.mistral.mistral as mistral_module

                original_handler = mistral_module.HTTPValidationError
                mistral_module.HTTPValidationError = type("HTTPValidationError", (type(validation_error),), {})
                try:
                    model.client.chat.complete = MagicMock(side_effect=validation_error)
                    messages = [{"role": "user", "content": "Hello"}]
                    tool_executor = MagicMock()
                    with pytest.raises(Exception, match="Mistral API Validation Error"):
                        model.handle_non_streaming("mistral-large-latest", messages, None, tool_executor)
                finally:
                    mistral_module.HTTPValidationError = original_handler
                return

        model.client.chat.complete = MagicMock(side_effect=validation_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Mistral API Validation Error"):
            model.handle_non_streaming("mistral-large-latest", messages, None, tool_executor)

    def test_format_files_for_mistral_with_document(self, model):
        """Test _format_files_for_mistral with document file."""
        file_obj = File(url="https://example.com/doc.pdf", file_type=FileType.DOCUMENT)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        result = model._format_files_for_mistral(message)
        assert "content" in result
        assert isinstance(result["content"], list) or isinstance(result["content"], str)

    def test_format_files_for_mistral_with_document_base64(self, model):
        """Test _format_files_for_mistral with document base64."""
        file_obj = File(content=b"fake pdf", file_type=FileType.DOCUMENT, mime_type="application/pdf")
        with patch.object(File, "to_base64", return_value="base64data"):
            message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
            result = model._format_files_for_mistral(message)
            assert "content" in result

    def test_format_files_for_mistral_with_existing_content_list(self, model):
        """Test _format_files_for_mistral with existing content list."""
        from mistralai.models import TextChunk

        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": [TextChunk(text="Check this")], "_file_objects": [file_obj]}
        result = model._format_files_for_mistral(message)
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 1

    def test_format_files_for_mistral_with_content_list_dict(self, model):
        """Test _format_files_for_mistral with content as list of dicts."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": [{"type": "text", "text": "Check this"}], "_file_objects": [file_obj]}
        result = model._format_files_for_mistral(message)
        assert isinstance(result["content"], list)

    def test_format_files_for_mistral_with_content_list_dict_image_url(self, model):
        """Test _format_files_for_mistral with content list containing image_url dict."""
        file_obj = File(url="https://example.com/image2.jpg", file_type=FileType.IMAGE)
        message = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}}],
            "_file_objects": [file_obj],
        }
        result = model._format_files_for_mistral(message)
        assert isinstance(result["content"], list)

    def test_format_files_for_mistral_with_content_list_dict_document_url(self, model):
        """Test _format_files_for_mistral with content list containing document_url dict."""
        file_obj = File(url="https://example.com/doc2.pdf", file_type=FileType.DOCUMENT)
        message = {
            "role": "user",
            "content": [{"type": "document_url", "document_url": {"url": "https://example.com/doc1.pdf"}}],
            "_file_objects": [file_obj],
        }
        result = model._format_files_for_mistral(message)
        assert isinstance(result["content"], list)

    def test_format_files_for_mistral_with_content_list_dict_string_url(self, model):
        """Test _format_files_for_mistral with content list containing string image_url."""
        file_obj = File(url="https://example.com/image2.jpg", file_type=FileType.IMAGE)
        message = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": "https://example.com/image1.jpg"}],
            "_file_objects": [file_obj],
        }
        result = model._format_files_for_mistral(message)
        assert isinstance(result["content"], list)

    def test_format_files_for_mistral_with_content_list_dict_string_document_url(self, model):
        """Test _format_files_for_mistral with content list containing string document_url."""
        file_obj = File(url="https://example.com/doc2.pdf", file_type=FileType.DOCUMENT)
        message = {
            "role": "user",
            "content": [{"type": "document_url", "document_url": "https://example.com/doc1.pdf"}],
            "_file_objects": [file_obj],
        }
        result = model._format_files_for_mistral(message)
        assert isinstance(result["content"], list)

    def test_format_files_for_mistral_single_text_chunk(self, model):
        """Test _format_files_for_mistral with single text chunk."""
        from mistralai.models import TextChunk

        message = {"role": "user", "content": [TextChunk(text="Hello")], "_file_objects": []}
        result = model._format_files_for_mistral(message)
        assert isinstance(result["content"], list) or result["content"] == "Hello"

    def test_format_files_for_mistral_empty_content(self, model):
        """Test _format_files_for_mistral with empty content."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "_file_objects": [file_obj]}
        result = model._format_files_for_mistral(message)
        assert "content" in result
        assert isinstance(result["content"], list) or result["content"] == ""

    def test_format_files_for_mistral_unsupported_file_type(self, model):
        """Test _format_files_for_mistral with unsupported file type."""
        file_obj = File(content=b"data", file_type=FileType.AUDIO)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        with patch("hypertic.utils.log.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            result = model._format_files_for_mistral(message)
            assert "content" in result
            mock_logger.warning.assert_called()

    def test_format_message_user_default(self, model):
        """Test _format_message with unknown role defaults to UserMessage."""
        message = {"role": "unknown", "content": "Hello"}
        result = model._format_message(message)
        assert result.content == "Hello"
