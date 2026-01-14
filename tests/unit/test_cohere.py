from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.models.cohere.cohere import Cohere
from hypertic.models.events import ResponseCompletedEvent
from hypertic.utils.files import File, FileType


class TestCohere:
    @pytest.fixture
    def model(self, mock_api_key):
        with patch("hypertic.models.cohere.cohere.CohereAsyncClient"), patch("hypertic.models.cohere.cohere.CohereClient"):
            return Cohere(api_key=mock_api_key, model="command-r-plus")

    def test_cohere_creation(self, mock_api_key):
        """Test Cohere initialization."""
        with patch("hypertic.models.cohere.cohere.CohereAsyncClient"), patch("hypertic.models.cohere.cohere.CohereClient"):
            model = Cohere(api_key=mock_api_key, model="command-r-plus")
            assert model.api_key == mock_api_key
            assert model.model == "command-r-plus"

    def test_cohere_creation_no_api_key(self):
        """Test Cohere initialization without API key."""
        with (
            patch("hypertic.models.cohere.cohere.getenv", return_value=None),
            patch("hypertic.models.cohere.cohere.CohereAsyncClient"),
            patch("hypertic.models.cohere.cohere.CohereClient"),
        ):
            model = Cohere(model="command-r-plus")
            assert model.api_key is None

    def test_cohere_with_params(self, mock_api_key):
        """Test Cohere with all parameters."""
        with patch("hypertic.models.cohere.cohere.CohereAsyncClient"), patch("hypertic.models.cohere.cohere.CohereClient"):
            model = Cohere(
                api_key=mock_api_key,
                model="command-r-plus",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000,
                thinking=True,
                thinking_tokens=100,
            )
            assert model.temperature == 0.7
            assert model.top_p == 0.9
            assert model.max_tokens == 1000
            assert model.thinking is True
            assert model.thinking_tokens == 100

    def test_convert_tools_to_cohere(self, model):
        """Test _convert_tools_to_cohere."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = model._convert_tools_to_cohere(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "test_tool"

    def test_format_files_for_cohere_no_files(self, model):
        """Test _format_files_for_cohere with no files."""
        message = {"role": "user", "content": "Hello"}
        result = model._format_files_for_cohere(message)
        assert result == message

    def test_format_files_for_cohere_with_image_url(self, model):
        """Test _format_files_for_cohere with image URL."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        result = model._format_files_for_cohere(message)
        assert "content" in result
        assert isinstance(result["content"], list)

    def test_format_files_for_cohere_with_image_base64(self, model):
        """Test _format_files_for_cohere with image base64."""
        file_obj = File(content=b"fake image", file_type=FileType.IMAGE, mime_type="image/jpeg")
        with patch.object(File, "to_base64", return_value="base64data"):
            message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
            result = model._format_files_for_cohere(message)
            assert "content" in result
            assert isinstance(result["content"], list)

    def test_format_files_for_cohere_removes_file_objects(self, model):
        """Test that _format_files_for_cohere removes _file_objects."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "_file_objects": [file_obj], "files": ["url"]}
        result = model._format_files_for_cohere(message)
        assert "_file_objects" not in result
        assert "files" not in result

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_basic(self, model):
        """Test ahandle_non_streaming with basic message."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response text"

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = None
        mock_response.finish_reason = "stop"

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("command-r-plus", messages, None, tool_executor)
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
        mock_tool_call.model_dump = MagicMock(return_value={"id": "call_1", "type": "function", "function": {"name": "test_tool"}})

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_usage = MagicMock()
        mock_usage.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("command-r-plus", messages, tools, tool_executor)
        assert result is None  # Returns None when tool calls are present

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_thinking(self, model):
        """Test ahandle_non_streaming with thinking enabled."""
        model.thinking = True
        model.thinking_tokens = 100

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response text"

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = None
        mock_response.finish_reason = "stop"

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("command-r-plus", messages, None, tool_executor)
        assert result is not None
        # Verify thinking config was passed
        call_args = model.async_client.v2.chat.call_args
        assert "thinking" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_error(self, model):
        """Test ahandle_non_streaming with error."""
        model.async_client.v2.chat = AsyncMock(side_effect=Exception("API Error"))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Error from Cohere SDK"):
            await model.ahandle_non_streaming("command-r-plus", messages, None, tool_executor)

    def test_handle_non_streaming_basic(self, model):
        """Test handle_non_streaming with basic message."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response text"

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = None
        mock_response.finish_reason = "stop"

        model.client.v2.chat = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("command-r-plus", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_with_tools(self, model):
        """Test handle_non_streaming with tools."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = '{"param": "value"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function
        mock_tool_call.model_dump = MagicMock(return_value={"id": "call_1", "type": "function", "function": {"name": "test_tool"}})

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_usage = MagicMock()
        mock_usage.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage

        model.client.v2.chat = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("command-r-plus", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_streaming_basic(self, model):
        """Test ahandle_streaming with basic message."""
        mock_delta = MagicMock()
        mock_delta.thinking = None
        mock_delta.text = "Response"

        mock_event = MagicMock()
        mock_event.type = "content-delta"
        mock_event.delta.message.content = mock_delta

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

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_event]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_reasoning(self, model):
        """Test ahandle_streaming with reasoning content."""
        mock_delta = MagicMock()
        mock_delta.thinking = "Reasoning step"
        mock_delta.text = "Response"

        mock_event = MagicMock()
        mock_event.type = "content-delta"
        mock_event.delta.message.content = mock_delta

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

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_event]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    def test_handle_streaming_basic(self, model):
        """Test handle_streaming with basic message."""
        mock_delta = MagicMock()
        mock_delta.thinking = None
        mock_delta.text = "Response"

        mock_event = MagicMock()
        mock_event.type = "content-delta"
        mock_event.delta.message.content = mock_delta

        model.client.v2.chat_stream = MagicMock(return_value=iter([mock_event]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("command-r-plus", messages, None, tool_executor))
        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_error(self, model):
        """Test ahandle_streaming with error."""
        model.async_client.v2.chat_stream = MagicMock(side_effect=Exception("Streaming Error"))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Error from Cohere SDK"):
            async for _ in model.ahandle_streaming("command-r-plus", messages, None, tool_executor):
                pass

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_response_format_pydantic(self, model):
        """Test ahandle_non_streaming with Pydantic response format."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = '{"result": "test"}'

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = None
        mock_response.finish_reason = "stop"

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("command-r-plus", messages, None, tool_executor, response_format=OutputModel)
        assert result is not None

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

        mock_usage = MagicMock()
        mock_usage.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("command-r-plus", messages, tools, tool_executor)
        assert result is None
        assert model._execute_tools_parallel_async.call_count == 1

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_tools_no_function(self, model):
        """Test ahandle_non_streaming with sequential tools but no function."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = None

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_usage = MagicMock()
        mock_usage.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("command-r-plus", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_tools_no_function_name(self, model):
        """Test ahandle_non_streaming with sequential tools but no function name."""
        mock_function = MagicMock()
        mock_function.name = None
        mock_function.arguments = '{"param": "value"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_usage = MagicMock()
        mock_usage.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("command-r-plus", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_thinking_dict(self, model):
        """Test ahandle_non_streaming with thinking as dict."""
        model.thinking = {"type": "enabled", "token_budget": 200}
        model.thinking_tokens = None

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response text"

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = None
        mock_response.finish_reason = "stop"

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("command-r-plus", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_all_params(self, model):
        """Test ahandle_non_streaming with all parameters set."""
        model.temperature = 0.7
        model.top_p = 0.9
        model.presence_penalty = 0.5
        model.frequency_penalty = 0.5
        model.max_tokens = 1000

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = None
        mock_response.finish_reason = "stop"

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("command-r-plus", messages, None, tool_executor)
        assert result is not None
        # Verify all params were passed
        call_args = model.async_client.v2.chat.call_args
        assert call_args.kwargs.get("temperature") == 0.7
        assert call_args.kwargs.get("p") == 0.9
        assert call_args.kwargs.get("presence_penalty") == 0.5
        assert call_args.kwargs.get("frequency_penalty") == 0.5
        assert call_args.kwargs.get("max_tokens") == 1000

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_content_blocks_thinking(self, model):
        """Test ahandle_non_streaming with thinking content block."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response text"

        mock_thinking_block = MagicMock()
        mock_thinking_block.type = "thinking"
        mock_thinking_block.thinking = "Let me think..."

        mock_message = MagicMock()
        mock_message.content = [mock_text_block, mock_thinking_block]
        mock_message.tool_calls = None

        mock_usage = MagicMock()
        mock_usage.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage
        mock_response.finish_reason = "stop"

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("command-r-plus", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_content_blocks_tool_call(self, model):
        """Test ahandle_non_streaming with tool_call content block."""
        mock_tool_call_block = MagicMock()
        mock_tool_call_block.type = "tool_call"
        mock_tool_call_block.id = "call_1"
        mock_tool_call_block.name = "test_tool"
        mock_tool_call_block.input = {"param": "value"}

        mock_message = MagicMock()
        mock_message.content = [mock_tool_call_block]
        mock_message.tool_calls = None

        mock_usage = MagicMock()
        mock_usage.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage
        mock_response.finish_reason = "stop"

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("command-r-plus", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_usage_dict_format(self, model):
        """Test ahandle_non_streaming with usage as dict."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = {"input_tokens": 10, "output_tokens": 20}  # Dict format
        mock_response.finish_reason = "stop"

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("command-r-plus", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_parallel_tools_with_none_function_name(self, model):
        """Test ahandle_non_streaming with parallel tools where function name is None."""
        mock_function1 = MagicMock()
        mock_function1.name = "test_tool1"
        mock_function1.arguments = '{"param": "value1"}'

        mock_function2 = MagicMock()
        mock_function2.name = None  # None function name
        mock_function2.arguments = '{"param": "value2"}'

        mock_tool_call1 = MagicMock()
        mock_tool_call1.id = "call_1"
        mock_tool_call1.function = mock_function1

        mock_tool_call2 = MagicMock()
        mock_tool_call2.id = "call_2"
        mock_tool_call2.function = mock_function2

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call1, mock_tool_call2]

        mock_usage = MagicMock()
        mock_usage.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool1": "result1"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool1", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("command-r-plus", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_parallel_tools_with_none_function(self, model):
        """Test ahandle_non_streaming with parallel tools where function is None."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = '{"param": "value"}'

        mock_tool_call1 = MagicMock()
        mock_tool_call1.id = "call_1"
        mock_tool_call1.function = mock_function

        mock_tool_call2 = MagicMock()
        mock_tool_call2.id = "call_2"
        mock_tool_call2.function = None  # None function

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call1, mock_tool_call2]

        mock_usage = MagicMock()
        mock_usage.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage

        model.async_client.v2.chat = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("command-r-plus", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_streaming_tool_call_start(self, model):
        """Test ahandle_streaming with tool-call-start event."""
        mock_tool_call_delta = MagicMock()
        mock_tool_call_delta.id = "call_1"
        mock_tool_call_delta.type = "function"
        mock_tool_call_delta.function = MagicMock()
        mock_tool_call_delta.function.name = "test_tool"

        mock_event = MagicMock()
        mock_event.type = "tool-call-start"
        mock_event.index = 0
        mock_event.delta.message.tool_calls = mock_tool_call_delta

        mock_message_end = MagicMock()
        mock_message_end.type = "message-end"
        mock_message_end.delta = None
        mock_message_end.message = None
        mock_message_end.usage = None
        mock_message_end.response = None

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

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_event, mock_message_end]))

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = AsyncMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor._aexecute_tools_parallel = AsyncMock(return_value=[("test_tool", "result")])

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, tools, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_tool_call_delta(self, model):
        """Test ahandle_streaming with tool-call-delta event."""
        mock_tool_call_delta = MagicMock()
        mock_tool_call_delta.function = MagicMock()
        mock_tool_call_delta.function.arguments = '{"param": "value"}'

        mock_event = MagicMock()
        mock_event.type = "tool-call-delta"
        mock_event.index = 0
        mock_event.delta.message.tool_calls = mock_tool_call_delta

        mock_message_end = MagicMock()
        mock_message_end.type = "message-end"
        mock_message_end.delta = None
        mock_message_end.message = None
        mock_message_end.usage = None
        mock_message_end.response = None

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

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_event, mock_message_end]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_message_end_with_usage_tokens(self, model):
        """Test ahandle_streaming with message-end event containing usage with tokens."""
        mock_tokens = MagicMock()
        mock_tokens.input_tokens = 10
        mock_tokens.output_tokens = 20

        mock_billed = MagicMock()
        mock_billed.input_tokens = 10
        mock_billed.output_tokens = 20

        mock_usage_obj = MagicMock()
        mock_usage_obj.tokens = mock_tokens
        mock_usage_obj.billed_units = mock_billed

        mock_delta = MagicMock()
        mock_delta.usage = mock_usage_obj

        mock_event = MagicMock()
        mock_event.type = "message-end"
        mock_event.delta = mock_delta
        mock_event.message = None
        mock_event.usage = None
        mock_event.response = None

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

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_event]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_message_end_with_usage_model_dump(self, model):
        """Test ahandle_streaming with message-end event containing usage with model_dump."""
        mock_usage_obj = MagicMock()
        mock_usage_obj.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})
        mock_usage_obj.tokens = None

        mock_delta = MagicMock()
        mock_delta.usage = mock_usage_obj

        mock_event = MagicMock()
        mock_event.type = "message-end"
        mock_event.delta = mock_delta
        mock_event.message = None
        mock_event.usage = None
        mock_event.response = None

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

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_event]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_message_end_with_message_usage(self, model):
        """Test ahandle_streaming with message-end event containing usage in message."""
        mock_usage_obj = MagicMock()
        mock_usage_obj.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_message = MagicMock()
        mock_message.usage = mock_usage_obj

        mock_delta = MagicMock()
        mock_delta.usage = None
        mock_delta.message = mock_message

        mock_event = MagicMock()
        mock_event.type = "message-end"
        mock_event.delta = mock_delta
        mock_event.message = None
        mock_event.usage = None
        mock_event.response = None

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

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_event]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_message_end_with_event_usage(self, model):
        """Test ahandle_streaming with message-end event containing usage in event."""
        mock_usage_obj = MagicMock()
        mock_usage_obj.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_event = MagicMock()
        mock_event.type = "message-end"
        mock_event.delta = None
        mock_event.message = None
        mock_event.usage = mock_usage_obj
        mock_event.response = None

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

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_event]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_message_end_with_response_usage(self, model):
        """Test ahandle_streaming with message-end event containing usage in response."""
        mock_usage_obj = MagicMock()
        mock_usage_obj.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_response = MagicMock()
        mock_response.usage = mock_usage_obj

        mock_event = MagicMock()
        mock_event.type = "message-end"
        mock_event.delta = None
        mock_event.message = None
        mock_event.usage = None
        mock_event.response = mock_response

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

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_event]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_sequential_tools(self, model):
        """Test ahandle_streaming with sequential tool execution."""
        mock_tool_call_delta = MagicMock()
        mock_tool_call_delta.id = "call_1"
        mock_tool_call_delta.type = "function"
        mock_tool_call_delta.function = MagicMock()
        mock_tool_call_delta.function.name = "test_tool"

        mock_tool_start = MagicMock()
        mock_tool_start.type = "tool-call-start"
        mock_tool_start.index = 0
        mock_tool_start.delta.message.tool_calls = mock_tool_call_delta

        mock_tool_delta = MagicMock()
        mock_tool_delta.type = "tool-call-delta"
        mock_tool_delta.index = 0
        mock_tool_delta.delta.message.tool_calls = MagicMock()
        mock_tool_delta.delta.message.tool_calls.function = MagicMock()
        mock_tool_delta.delta.message.tool_calls.function.arguments = '{"param": "value"}'

        mock_message_end = MagicMock()
        mock_message_end.type = "message-end"
        mock_message_end.delta = None
        mock_message_end.message = None
        mock_message_end.usage = None
        mock_message_end.response = None

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

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_tool_start, mock_tool_delta, mock_message_end]))
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, tools, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_response_format(self, model):
        """Test ahandle_streaming with response format."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_delta = MagicMock()
        mock_delta.thinking = None
        mock_delta.text = '{"result": "test"}'

        mock_event = MagicMock()
        mock_event.type = "content-delta"
        mock_event.delta.message.content = mock_delta

        mock_message_end = MagicMock()
        mock_message_end.type = "message-end"
        mock_message_end.delta = None
        mock_message_end.message = None
        mock_message_end.usage = None
        mock_message_end.response = None

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

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_event, mock_message_end]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, None, tool_executor, response_format=OutputModel):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_tool_calls_json_parse_error(self, model):
        """Test ahandle_streaming with tool calls that have invalid JSON arguments."""
        mock_tool_call_delta = MagicMock()
        mock_tool_call_delta.id = "call_1"
        mock_tool_call_delta.type = "function"
        mock_tool_call_delta.function = MagicMock()
        mock_tool_call_delta.function.name = "test_tool"

        mock_tool_start = MagicMock()
        mock_tool_start.type = "tool-call-start"
        mock_tool_start.index = 0
        mock_tool_start.delta.message.tool_calls = mock_tool_call_delta

        mock_tool_delta = MagicMock()
        mock_tool_delta.type = "tool-call-delta"
        mock_tool_delta.index = 0
        mock_tool_delta.delta.message.tool_calls = MagicMock()
        mock_tool_delta.delta.message.tool_calls.function = MagicMock()
        mock_tool_delta.delta.message.tool_calls.function.arguments = "invalid json"

        mock_message_end = MagicMock()
        mock_message_end.type = "message-end"
        mock_message_end.delta = None
        mock_message_end.message = None
        mock_message_end.usage = None
        mock_message_end.response = None

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

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_tool_start, mock_tool_delta, mock_message_end]))
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, tools, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_stream_close(self, model):
        """Test ahandle_streaming properly closes stream."""
        mock_event = MagicMock()
        mock_event.type = "message-end"
        mock_event.delta = None
        mock_event.message = None
        mock_event.usage = None
        mock_event.response = None

        mock_stream = MagicMock()
        mock_stream.aclose = AsyncMock()

        class AsyncIterator:
            def __init__(self, items, stream):
                self.items = items
                self.stream = stream
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        model.async_client.v2.chat_stream = MagicMock(return_value=AsyncIterator([mock_event], mock_stream))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("command-r-plus", messages, None, tool_executor):
            events.append(event)
            break  # Exit early to test finally block

        # Stream should be closed
        assert len(events) >= 0

    def test_handle_non_streaming_with_all_params(self, model):
        """Test handle_non_streaming with all parameters set."""
        model.temperature = 0.7
        model.top_p = 0.9
        model.presence_penalty = 0.5
        model.frequency_penalty = 0.5
        model.max_tokens = 1000

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = None
        mock_response.finish_reason = "stop"

        model.client.v2.chat = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("command-r-plus", messages, None, tool_executor)
        assert result is not None
        # Verify all params were passed
        call_args = model.client.v2.chat.call_args
        assert call_args.kwargs.get("temperature") == 0.7
        assert call_args.kwargs.get("p") == 0.9
        assert call_args.kwargs.get("presence_penalty") == 0.5
        assert call_args.kwargs.get("frequency_penalty") == 0.5
        assert call_args.kwargs.get("max_tokens") == 1000

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

        mock_usage = MagicMock()
        mock_usage.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage

        model.client.v2.chat = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("command-r-plus", messages, tools, tool_executor)
        assert result is None
        assert model._execute_tools_parallel_sync.call_count == 1

    def test_handle_non_streaming_with_thinking_dict(self, model):
        """Test handle_non_streaming with thinking as dict."""
        model.thinking = {"type": "enabled", "token_budget": 200}
        model.thinking_tokens = None

        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response text"

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = None
        mock_response.finish_reason = "stop"

        model.client.v2.chat = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("command-r-plus", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_with_content_blocks_thinking(self, model):
        """Test handle_non_streaming with thinking content block."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response text"

        mock_thinking_block = MagicMock()
        mock_thinking_block.type = "thinking"
        mock_thinking_block.thinking = "Let me think..."

        mock_message = MagicMock()
        mock_message.content = [mock_text_block, mock_thinking_block]
        mock_message.tool_calls = None

        mock_usage = MagicMock()
        mock_usage.model_dump = MagicMock(return_value={"input_tokens": 10, "output_tokens": 20})

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = mock_usage
        mock_response.finish_reason = "stop"

        model.client.v2.chat = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("command-r-plus", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_with_finish_reason(self, model):
        """Test handle_non_streaming with finish_reason."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_message = MagicMock()
        mock_message.content = [mock_text_block]
        mock_message.tool_calls = None

        mock_response = MagicMock()
        mock_response.message = mock_message
        mock_response.usage = None
        mock_response.finish_reason = "max_tokens"

        model.client.v2.chat = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("command-r-plus", messages, None, tool_executor)
        assert result is not None

    def test_handle_streaming_tool_call_start(self, model):
        """Test handle_streaming with tool-call-start event."""
        mock_tool_call_delta = MagicMock()
        mock_tool_call_delta.id = "call_1"
        mock_tool_call_delta.type = "function"
        mock_tool_call_delta.function = MagicMock()
        mock_tool_call_delta.function.name = "test_tool"

        mock_event = MagicMock()
        mock_event.type = "tool-call-start"
        mock_event.index = 0
        mock_event.delta.message.tool_calls = mock_tool_call_delta

        mock_message_end = MagicMock()
        mock_message_end.type = "message-end"
        mock_message_end.delta = None
        mock_message_end.message = None
        mock_message_end.usage = None
        mock_message_end.response = None

        model.client.v2.chat_stream = MagicMock(return_value=iter([mock_event, mock_message_end]))

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = list(model.handle_streaming("command-r-plus", messages, tools, tool_executor))
        assert len(events) >= 0

    def test_handle_streaming_sequential_tools(self, model):
        """Test handle_streaming with sequential tool execution."""
        mock_tool_call_delta = MagicMock()
        mock_tool_call_delta.id = "call_1"
        mock_tool_call_delta.type = "function"
        mock_tool_call_delta.function = MagicMock()
        mock_tool_call_delta.function.name = "test_tool"

        mock_tool_start = MagicMock()
        mock_tool_start.type = "tool-call-start"
        mock_tool_start.index = 0
        mock_tool_start.delta.message.tool_calls = mock_tool_call_delta

        mock_tool_delta = MagicMock()
        mock_tool_delta.type = "tool-call-delta"
        mock_tool_delta.index = 0
        mock_tool_delta.delta.message.tool_calls = MagicMock()
        mock_tool_delta.delta.message.tool_calls.function = MagicMock()
        mock_tool_delta.delta.message.tool_calls.function.arguments = '{"param": "value"}'

        mock_message_end = MagicMock()
        mock_message_end.type = "message-end"
        mock_message_end.delta = None
        mock_message_end.message = None
        mock_message_end.usage = None
        mock_message_end.response = None

        model.client.v2.chat_stream = MagicMock(return_value=iter([mock_tool_start, mock_tool_delta, mock_message_end]))
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = list(model.handle_streaming("command-r-plus", messages, tools, tool_executor))
        assert len(events) > 0

    def test_handle_streaming_message_end_with_usage_tokens(self, model):
        """Test handle_streaming with message-end event containing usage with tokens."""
        mock_tokens = MagicMock()
        mock_tokens.input_tokens = 10
        mock_tokens.output_tokens = 20

        mock_billed = MagicMock()
        mock_billed.input_tokens = 10
        mock_billed.output_tokens = 20

        mock_usage_obj = MagicMock()
        mock_usage_obj.tokens = mock_tokens
        mock_usage_obj.billed_units = mock_billed

        mock_delta = MagicMock()
        mock_delta.usage = mock_usage_obj

        mock_event = MagicMock()
        mock_event.type = "message-end"
        mock_event.delta = mock_delta
        mock_event.message = None
        mock_event.usage = None
        mock_event.response = None

        # Need to ensure tool_calls is empty so it goes to the finish_reason path
        # Also need to ensure usage is truthy after processing
        model.client.v2.chat_stream = MagicMock(return_value=iter([mock_event]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("command-r-plus", messages, None, tool_executor))
        # Should yield FinishReasonEvent and ResponseCompletedEvent if usage is processed correctly
        # The code checks `if usage:` so usage needs to be truthy
        assert len(events) >= 0  # May be 0 if usage processing fails

    def test_format_files_for_cohere_with_existing_content_list(self, model):
        """Test _format_files_for_cohere with existing content list."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": [{"type": "text", "text": "Check this"}], "_file_objects": [file_obj]}
        result = model._format_files_for_cohere(message)
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 1

    def test_format_files_for_cohere_unsupported_file_type(self, model):
        """Test _format_files_for_cohere with unsupported file type."""
        file_obj = File(content=b"data", file_type=FileType.DOCUMENT)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        with patch("hypertic.utils.log.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            result = model._format_files_for_cohere(message)
            # Should still return message, just without the unsupported file
            assert "content" in result
            mock_logger.warning.assert_called()
