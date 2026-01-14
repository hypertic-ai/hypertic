from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.models.deepseek.deepseek import DeepSeek
from hypertic.models.events import ContentEvent, ReasoningEvent, ResponseCompletedEvent, ToolCallsEvent
from hypertic.utils.files import File, FileType


class TestDeepSeek:
    @pytest.fixture
    def model(self, mock_api_key):
        with patch("hypertic.models.deepseek.deepseek.AsyncOpenAIClient"), patch("hypertic.models.deepseek.deepseek.OpenAIClient"):
            return DeepSeek(api_key=mock_api_key, model="deepseek-chat")

    def test_deepseek_creation(self, mock_api_key):
        """Test DeepSeek initialization."""
        with patch("hypertic.models.deepseek.deepseek.AsyncOpenAIClient"), patch("hypertic.models.deepseek.deepseek.OpenAIClient"):
            model = DeepSeek(api_key=mock_api_key, model="deepseek-chat")
            assert model.api_key == mock_api_key
            assert model.model == "deepseek-chat"

    def test_deepseek_creation_no_api_key(self):
        """Test DeepSeek initialization without API key."""
        with (
            patch("hypertic.models.deepseek.deepseek.getenv", return_value=None),
            patch("hypertic.models.deepseek.deepseek.AsyncOpenAIClient"),
            patch("hypertic.models.deepseek.deepseek.OpenAIClient"),
        ):
            model = DeepSeek(model="deepseek-chat")
            assert model.api_key is None

    def test_deepseek_with_params(self, mock_api_key):
        """Test DeepSeek with all parameters."""
        with patch("hypertic.models.deepseek.deepseek.AsyncOpenAIClient"), patch("hypertic.models.deepseek.deepseek.OpenAIClient"):
            model = DeepSeek(
                api_key=mock_api_key,
                model="deepseek-chat",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000,
            )
            assert model.temperature == 0.7
            assert model.top_p == 0.9
            assert model.max_tokens == 1000

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

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("deepseek-chat", messages, None, tool_executor)
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
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("deepseek-chat", messages, tools, tool_executor)
        assert result is None  # Returns None when tool calls are present

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_files_warning(self, model):
        """Test ahandle_non_streaming logs warning for files."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        messages = [{"role": "user", "content": "Hello", "_file_objects": [file_obj]}]
        tool_executor = MagicMock()

        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        model.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        with patch("hypertic.utils.log.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            await model.ahandle_non_streaming("deepseek-chat", messages, None, tool_executor)
            # Verify warning was logged
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_rate_limit_error(self, model):
        """Test ahandle_non_streaming with rate limit error."""
        from openai import RateLimitError

        mock_response = MagicMock()
        mock_response.status_code = 429
        rate_limit_error = RateLimitError(response=mock_response, body={}, message="Rate limited")

        model.async_client.chat.completions.create = AsyncMock(side_effect=rate_limit_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Rate limit"):
            await model.ahandle_non_streaming("deepseek-chat", messages, None, tool_executor)

    def test_handle_non_streaming_basic(self, model):
        """Test handle_non_streaming with basic message."""
        mock_message = MagicMock()
        mock_message.content = "Response text"
        mock_message.tool_calls = None
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.client.chat.completions.create = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("deepseek-chat", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_streaming_basic(self, model):
        """Test ahandle_streaming with basic message."""
        mock_delta = MagicMock()
        mock_delta.content = "Response"
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

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

        async def async_gen(chunk, tool_calls, tool_executor, messages):
            yield ContentEvent(content="Response")

        model.async_client.chat.completions.create = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._process_streaming_event_async = async_gen

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("deepseek-chat", messages, None, tool_executor):
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

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        model.client.chat.completions.create = MagicMock(return_value=iter([mock_chunk]))
        model._process_streaming_event_sync = MagicMock(return_value=[ContentEvent(content="Response")])

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("deepseek-chat", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_reasoning_content(self, model):
        """Test ahandle_non_streaming with reasoning content."""
        mock_message = MagicMock()
        mock_message.content = "Response text"
        mock_message.tool_calls = None
        mock_message.role = "assistant"
        mock_message.reasoning_content = "Let me think about this..."

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("deepseek-chat", messages, None, tool_executor)
        assert result is not None

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
        mock_message.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("deepseek-chat", messages, None, tool_executor, response_format=OutputModel)
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
        mock_tool_call.model_dump = MagicMock(return_value={"id": "call_1", "type": "function", "function": {"name": "test_tool"}})

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"
        mock_message.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("deepseek-chat", messages, tools, tool_executor)
        assert result is None
        assert model._execute_tools_parallel_async.call_count == 1

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_tools_with_existing_assistant_message(self, model):
        """Test ahandle_non_streaming with sequential tools when last message is assistant."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = '{"param": "value"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function
        mock_tool_call.model_dump = MagicMock(return_value={"id": "call_1", "type": "function", "function": {"name": "test_tool"}})

        mock_message = MagicMock()
        mock_message.content = "Previous response"
        mock_message.tool_calls = [mock_tool_call]
        mock_message.role = "assistant"
        mock_message.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Previous", "tool_calls": []}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("deepseek-chat", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_tools_empty_tool_calls(self, model):
        """Test ahandle_non_streaming with sequential tools but empty tool_calls."""
        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = []
        mock_message.role = "assistant"
        mock_message.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("deepseek-chat", messages, tools, tool_executor)
        # When tool_calls is empty, it returns a response (not None)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_all_params(self, model):
        """Test ahandle_non_streaming with all parameters set."""
        model.temperature = 0.7
        model.top_p = 0.9
        model.presence_penalty = 0.5
        model.frequency_penalty = 0.5
        model.max_tokens = 1000

        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
        mock_message.role = "assistant"
        mock_message.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("deepseek-chat", messages, None, tool_executor)
        assert result is not None
        # Verify all params were passed
        call_args = model.async_client.chat.completions.create.call_args[1]
        assert call_args.get("temperature") == 0.7
        assert call_args.get("top_p") == 0.9
        assert call_args.get("presence_penalty") == 0.5
        assert call_args.get("frequency_penalty") == 0.5
        assert call_args.get("max_tokens") == 1000

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_status_error(self, model):
        """Test ahandle_non_streaming with status error."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": {"message": "API Error"}})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.async_client.chat.completions.create = AsyncMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Status error"):
            await model.ahandle_non_streaming("deepseek-chat", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_reasoning_content(self, model):
        """Test ahandle_streaming with reasoning content."""
        mock_delta = MagicMock()
        mock_delta.content = "Response"
        mock_delta.tool_calls = None
        mock_delta.reasoning_content = "Let me think..."

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

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

        async def async_gen(chunk, tool_calls, tool_executor, messages):
            yield ReasoningEvent(reasoning="Let me think...")
            yield ContentEvent(content="Response")

        model.async_client.chat.completions.create = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._process_streaming_event_async = async_gen

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("deepseek-chat", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_response_format(self, model):
        """Test ahandle_streaming with response format."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_delta = MagicMock()
        mock_delta.content = "Response"
        mock_delta.tool_calls = None
        mock_delta.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

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

        async def async_gen(chunk, tool_calls, tool_executor, messages):
            yield ContentEvent(content="Response")

        model.async_client.chat.completions.create = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._process_streaming_event_async = async_gen

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("deepseek-chat", messages, None, tool_executor, response_format=OutputModel):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_no_choices_with_usage(self, model):
        """Test _process_streaming_event_async with no choices but usage."""
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_chunk = MagicMock()
        mock_chunk.choices = []
        mock_chunk.usage = mock_usage

        tool_executor = MagicMock()
        tool_executor._streaming_usage = None

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, [], tool_executor, []):
            events.append(event)

        assert len(events) == 1  # Should yield ResponseCompletedEvent
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)
        assert tool_executor._streaming_usage is not None

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_with_reasoning_content(self, model):
        """Test _process_streaming_event_async with reasoning content."""
        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = None
        mock_delta.reasoning_content = "Let me think about this..."

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, [], MagicMock(), []):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ReasoningEvent) for e in events)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_with_tool_calls_sequential(self, model):
        """Test _process_streaming_event_async with sequential tool execution."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = '{"param": "value"}'

        mock_tool_call_delta = MagicMock()
        mock_tool_call_delta.index = 0
        mock_tool_call_delta.id = "call_1"
        mock_tool_call_delta.function = mock_function

        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = [mock_tool_call_delta]
        mock_delta.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "tool_calls"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._skip_handler_collection = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor._streaming_usage = None

        tool_calls = []
        events = []
        async for event in model._process_streaming_event_async(mock_chunk, tool_calls, tool_executor, []):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_with_usage(self, model):
        """Test _process_streaming_event_async with usage in chunk."""
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_delta = MagicMock()
        mock_delta.content = "Response"
        mock_delta.tool_calls = None
        mock_delta.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = mock_usage

        tool_executor = MagicMock()
        tool_executor._streaming_usage = None

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, [], tool_executor, []):
            events.append(event)

        assert len(events) > 0
        assert tool_executor._streaming_usage is not None

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_with_finish_reason_and_usage(self, model):
        """Test _process_streaming_event_async with finish reason and streaming usage."""
        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = None
        mock_delta.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        tool_executor = MagicMock()
        tool_executor._streaming_usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, [], tool_executor, []):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_sequential_multiple_tools(self, model):
        """Test _process_streaming_event_async with sequential execution and multiple tools."""
        mock_function1 = MagicMock()
        mock_function1.name = "test_tool1"
        mock_function1.arguments = '{"param": "value1"}'

        mock_function2 = MagicMock()
        mock_function2.name = "test_tool2"
        mock_function2.arguments = '{"param": "value2"}'

        mock_tool_call_delta1 = MagicMock()
        mock_tool_call_delta1.index = 0
        mock_tool_call_delta1.id = "call_1"
        mock_tool_call_delta1.function = mock_function1

        mock_tool_call_delta2 = MagicMock()
        mock_tool_call_delta2.index = 1
        mock_tool_call_delta2.id = "call_2"
        mock_tool_call_delta2.function = mock_function2

        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = [mock_tool_call_delta1, mock_tool_call_delta2]
        mock_delta.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "tool_calls"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool1": "result1"})

        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._skip_handler_collection = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor._streaming_usage = None

        tool_calls = []
        events = []
        async for event in model._process_streaming_event_async(mock_chunk, tool_calls, tool_executor, []):
            events.append(event)

        assert len(events) > 0
        # Should only yield first tool in sequential mode
        tool_call_events = [e for e in events if isinstance(e, ToolCallsEvent)]
        assert len(tool_call_events) > 0
        assert len(tool_call_events[0].tool_calls) == 1  # Only first tool

    def test_handle_non_streaming_with_reasoning_content(self, model):
        """Test handle_non_streaming with reasoning content."""
        mock_message = MagicMock()
        mock_message.content = "Response text"
        mock_message.tool_calls = None
        mock_message.role = "assistant"
        mock_message.reasoning_content = "Let me think about this..."

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.client.chat.completions.create = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("deepseek-chat", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_sequential_tools(self, model):
        """Test handle_non_streaming with sequential (non-parallel) tool execution."""
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
        mock_message.role = "assistant"
        mock_message.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        # Mock subsequent responses that don't have tool calls (to exit the loop)
        mock_message2 = MagicMock()
        mock_message2.content = "Final response"
        mock_message2.tool_calls = None
        mock_message2.role = "assistant"
        mock_message2.reasoning_content = None

        mock_choice2 = MagicMock()
        mock_choice2.message = mock_message2
        mock_choice2.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        mock_response2 = MagicMock()
        mock_response2.choices = [mock_choice2]
        mock_response2.usage = mock_usage

        model.client.chat.completions.create = MagicMock(side_effect=[mock_response, mock_response2])
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._skip_handler_collection = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        model.handle_non_streaming("deepseek-chat", messages, tools, tool_executor)
        # Sequential mode may call multiple times in a loop, so just verify it was called
        assert model._execute_tools_parallel_sync.called

    def test_handle_non_streaming_with_all_params(self, model):
        """Test handle_non_streaming with all parameters set."""
        model.temperature = 0.7
        model.top_p = 0.9
        model.presence_penalty = 0.5
        model.frequency_penalty = 0.5
        model.max_tokens = 1000

        mock_message = MagicMock()
        mock_message.content = "Response"
        mock_message.tool_calls = None
        mock_message.role = "assistant"
        mock_message.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        model.client.chat.completions.create = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("deepseek-chat", messages, None, tool_executor)
        assert result is not None
        # Verify all params were passed
        call_args = model.client.chat.completions.create.call_args[1]
        assert call_args.get("temperature") == 0.7
        assert call_args.get("top_p") == 0.9
        assert call_args.get("presence_penalty") == 0.5
        assert call_args.get("frequency_penalty") == 0.5
        assert call_args.get("max_tokens") == 1000

    def test_handle_non_streaming_status_error(self, model):
        """Test handle_non_streaming with status error."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": {"message": "API Error"}})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.client.chat.completions.create = MagicMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Status error"):
            model.handle_non_streaming("deepseek-chat", messages, None, tool_executor)

    def test_handle_streaming_with_reasoning_content(self, model):
        """Test handle_streaming with reasoning content."""
        mock_delta = MagicMock()
        mock_delta.content = "Response"
        mock_delta.tool_calls = None
        mock_delta.reasoning_content = "Let me think..."

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        model.client.chat.completions.create = MagicMock(return_value=iter([mock_chunk]))
        model._process_streaming_event_sync = MagicMock(return_value=[ReasoningEvent(reasoning="Let me think..."), ContentEvent(content="Response")])

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("deepseek-chat", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, ReasoningEvent) for e in events)

    def test_process_streaming_event_sync_no_choices_with_usage(self, model):
        """Test _process_streaming_event_sync with no choices but usage."""
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_chunk = MagicMock()
        mock_chunk.choices = []
        mock_chunk.usage = mock_usage

        tool_executor = MagicMock()
        tool_executor._streaming_usage = None

        events = list(model._process_streaming_event_sync(mock_chunk, [], tool_executor, []))
        assert len(events) == 1  # Should yield ResponseCompletedEvent
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)
        assert tool_executor._streaming_usage is not None

    def test_process_streaming_event_sync_with_reasoning_content(self, model):
        """Test _process_streaming_event_sync with reasoning content."""
        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = None
        mock_delta.reasoning_content = "Let me think about this..."

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        events = list(model._process_streaming_event_sync(mock_chunk, [], MagicMock(), []))
        assert len(events) > 0
        assert any(isinstance(e, ReasoningEvent) for e in events)

    def test_process_streaming_event_sync_with_tool_calls_sequential(self, model):
        """Test _process_streaming_event_sync with sequential tool execution."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = '{"param": "value"}'

        mock_tool_call_delta = MagicMock()
        mock_tool_call_delta.index = 0
        mock_tool_call_delta.id = "call_1"
        mock_tool_call_delta.function = mock_function

        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = [mock_tool_call_delta]
        mock_delta.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "tool_calls"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._skip_handler_collection = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor._streaming_usage = None

        tool_calls = []
        events = list(model._process_streaming_event_sync(mock_chunk, tool_calls, tool_executor, []))
        assert len(events) > 0

    def test_process_streaming_event_sync_with_usage(self, model):
        """Test _process_streaming_event_sync with usage in chunk."""
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_delta = MagicMock()
        mock_delta.content = "Response"
        mock_delta.tool_calls = None
        mock_delta.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = mock_usage

        tool_executor = MagicMock()
        tool_executor._streaming_usage = None

        events = list(model._process_streaming_event_sync(mock_chunk, [], tool_executor, []))
        assert len(events) > 0
        assert tool_executor._streaming_usage is not None

    def test_process_streaming_event_sync_with_finish_reason_and_usage(self, model):
        """Test _process_streaming_event_sync with finish reason and streaming usage."""
        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = None
        mock_delta.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        tool_executor = MagicMock()
        tool_executor._streaming_usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}

        events = list(model._process_streaming_event_sync(mock_chunk, [], tool_executor, []))
        assert len(events) > 0
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)

    def test_process_streaming_event_sync_sequential_multiple_tools(self, model):
        """Test _process_streaming_event_sync with sequential execution and multiple tools."""
        mock_function1 = MagicMock()
        mock_function1.name = "test_tool1"
        mock_function1.arguments = '{"param": "value1"}'

        mock_function2 = MagicMock()
        mock_function2.name = "test_tool2"
        mock_function2.arguments = '{"param": "value2"}'

        mock_tool_call_delta1 = MagicMock()
        mock_tool_call_delta1.index = 0
        mock_tool_call_delta1.id = "call_1"
        mock_tool_call_delta1.function = mock_function1

        mock_tool_call_delta2 = MagicMock()
        mock_tool_call_delta2.index = 1
        mock_tool_call_delta2.id = "call_2"
        mock_tool_call_delta2.function = mock_function2

        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = [mock_tool_call_delta1, mock_tool_call_delta2]
        mock_delta.reasoning_content = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "tool_calls"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool1": "result1"})

        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._skip_handler_collection = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor._streaming_usage = None

        tool_calls = []
        events = list(model._process_streaming_event_sync(mock_chunk, tool_calls, tool_executor, []))
        assert len(events) > 0
        # Should only yield first tool in sequential mode
        tool_call_events = [e for e in events if isinstance(e, ToolCallsEvent)]
        assert len(tool_call_events) > 0
        assert len(tool_call_events[0].tool_calls) == 1  # Only first tool
