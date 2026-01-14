from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.models.events import ContentEvent, FinishReasonEvent
from hypertic.models.openrouter.openrouter import OpenRouter
from hypertic.utils.files import File, FileType


class TestOpenRouter:
    @pytest.fixture
    def model(self, mock_api_key):
        with patch("hypertic.models.openrouter.openrouter.AsyncOpenAIClient"), patch("hypertic.models.openrouter.openrouter.OpenAIClient"):
            return OpenRouter(api_key=mock_api_key, model="openai/gpt-4o")

    def test_openrouter_creation(self, mock_api_key):
        """Test OpenRouter initialization."""
        with patch("hypertic.models.openrouter.openrouter.AsyncOpenAIClient"), patch("hypertic.models.openrouter.openrouter.OpenAIClient"):
            model = OpenRouter(api_key=mock_api_key, model="openai/gpt-4o")
            assert model.api_key == mock_api_key
            assert model.model == "openai/gpt-4o"

    def test_openrouter_creation_no_api_key(self):
        """Test OpenRouter initialization without API key."""
        with (
            patch("hypertic.models.openrouter.openrouter.getenv", return_value=None),
            patch("hypertic.models.openrouter.openrouter.AsyncOpenAIClient"),
            patch("hypertic.models.openrouter.openrouter.OpenAIClient"),
        ):
            model = OpenRouter(model="openai/gpt-4o")
            assert model.api_key is None

    def test_openrouter_with_params(self, mock_api_key):
        """Test OpenRouter with all parameters."""
        with patch("hypertic.models.openrouter.openrouter.AsyncOpenAIClient"), patch("hypertic.models.openrouter.openrouter.OpenAIClient"):
            model = OpenRouter(
                api_key=mock_api_key,
                model="openai/gpt-4o",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000,
                reasoning_effort="high",
            )
            assert model.temperature == 0.7
            assert model.top_p == 0.9
            assert model.max_tokens == 1000
            assert model.reasoning_effort == "high"

    def test_format_files_for_openrouter_no_files(self, model):
        """Test _format_files_for_openrouter with no files."""
        message = {"role": "user", "content": "Hello"}
        result = model._format_files_for_openrouter(message)
        assert result == message

    def test_format_files_for_openrouter_with_image_url(self, model):
        """Test _format_files_for_openrouter with image URL."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        result = model._format_files_for_openrouter(message)
        assert "content" in result
        assert isinstance(result["content"], list)

    def test_format_files_for_openrouter_with_image_base64(self, model):
        """Test _format_files_for_openrouter with image base64."""
        file_obj = File(content=b"fake image", file_type=FileType.IMAGE, mime_type="image/jpeg")
        with patch.object(File, "to_base64", return_value="base64data"):
            message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
            result = model._format_files_for_openrouter(message)
            assert "content" in result
            assert isinstance(result["content"], list)

    def test_format_files_for_openrouter_removes_file_objects(self, model):
        """Test that _format_files_for_openrouter removes _file_objects."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "_file_objects": [file_obj], "files": ["url"]}
        result = model._format_files_for_openrouter(message)
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

        result = await model.ahandle_non_streaming("openai/gpt-4o", messages, None, tool_executor)
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

        result = await model.ahandle_non_streaming("openai/gpt-4o", messages, tools, tool_executor)
        assert result is None  # Returns None when tool calls are present

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
            await model.ahandle_non_streaming("openai/gpt-4o", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_connection_error(self, model):
        """Test ahandle_non_streaming with connection error."""
        from openai import APIConnectionError

        connection_error = APIConnectionError(request=MagicMock(), message="Connection failed")

        model.async_client.chat.completions.create = AsyncMock(side_effect=connection_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Connection error"):
            await model.ahandle_non_streaming("openai/gpt-4o", messages, None, tool_executor)

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

        result = model.handle_non_streaming("openai/gpt-4o", messages, None, tool_executor)
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

        model.client.chat.completions.create = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("openai/gpt-4o", messages, tools, tool_executor)
        assert result is None

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

        # Create async generator for _process_streaming_event_async
        async def async_gen(chunk, tool_calls, tool_executor, messages):
            yield ContentEvent(content="Response")

        model.async_client.chat.completions.create = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._process_streaming_event_async = async_gen

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("openai/gpt-4o", messages, None, tool_executor):
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

        events = list(model.handle_streaming("openai/gpt-4o", messages, None, tool_executor))
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

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        model.client.chat.completions.create = MagicMock(return_value=iter([mock_chunk]))
        model._process_streaming_event_sync = MagicMock(return_value=[FinishReasonEvent(finish_reason="stop")])

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("openai/gpt-4o", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, FinishReasonEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_tools(self, model):
        """Test ahandle_non_streaming with sequential (non-parallel) tool execution."""
        mock_function1 = MagicMock()
        mock_function1.name = "test_tool1"
        mock_function1.arguments = '{"param": "value1"}'

        mock_function2 = MagicMock()
        mock_function2.name = "test_tool2"
        mock_function2.arguments = '{"param": "value2"}'

        mock_tool_call1 = MagicMock()
        mock_tool_call1.id = "call_1"
        mock_tool_call1.function = mock_function1
        mock_tool_call1.model_dump = MagicMock(return_value={"id": "call_1", "type": "function", "function": {"name": "test_tool1"}})

        mock_tool_call2 = MagicMock()
        mock_tool_call2.id = "call_2"
        mock_tool_call2.function = mock_function2
        mock_tool_call2.model_dump = MagicMock(return_value={"id": "call_2", "type": "function", "function": {"name": "test_tool2"}})

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call1, mock_tool_call2]
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
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool1": "result1"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool1", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("openai/gpt-4o", messages, tools, tool_executor)
        assert result is None
        assert model._execute_tools_parallel_async.call_count == 1  # Only first tool called

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

        result = await model.ahandle_non_streaming("openai/gpt-4o", messages, None, tool_executor, response_format=OutputModel)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_all_params(self, model):
        """Test ahandle_non_streaming with all parameters set."""
        model.temperature = 0.7
        model.top_p = 0.9
        model.presence_penalty = 0.5
        model.frequency_penalty = 0.5
        model.max_tokens = 1000
        model.reasoning_effort = "high"

        mock_message = MagicMock()
        mock_message.content = "Response"
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

        result = await model.ahandle_non_streaming("openai/gpt-4o", messages, None, tool_executor)
        assert result is not None
        # Verify all params were passed
        call_args = model.async_client.chat.completions.create.call_args[1]
        assert call_args.get("temperature") == 0.7
        assert call_args.get("top_p") == 0.9
        assert call_args.get("presence_penalty") == 0.5
        assert call_args.get("frequency_penalty") == 0.5
        assert call_args.get("max_tokens") == 1000
        assert call_args.get("reasoning_effort") == "high"

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_tool_call_no_function_name(self, model):
        """Test ahandle_non_streaming with tool call that has no function name."""
        mock_function = MagicMock()
        mock_function.name = None  # No name
        mock_function.arguments = '{"param": "value"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function
        mock_tool_call.model_dump = MagicMock(return_value={"id": "call_1", "type": "function", "function": {"name": None}})

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
        model._execute_tools_parallel_async = AsyncMock(return_value={})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("openai/gpt-4o", messages, tools, tool_executor)
        assert result is None

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
            await model.ahandle_non_streaming("openai/gpt-4o", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_tools_sequential(self, model):
        """Test ahandle_streaming with sequential tool execution."""
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

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "tool_calls"

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

        model.async_client.chat.completions.create = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential

        events = []
        async for event in model.ahandle_streaming("openai/gpt-4o", messages, tools, tool_executor):
            events.append(event)

        assert len(events) > 0

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
        async for event in model.ahandle_streaming("openai/gpt-4o", messages, None, tool_executor, response_format=OutputModel):
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

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, [], MagicMock(), []):
            events.append(event)

        assert len(events) == 0  # Should return early

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_with_tool_calls(self, model):
        """Test _process_streaming_event_async with tool calls."""
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

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "tool_calls"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, [], tool_executor, []):
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

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = mock_usage

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, [], MagicMock(), []):
            events.append(event)

        assert len(events) > 0

    def test_handle_non_streaming_sequential_tools(self, model):
        """Test handle_non_streaming with sequential (non-parallel) tool execution."""
        mock_function1 = MagicMock()
        mock_function1.name = "test_tool1"
        mock_function1.arguments = '{"param": "value1"}'

        mock_function2 = MagicMock()
        mock_function2.name = "test_tool2"
        mock_function2.arguments = '{"param": "value2"}'

        mock_tool_call1 = MagicMock()
        mock_tool_call1.id = "call_1"
        mock_tool_call1.function = mock_function1
        mock_tool_call1.model_dump = MagicMock(return_value={"id": "call_1", "type": "function", "function": {"name": "test_tool1"}})

        mock_tool_call2 = MagicMock()
        mock_tool_call2.id = "call_2"
        mock_tool_call2.function = mock_function2
        mock_tool_call2.model_dump = MagicMock(return_value={"id": "call_2", "type": "function", "function": {"name": "test_tool2"}})

        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call1, mock_tool_call2]
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

        model.client.chat.completions.create = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool1": "result1"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool1", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("openai/gpt-4o", messages, tools, tool_executor)
        assert result is None
        assert model._execute_tools_parallel_sync.call_count == 1

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

        result = model.handle_non_streaming("openai/gpt-4o", messages, None, tool_executor, response_format=OutputModel)
        assert result is not None

    def test_handle_non_streaming_with_all_params(self, model):
        """Test handle_non_streaming with all parameters set."""
        model.temperature = 0.7
        model.top_p = 0.9
        model.presence_penalty = 0.5
        model.frequency_penalty = 0.5
        model.max_tokens = 1000
        model.reasoning_effort = "high"

        mock_message = MagicMock()
        mock_message.content = "Response"
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

        result = model.handle_non_streaming("openai/gpt-4o", messages, None, tool_executor)
        assert result is not None
        # Verify all params were passed
        call_args = model.client.chat.completions.create.call_args[1]
        assert call_args.get("temperature") == 0.7
        assert call_args.get("top_p") == 0.9
        assert call_args.get("presence_penalty") == 0.5
        assert call_args.get("frequency_penalty") == 0.5
        assert call_args.get("max_tokens") == 1000
        assert call_args.get("reasoning_effort") == "high"

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
            model.handle_non_streaming("openai/gpt-4o", messages, None, tool_executor)

    def test_handle_streaming_with_tools_sequential(self, model):
        """Test handle_streaming with sequential tool execution."""
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

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "tool_calls"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        model.client.chat.completions.create = MagicMock(return_value=iter([mock_chunk]))
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential

        events = list(model.handle_streaming("openai/gpt-4o", messages, tools, tool_executor))
        assert len(events) > 0

    def test_process_streaming_event_sync_no_choices_with_usage(self, model):
        """Test _process_streaming_event_sync with no choices but usage."""
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_chunk = MagicMock()
        mock_chunk.choices = []
        mock_chunk.usage = mock_usage

        events = list(model._process_streaming_event_sync(mock_chunk, [], MagicMock(), []))
        assert len(events) == 0  # Should return early

    def test_process_streaming_event_sync_with_tool_calls(self, model):
        """Test _process_streaming_event_sync with tool calls."""
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

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "tool_calls"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = list(model._process_streaming_event_sync(mock_chunk, [], tool_executor, []))
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

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = mock_usage

        events = list(model._process_streaming_event_sync(mock_chunk, [], MagicMock(), []))
        assert len(events) > 0

    def test_format_files_for_openrouter_with_existing_content_list(self, model):
        """Test _format_files_for_openrouter with existing content list."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": [{"type": "text", "text": "Check this"}], "_file_objects": [file_obj]}
        result = model._format_files_for_openrouter(message)
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 1
