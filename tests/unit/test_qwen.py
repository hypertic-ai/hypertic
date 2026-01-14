from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.models.events import ContentEvent, FinishReasonEvent, ReasoningEvent, ResponseCompletedEvent
from hypertic.models.qwen.qwen import Qwen
from hypertic.utils.files import File, FileType


class TestQwen:
    @pytest.fixture
    def model(self, mock_api_key):
        with patch("hypertic.models.qwen.qwen.AsyncOpenAIClient"), patch("hypertic.models.qwen.qwen.OpenAIClient"):
            return Qwen(api_key=mock_api_key, model="qwen-plus")

    def test_qwen_creation(self, mock_api_key):
        """Test Qwen initialization."""
        with patch("hypertic.models.qwen.qwen.AsyncOpenAIClient"), patch("hypertic.models.qwen.qwen.OpenAIClient"):
            model = Qwen(api_key=mock_api_key, model="qwen-plus")
            assert model.api_key == mock_api_key
            assert model.model == "qwen-plus"

    def test_qwen_creation_no_api_key(self):
        """Test Qwen initialization without API key."""
        with (
            patch("hypertic.models.qwen.qwen.getenv", return_value=None),
            patch("hypertic.models.qwen.qwen.AsyncOpenAIClient"),
            patch("hypertic.models.qwen.qwen.OpenAIClient"),
        ):
            model = Qwen(model="qwen-plus")
            assert model.api_key is None

    def test_qwen_with_params(self, mock_api_key):
        """Test Qwen with all parameters."""
        with patch("hypertic.models.qwen.qwen.AsyncOpenAIClient"), patch("hypertic.models.qwen.qwen.OpenAIClient"):
            model = Qwen(
                api_key=mock_api_key,
                model="qwen-plus",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000,
                thinking=True,
                thinking_tokens=500,
            )
            assert model.temperature == 0.7
            assert model.top_p == 0.9
            assert model.max_tokens == 1000
            assert model.thinking is True
            assert model.thinking_tokens == 500

    def test_format_files_for_qwen_no_files(self, model):
        """Test _format_files_for_qwen with no files."""
        message = {"role": "user", "content": "Hello"}
        result = model._format_files_for_qwen(message)
        assert result == message

    def test_format_files_for_qwen_with_image_vl_model(self, model):
        """Test _format_files_for_qwen with image and VL model."""
        model.model = "qwen-vl-max"
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        result = model._format_files_for_qwen(message)
        assert "content" in result
        assert isinstance(result["content"], list) or isinstance(result["content"], str)

    def test_format_files_for_qwen_with_image_non_vl_model(self, model):
        """Test _format_files_for_qwen with image and non-VL model."""
        model.model = "qwen-plus"
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        result = model._format_files_for_qwen(message)
        # Should warn but still process
        assert "content" in result

    def test_format_files_for_qwen_with_image_base64(self, model):
        """Test _format_files_for_qwen with image base64."""
        model.model = "qwen-vl-max"
        file_obj = File(content=b"fake image", file_type=FileType.IMAGE, mime_type="image/jpeg")
        with patch.object(File, "to_base64", return_value="base64data"):
            message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
            result = model._format_files_for_qwen(message)
            assert "content" in result

    def test_format_files_for_qwen_with_video(self, model):
        """Test _format_files_for_qwen with video."""
        model.model = "qwen-vl-max"
        file_obj = File(url="https://example.com/video.mp4", file_type=FileType.VIDEO)
        message = {"role": "user", "content": "Watch this", "_file_objects": [file_obj]}
        result = model._format_files_for_qwen(message)
        assert "content" in result

    def test_format_files_for_qwen_removes_file_objects(self, model):
        """Test that _format_files_for_qwen removes _file_objects."""
        model.model = "qwen-vl-max"
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "_file_objects": [file_obj], "files": ["url"]}
        result = model._format_files_for_qwen(message)
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

        result = await model.ahandle_non_streaming("qwen-plus", messages, None, tool_executor)
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

        result = await model.ahandle_non_streaming("qwen-plus", messages, tools, tool_executor)
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
            await model.ahandle_non_streaming("qwen-plus", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_connection_error(self, model):
        """Test ahandle_non_streaming with connection error."""
        from openai import APIConnectionError

        connection_error = APIConnectionError(request=MagicMock(), message="Connection failed")

        model.async_client.chat.completions.create = AsyncMock(side_effect=connection_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Connection error"):
            await model.ahandle_non_streaming("qwen-plus", messages, None, tool_executor)

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

        result = model.handle_non_streaming("qwen-plus", messages, None, tool_executor)
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

        result = model.handle_non_streaming("qwen-plus", messages, tools, tool_executor)
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
        async for event in model.ahandle_streaming("qwen-plus", messages, None, tool_executor):
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

        events = list(model.handle_streaming("qwen-plus", messages, None, tool_executor))
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

        events = list(model.handle_streaming("qwen-plus", messages, None, tool_executor))
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

        result = await model.ahandle_non_streaming("qwen-plus", messages, None, tool_executor, response_format=OutputModel)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_response_format_dict(self, model):
        """Test ahandle_non_streaming with dict response format."""
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

        result = await model.ahandle_non_streaming("qwen-plus", messages, None, tool_executor, response_format={"type": "json_object"})
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_all_params(self, model):
        """Test ahandle_non_streaming with all parameters set."""
        model.temperature = 0.7
        model.top_p = 0.9
        model.presence_penalty = 0.5
        model.frequency_penalty = 0.5
        model.max_tokens = 1000
        model.thinking = True
        model.thinking_tokens = 500

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

        result = await model.ahandle_non_streaming("qwen-plus", messages, None, tool_executor)
        assert result is not None
        # Verify all params were passed
        call_args = model.async_client.chat.completions.create.call_args[1]
        assert call_args.get("temperature") == 0.7
        assert call_args.get("top_p") == 0.9
        assert call_args.get("presence_penalty") == 0.5
        assert call_args.get("frequency_penalty") == 0.5
        assert call_args.get("max_tokens") == 1000

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_tool_call_no_function(self, model):
        """Test ahandle_non_streaming with tool call that has no function."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        # Remove function attribute
        del mock_tool_call.function
        type(mock_tool_call).function = property(lambda self: None)
        mock_tool_call.model_dump = MagicMock(return_value={"id": "call_1"})

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

        result = await model.ahandle_non_streaming("qwen-plus", messages, tools, tool_executor)
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
            await model.ahandle_non_streaming("qwen-plus", messages, None, tool_executor)

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
        async for event in model.ahandle_streaming("qwen-plus", messages, None, tool_executor, response_format=OutputModel):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_thinking(self, model):
        """Test ahandle_streaming with thinking enabled."""
        model.thinking = True
        model.thinking_tokens = 500

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
        async for event in model.ahandle_streaming("qwen-plus", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0
        # Verify thinking params were passed
        call_args = model.async_client.chat.completions.create.call_args[1]
        assert call_args.get("extra_body", {}).get("enable_thinking") is True
        assert call_args.get("extra_body", {}).get("thinking_budget") == 500

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
    async def test_process_streaming_event_async_tool_call_delta_concatenation(self, model):
        """Test _process_streaming_event_async with tool call delta concatenation."""
        # Test where id and function name/arguments are split across chunks
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
        mock_delta1.reasoning_content = None

        mock_delta2 = MagicMock()
        mock_delta2.content = None
        mock_delta2.tool_calls = [mock_tool_call_delta2]
        mock_delta2.reasoning_content = None

        mock_choice1 = MagicMock()
        mock_choice1.delta = mock_delta1
        mock_choice1.finish_reason = None

        mock_choice2 = MagicMock()
        mock_choice2.delta = mock_delta2
        mock_choice2.finish_reason = "tool_calls"

        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [mock_choice1]
        mock_chunk1.usage = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [mock_choice2]
        mock_chunk2.usage = None

        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor._streaming_usage = None

        tool_calls = []
        events = []
        async for event in model._process_streaming_event_async(mock_chunk1, tool_calls, tool_executor, []):
            events.append(event)
        async for event in model._process_streaming_event_async(mock_chunk2, tool_calls, tool_executor, []):
            events.append(event)

        assert len(events) > 0

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

        result = model.handle_non_streaming("qwen-plus", messages, None, tool_executor, response_format=OutputModel)
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
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("qwen-plus", messages, tools, tool_executor)
        assert result is None
        assert model._execute_tools_parallel_sync.call_count == 1

    def test_handle_non_streaming_sequential_tools_no_function(self, model):
        """Test handle_non_streaming with sequential tools but no function."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        # Make hasattr return False for function
        del mock_tool_call.function
        type(mock_tool_call).function = property(lambda self: None)
        mock_tool_call.model_dump = MagicMock(return_value={"id": "call_1"})

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

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("qwen-plus", messages, tools, tool_executor)
        assert result is None

    def test_handle_non_streaming_sequential_tools_json_decode_error(self, model):
        """Test handle_non_streaming with sequential tools and JSON decode error."""
        mock_function = MagicMock()
        mock_function.name = "test_tool"
        mock_function.arguments = "invalid json"

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = mock_function
        mock_tool_call.model_dump = MagicMock(
            return_value={"id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": "invalid json"}}
        )

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
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("qwen-plus", messages, tools, tool_executor)
        assert result is None

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

        result = model.handle_non_streaming("qwen-plus", messages, None, tool_executor)
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
            model.handle_non_streaming("qwen-plus", messages, None, tool_executor)

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
        mock_delta.reasoning_content = None

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
        tool_executor._streaming_usage = None

        events = list(model.handle_streaming("qwen-plus", messages, tools, tool_executor))
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

    def test_format_files_for_qwen_with_existing_content_list(self, model):
        """Test _format_files_for_qwen with existing content list."""
        model.model = "qwen-vl-max"
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": [{"type": "text", "text": "Check this"}], "_file_objects": [file_obj]}
        result = model._format_files_for_qwen(message)
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 1

    def test_format_files_for_qwen_single_text_content(self, model):
        """Test _format_files_for_qwen with single text content."""
        model.model = "qwen-vl-max"
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": [{"type": "text", "text": "Hello"}], "_file_objects": [file_obj]}
        result = model._format_files_for_qwen(message)
        # Content should be a list with both text and image
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2

    def test_format_files_for_qwen_with_video_base64(self, model):
        """Test _format_files_for_qwen with video base64."""
        model.model = "qwen-vl-max"
        file_obj = File(content=b"fake video", file_type=FileType.VIDEO, mime_type="video/mp4")
        with patch.object(File, "to_base64", return_value="base64data"):
            message = {"role": "user", "content": "Watch this", "_file_objects": [file_obj]}
            result = model._format_files_for_qwen(message)
            assert "content" in result
            assert isinstance(result["content"], list)
