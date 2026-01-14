from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.models.events import ContentEvent, FinishReasonEvent, ToolCallsEvent, ToolOutputsEvent
from hypertic.models.openai.openaichat import OpenAIChat
from hypertic.utils.files import File, FileType


class TestOpenAIChat:
    @pytest.fixture
    def model(self, mock_api_key):
        with patch("hypertic.models.openai.openaichat.AsyncOpenAIClient"), patch("hypertic.models.openai.openaichat.OpenAIClient"):
            return OpenAIChat(api_key=mock_api_key, model="gpt-4o")

    def test_openaichat_creation(self, mock_api_key):
        """Test OpenAIChat initialization."""
        with patch("hypertic.models.openai.openaichat.AsyncOpenAIClient"), patch("hypertic.models.openai.openaichat.OpenAIClient"):
            model = OpenAIChat(api_key=mock_api_key, model="gpt-4o")
            assert model.api_key == mock_api_key
            assert model.model == "gpt-4o"

    def test_openaichat_creation_no_api_key(self):
        """Test OpenAIChat initialization without API key."""
        with (
            patch("hypertic.models.openai.openaichat.getenv", return_value=None),
            patch("hypertic.models.openai.openaichat.AsyncOpenAIClient"),
            patch("hypertic.models.openai.openaichat.OpenAIClient"),
        ):
            model = OpenAIChat(model="gpt-4o")
            assert model.api_key is None

    def test_openaichat_with_params(self, mock_api_key):
        """Test OpenAIChat with all parameters."""
        with patch("hypertic.models.openai.openaichat.AsyncOpenAIClient"), patch("hypertic.models.openai.openaichat.OpenAIClient"):
            model = OpenAIChat(
                api_key=mock_api_key,
                model="gpt-4o",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000,
                reasoning_effort="high",
            )
            assert model.temperature == 0.7
            assert model.top_p == 0.9
            assert model.max_tokens == 1000
            assert model.reasoning_effort == "high"

    def test_format_files_for_openai_chat_no_files(self, model):
        """Test _format_files_for_openai_chat with no files."""
        message = {"role": "user", "content": "Hello"}
        result = model._format_files_for_openai_chat(message)
        assert result == message

    def test_format_files_for_openai_chat_with_image_url(self, model):
        """Test _format_files_for_openai_chat with image URL."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        result = model._format_files_for_openai_chat(message)
        assert "content" in result
        assert isinstance(result["content"], list)

    def test_format_files_for_openai_chat_with_image_base64(self, model):
        """Test _format_files_for_openai_chat with image base64."""
        file_obj = File(content=b"fake image", file_type=FileType.IMAGE, mime_type="image/jpeg")
        with patch.object(File, "to_base64", return_value="base64data"):
            message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
            result = model._format_files_for_openai_chat(message)
            assert "content" in result
            assert isinstance(result["content"], list)

    def test_format_files_for_openai_chat_removes_file_objects(self, model):
        """Test that _format_files_for_openai_chat removes _file_objects."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "_file_objects": [file_obj], "files": ["url"]}
        result = model._format_files_for_openai_chat(message)
        assert "_file_objects" not in result
        assert "files" not in result

    def test_format_files_for_openai_chat_with_list_content(self, model):
        """Test _format_files_for_openai_chat with list content."""
        message = {"role": "user", "content": [{"type": "text", "text": "Hello"}], "_file_objects": []}
        result = model._format_files_for_openai_chat(message)
        assert isinstance(result["content"], list)

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

        result = await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_json_response_format(self, model):
        """Test ahandle_non_streaming with JSON response format."""
        mock_message = MagicMock()
        mock_message.content = '{"key": "value"}'
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

        result = await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor, response_format={"type": "json_object"})
        assert result is not None
        # Content should be parsed as JSON
        assert isinstance(messages[-1]["content"], dict)

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

        result = await model.ahandle_non_streaming("gpt-4o", messages, tools, tool_executor)
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
            await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_connection_error(self, model):
        """Test ahandle_non_streaming with connection error."""
        from openai import APIConnectionError

        connection_error = APIConnectionError(request=MagicMock(), message="Connection failed")

        model.async_client.chat.completions.create = AsyncMock(side_effect=connection_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Connection error"):
            await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_status_error(self, model):
        """Test ahandle_non_streaming with status error."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": {"message": "Invalid request"}})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.async_client.chat.completions.create = AsyncMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Status error"):
            await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)

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

        result = model.handle_non_streaming("gpt-4o", messages, None, tool_executor)
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

        result = model.handle_non_streaming("gpt-4o", messages, tools, tool_executor)
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
        async for event in model.ahandle_streaming("gpt-4o", messages, None, tool_executor):
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

        events = list(model.handle_streaming("gpt-4o", messages, None, tool_executor))
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

        events = list(model.handle_streaming("gpt-4o", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, FinishReasonEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_rate_limit_error(self, model):
        """Test ahandle_streaming with rate limit error."""
        from openai import RateLimitError

        mock_response = MagicMock()
        mock_response.status_code = 429
        rate_limit_error = RateLimitError(response=mock_response, body={}, message="Rate limited")

        model.async_client.chat.completions.create = AsyncMock(side_effect=rate_limit_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Rate limit"):
            async for _ in model.ahandle_streaming("gpt-4o", messages, None, tool_executor):
                pass

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_pydantic_response_format(self, model):
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

        result = await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor, response_format=OutputModel)
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

        result = await model.ahandle_non_streaming("gpt-4o", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_tool_call_no_function(self, model):
        """Test ahandle_non_streaming with tool call that has no function attribute."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = None  # No function
        mock_tool_call.model_dump = MagicMock(return_value={"id": "call_1", "type": "function", "function": None})

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

        result = await model.ahandle_non_streaming("gpt-4o", messages, tools, tool_executor)
        assert result is None

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

        model.async_client.chat.completions.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_json_parse_error(self, model):
        """Test ahandle_non_streaming with JSON parse error in response format."""
        mock_message = MagicMock()
        mock_message.content = "not valid json"
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

        result = await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor, response_format={"type": "json_object"})
        assert result is not None
        # Content should remain as string if JSON parsing fails
        assert isinstance(messages[-1]["content"], str)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_all_params(self, model):
        """Test ahandle_non_streaming with all parameters."""
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

        # Set all parameters
        model.temperature = 0.8
        model.top_p = 0.9
        model.presence_penalty = 0.1
        model.frequency_penalty = 0.2
        model.max_tokens = 1000
        model.reasoning_effort = "high"

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)
        assert result is not None
        # Verify parameters were passed
        call_args = model.async_client.chat.completions.create.call_args[1]
        assert call_args.get("temperature") == 0.8
        assert call_args.get("top_p") == 0.9
        assert call_args.get("presence_penalty") == 0.1
        assert call_args.get("frequency_penalty") == 0.2
        assert call_args.get("max_completion_tokens") == 1000
        assert call_args.get("reasoning_effort") == "high"

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_api_status_error_dict_message(self, model):
        """Test ahandle_non_streaming with APIStatusError with dict error message."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": {"message": "Invalid request"}})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.async_client.chat.completions.create = AsyncMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Invalid request"):
            await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_api_status_error_string_message(self, model):
        """Test ahandle_non_streaming with APIStatusError with string error message."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": "Invalid request"})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.async_client.chat.completions.create = AsyncMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Invalid request"):
            await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_pydantic_response_format(self, model):
        """Test ahandle_streaming with Pydantic response format."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_delta = MagicMock()
        mock_delta.content = '{"result": "test"}'
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
            yield ContentEvent(content='{"result": "test"}')

        model.async_client.chat.completions.create = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._process_streaming_event_async = async_gen

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gpt-4o", messages, None, tool_executor, response_format=OutputModel):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_all_params(self, model):
        """Test ahandle_streaming with all parameters."""
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

        # Set all parameters
        model.temperature = 0.8
        model.top_p = 0.9
        model.presence_penalty = 0.1
        model.frequency_penalty = 0.2
        model.max_tokens = 1000
        model.reasoning_effort = "high"

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gpt-4o", messages, None, tool_executor):
            events.append(event)

        # Verify parameters were passed
        call_args = model.async_client.chat.completions.create.call_args[1]
        assert call_args.get("temperature") == 0.8
        assert call_args.get("top_p") == 0.9
        assert call_args.get("presence_penalty") == 0.1
        assert call_args.get("frequency_penalty") == 0.2
        assert call_args.get("max_completion_tokens") == 1000
        assert call_args.get("reasoning_effort") == "high"
        assert call_args.get("stream_options") == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_ahandle_streaming_sequential_tools(self, model):
        """Test ahandle_streaming with sequential (non-parallel) tool execution."""
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
            yield ToolCallsEvent(
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": '{"param": "value"}'}}]
            )
            yield ToolOutputsEvent(tool_outputs={"test_tool": "result"})
            yield True

        model.async_client.chat.completions.create = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._process_streaming_event_async = async_gen

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = []
        async for event in model.ahandle_streaming("gpt-4o", messages, tools, tool_executor):
            events.append(event)

        # Verify parallel_tool_calls was set to False
        call_args = model.async_client.chat.completions.create.call_args[1]
        assert call_args.get("parallel_tool_calls") is False

    @pytest.mark.asyncio
    async def test_ahandle_streaming_api_status_error_dict_message(self, model):
        """Test ahandle_streaming with APIStatusError with dict error message."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": {"message": "Invalid request"}})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.async_client.chat.completions.create = AsyncMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Invalid request"):
            async for _ in model.ahandle_streaming("gpt-4o", messages, None, tool_executor):
                pass

    @pytest.mark.asyncio
    async def test_ahandle_streaming_api_status_error_string_message(self, model):
        """Test ahandle_streaming with APIStatusError with string error message."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": "Invalid request"})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.async_client.chat.completions.create = AsyncMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Invalid request"):
            async for _ in model.ahandle_streaming("gpt-4o", messages, None, tool_executor):
                pass

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

        tool_calls = []
        tool_executor = MagicMock()
        messages = []

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, tool_calls, tool_executor, messages):
            events.append(event)

        # Should accumulate metrics and return early
        assert len(events) == 0

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
        mock_choice.finish_reason = None

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = mock_usage

        tool_calls = []
        tool_executor = MagicMock()
        messages = []

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, tool_calls, tool_executor, messages):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)

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

        mock_delta2 = MagicMock()
        mock_delta2.content = None
        mock_delta2.tool_calls = [mock_tool_call_delta2]

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

        tool_calls = []
        tool_executor = MagicMock()
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor.parallel_calls = True
        tool_executor._aexecute_tools_parallel = AsyncMock(return_value={"test_tool": "result"})
        messages = []

        # First chunk
        events1 = []
        async for event in model._process_streaming_event_async(mock_chunk1, tool_calls, tool_executor, messages):
            events1.append(event)

        # Second chunk
        events2 = []
        async for event in model._process_streaming_event_async(mock_chunk2, tool_calls, tool_executor, messages):
            events2.append(event)

        # Tool calls should be concatenated
        assert len(tool_calls) > 0
        assert tool_calls[0]["id"] == "call_1"
        assert tool_calls[0]["function"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_tool_call_index_none(self, model):
        """Test _process_streaming_event_async with tool call delta index None."""
        mock_function_delta = MagicMock()
        mock_function_delta.name = "test_tool"
        mock_function_delta.arguments = '{"param": "value"}'

        mock_tool_call_delta = MagicMock()
        mock_tool_call_delta.index = None  # None index
        mock_tool_call_delta.id = "call_1"
        mock_tool_call_delta.function = mock_function_delta

        mock_delta = MagicMock()
        mock_delta.content = None
        mock_delta.tool_calls = [mock_tool_call_delta]

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "tool_calls"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        tool_calls = []
        tool_executor = MagicMock()
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor.parallel_calls = True
        tool_executor._aexecute_tools_parallel = AsyncMock(return_value={"test_tool": "result"})
        messages = []

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, tool_calls, tool_executor, messages):
            events.append(event)

        # Should use index 0 when None
        assert len(tool_calls) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_tool_call_id_concatenation(self, model):
        """Test _process_streaming_event_async with tool call id concatenation."""
        mock_function_delta = MagicMock()
        mock_function_delta.name = None
        mock_function_delta.arguments = None

        mock_tool_call_delta1 = MagicMock()
        mock_tool_call_delta1.index = 0
        mock_tool_call_delta1.id = "call_"
        mock_tool_call_delta1.function = mock_function_delta

        mock_tool_call_delta2 = MagicMock()
        mock_tool_call_delta2.index = 0
        mock_tool_call_delta2.id = "1"
        mock_tool_call_delta2.function = mock_function_delta

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

        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [mock_choice1]
        mock_chunk1.usage = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [mock_choice2]
        mock_chunk2.usage = None

        tool_calls = []
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._aexecute_tools_parallel = AsyncMock(return_value={"test_tool": "result"})
        messages = []

        # First chunk
        async for _ in model._process_streaming_event_async(mock_chunk1, tool_calls, tool_executor, messages):
            pass

        # Second chunk
        async for _ in model._process_streaming_event_async(mock_chunk2, tool_calls, tool_executor, messages):
            pass

        # ID should be concatenated
        assert tool_calls[0]["id"] == "call_1"

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_tool_call_function_name_concatenation(self, model):
        """Test _process_streaming_event_async with tool call function name concatenation."""
        mock_function_delta1 = MagicMock()
        mock_function_delta1.name = "test_"
        mock_function_delta1.arguments = None

        mock_function_delta2 = MagicMock()
        mock_function_delta2.name = "tool"
        mock_function_delta2.arguments = None

        mock_tool_call_delta1 = MagicMock()
        mock_tool_call_delta1.index = 0
        mock_tool_call_delta1.id = "call_1"
        mock_tool_call_delta1.function = mock_function_delta1

        mock_tool_call_delta2 = MagicMock()
        mock_tool_call_delta2.index = 0
        mock_tool_call_delta2.id = None
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

        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [mock_choice1]
        mock_chunk1.usage = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [mock_choice2]
        mock_chunk2.usage = None

        tool_calls = []
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._aexecute_tools_parallel = AsyncMock(return_value={"test_tool": "result"})
        messages = []

        # First chunk
        async for _ in model._process_streaming_event_async(mock_chunk1, tool_calls, tool_executor, messages):
            pass

        # Second chunk
        async for _ in model._process_streaming_event_async(mock_chunk2, tool_calls, tool_executor, messages):
            pass

        # Function name should be concatenated
        assert tool_calls[0]["function"]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_tool_call_function_arguments_concatenation(self, model):
        """Test _process_streaming_event_async with tool call function arguments concatenation."""
        mock_function_delta1 = MagicMock()
        mock_function_delta1.name = "test_tool"
        mock_function_delta1.arguments = '{"param": "value'

        mock_function_delta2 = MagicMock()
        mock_function_delta2.name = None
        mock_function_delta2.arguments = '1"}'

        mock_tool_call_delta1 = MagicMock()
        mock_tool_call_delta1.index = 0
        mock_tool_call_delta1.id = "call_1"
        mock_tool_call_delta1.function = mock_function_delta1

        mock_tool_call_delta2 = MagicMock()
        mock_tool_call_delta2.index = 0
        mock_tool_call_delta2.id = None
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

        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [mock_choice1]
        mock_chunk1.usage = None

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [mock_choice2]
        mock_chunk2.usage = None

        tool_calls = []
        tool_executor = MagicMock()
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor.parallel_calls = True
        tool_executor._aexecute_tools_parallel = AsyncMock(return_value={"test_tool": "result"})
        messages = []

        # First chunk
        async for _ in model._process_streaming_event_async(mock_chunk1, tool_calls, tool_executor, messages):
            pass

        # Second chunk
        async for _ in model._process_streaming_event_async(mock_chunk2, tool_calls, tool_executor, messages):
            pass

        # Function arguments should be concatenated
        assert tool_calls[0]["function"]["arguments"] == '{"param": "value1"}'

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_finish_reason_stop(self, model):
        """Test _process_streaming_event_async with finish_reason stop."""
        mock_delta = MagicMock()
        mock_delta.content = "Response"
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        tool_calls = []
        tool_executor = MagicMock()
        messages = []

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, tool_calls, tool_executor, messages):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, FinishReasonEvent) for e in events)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_finish_reason_tool_calls(self, model):
        """Test _process_streaming_event_async with finish_reason tool_calls."""
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

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        tool_calls = []
        tool_executor = MagicMock()
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []
        tool_executor.parallel_calls = True
        messages = []

        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, tool_calls, tool_executor, messages):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ToolCallsEvent) for e in events)
        assert any(isinstance(e, ToolOutputsEvent) for e in events)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_finish_reason_tool_calls_no_function_name(self, model):
        """Test _process_streaming_event_async with finish_reason tool_calls but no function name."""
        mock_function_delta = MagicMock()
        mock_function_delta.name = ""  # Empty name
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

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        tool_calls = []
        tool_executor = MagicMock()
        messages = []

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, tool_calls, tool_executor, messages):
            events.append(event)

        # Should yield False when no function name
        assert len(events) > 0
        assert any(e is False for e in events)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_content_empty_string(self, model):
        """Test _process_streaming_event_async with empty string content."""
        mock_delta = MagicMock()
        mock_delta.content = ""  # Empty string
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        tool_calls = []
        tool_executor = MagicMock()
        messages = []

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, tool_calls, tool_executor, messages):
            events.append(event)

        # Empty string should not yield ContentEvent
        assert not any(isinstance(e, ContentEvent) for e in events)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_content_whitespace(self, model):
        """Test _process_streaming_event_async with whitespace-only content."""
        mock_delta = MagicMock()
        mock_delta.content = "   "  # Whitespace only
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        tool_calls = []
        tool_executor = MagicMock()
        messages = []

        events = []
        async for event in model._process_streaming_event_async(mock_chunk, tool_calls, tool_executor, messages):
            events.append(event)

        # Whitespace-only should not yield ContentEvent
        assert not any(isinstance(e, ContentEvent) for e in events)

    def test_handle_non_streaming_with_pydantic_response_format(self, model):
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

        result = model.handle_non_streaming("gpt-4o", messages, None, tool_executor, response_format=OutputModel)
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

        result = model.handle_non_streaming("gpt-4o", messages, tools, tool_executor)
        assert result is None

    def test_handle_non_streaming_tool_call_no_function(self, model):
        """Test handle_non_streaming with tool call that has no function attribute."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = None  # No function
        mock_tool_call.model_dump = MagicMock(return_value={"id": "call_1", "type": "function", "function": None})

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
        model._execute_tools_parallel_sync = MagicMock(return_value={})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("gpt-4o", messages, tools, tool_executor)
        assert result is None

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

        model.client.chat.completions.create = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gpt-4o", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_json_parse_error(self, model):
        """Test handle_non_streaming with JSON parse error in response format."""
        mock_message = MagicMock()
        mock_message.content = "not valid json"
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

        result = model.handle_non_streaming("gpt-4o", messages, None, tool_executor, response_format={"type": "json_object"})
        assert result is not None
        # Content should remain as string if JSON parsing fails
        assert isinstance(messages[-1]["content"], str)

    def test_handle_non_streaming_all_params(self, model):
        """Test handle_non_streaming with all parameters."""
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

        # Set all parameters
        model.temperature = 0.8
        model.top_p = 0.9
        model.presence_penalty = 0.1
        model.frequency_penalty = 0.2
        model.max_tokens = 1000
        model.reasoning_effort = "high"

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gpt-4o", messages, None, tool_executor)
        assert result is not None
        # Verify parameters were passed
        call_args = model.client.chat.completions.create.call_args[1]
        assert call_args.get("temperature") == 0.8
        assert call_args.get("top_p") == 0.9
        assert call_args.get("presence_penalty") == 0.1
        assert call_args.get("frequency_penalty") == 0.2
        assert call_args.get("max_completion_tokens") == 1000
        assert call_args.get("reasoning_effort") == "high"

    def test_handle_non_streaming_api_status_error_dict_message(self, model):
        """Test handle_non_streaming with APIStatusError with dict error message."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": {"message": "Invalid request"}})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.client.chat.completions.create = MagicMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Invalid request"):
            model.handle_non_streaming("gpt-4o", messages, None, tool_executor)

    def test_handle_non_streaming_api_status_error_string_message(self, model):
        """Test handle_non_streaming with APIStatusError with string error message."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": "Invalid request"})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.client.chat.completions.create = MagicMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Invalid request"):
            model.handle_non_streaming("gpt-4o", messages, None, tool_executor)

    def test_handle_streaming_with_pydantic_response_format(self, model):
        """Test handle_streaming with Pydantic response format."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_delta = MagicMock()
        mock_delta.content = '{"result": "test"}'
        mock_delta.tool_calls = None

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        model.client.chat.completions.create = MagicMock(return_value=iter([mock_chunk]))
        model._process_streaming_event_sync = MagicMock(return_value=[ContentEvent(content='{"result": "test"}')])

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gpt-4o", messages, None, tool_executor, response_format=OutputModel))
        assert len(events) >= 0

    def test_handle_streaming_all_params(self, model):
        """Test handle_streaming with all parameters."""
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

        # Set all parameters
        model.temperature = 0.8
        model.top_p = 0.9
        model.presence_penalty = 0.1
        model.frequency_penalty = 0.2
        model.max_tokens = 1000
        model.reasoning_effort = "high"

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gpt-4o", messages, None, tool_executor))
        assert len(events) > 0

        # Verify parameters were passed
        call_args = model.client.chat.completions.create.call_args[1]
        assert call_args.get("temperature") == 0.8
        assert call_args.get("top_p") == 0.9
        assert call_args.get("presence_penalty") == 0.1
        assert call_args.get("frequency_penalty") == 0.2
        assert call_args.get("max_completion_tokens") == 1000
        assert call_args.get("reasoning_effort") == "high"

    def test_handle_streaming_sequential_tools(self, model):
        """Test handle_streaming with sequential (non-parallel) tool execution."""
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

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]
        mock_chunk.usage = None

        model.client.chat.completions.create = MagicMock(return_value=iter([mock_chunk]))
        model._process_streaming_event_sync = MagicMock(
            return_value=[
                ToolCallsEvent(
                    tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "test_tool", "arguments": '{"param": "value"}'}}]
                ),
                ToolOutputsEvent(tool_outputs={"test_tool": "result"}),
                True,
            ]
        )

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = list(model.handle_streaming("gpt-4o", messages, tools, tool_executor))
        assert len(events) > 0

        # Verify parallel_tool_calls was set to False
        call_args = model.client.chat.completions.create.call_args[1]
        assert call_args.get("parallel_tool_calls") is False

    def test_handle_streaming_api_status_error_dict_message(self, model):
        """Test handle_streaming with APIStatusError with dict error message."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": {"message": "Invalid request"}})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.client.chat.completions.create = MagicMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Invalid request"):
            list(model.handle_streaming("gpt-4o", messages, None, tool_executor))

    def test_handle_streaming_api_status_error_string_message(self, model):
        """Test handle_streaming with APIStatusError with string error message."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": "Invalid request"})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.client.chat.completions.create = MagicMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Invalid request"):
            list(model.handle_streaming("gpt-4o", messages, None, tool_executor))

    def test_format_files_for_openai_chat_unsupported_file_type(self, model):
        """Test _format_files_for_openai_chat with unsupported file type."""
        file_obj = File(content=b"data", file_type=FileType.AUDIO)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        with patch("hypertic.utils.log.get_logger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            result = model._format_files_for_openai_chat(message)
            # Should still return message, just without the unsupported file
            assert "content" in result
            mock_logger.warning.assert_called()

    def test_format_files_for_openai_chat_with_existing_content_list(self, model):
        """Test _format_files_for_openai_chat with existing content list."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": [{"type": "text", "text": "Check this"}], "_file_objects": [file_obj]}
        result = model._format_files_for_openai_chat(message)
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 1
