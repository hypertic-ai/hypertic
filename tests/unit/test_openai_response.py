from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.models.events import ContentEvent, ResponseCompletedEvent, ResponseCreatedEvent
from hypertic.models.openai.openairesponse import OpenAIResponse
from hypertic.utils.files import File, FileType


class TestOpenAIResponse:
    @pytest.fixture
    def model(self, mock_api_key):
        with patch("hypertic.models.openai.openairesponse.AsyncOpenAIClient"), patch("hypertic.models.openai.openairesponse.OpenAIClient"):
            return OpenAIResponse(api_key=mock_api_key, model="gpt-4o")

    def test_openai_response_creation(self, mock_api_key):
        """Test OpenAIResponse initialization."""
        with patch("hypertic.models.openai.openairesponse.AsyncOpenAIClient"), patch("hypertic.models.openai.openairesponse.OpenAIClient"):
            model = OpenAIResponse(api_key=mock_api_key, model="gpt-4o")
            assert model.api_key == mock_api_key
            assert model.model == "gpt-4o"

    def test_openai_response_creation_no_api_key(self):
        """Test OpenAIResponse initialization without API key."""
        with (
            patch("hypertic.models.openai.openairesponse.getenv", return_value=None),
            patch("hypertic.models.openai.openairesponse.AsyncOpenAIClient"),
            patch("hypertic.models.openai.openairesponse.OpenAIClient"),
        ):
            model = OpenAIResponse(model="gpt-4o")
            assert model.api_key is None

    def test_openai_response_with_params(self, mock_api_key):
        """Test OpenAIResponse with all parameters."""
        with patch("hypertic.models.openai.openairesponse.AsyncOpenAIClient"), patch("hypertic.models.openai.openairesponse.OpenAIClient"):
            model = OpenAIResponse(
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

    def test_clean_output_item_function_call(self, model):
        """Test _clean_output_item for function_call type."""
        item = {
            "type": "function_call",
            "id": "call_123",
            "name": "test_tool",
            "arguments": '{"param": "value"}',
            "invalid_field": "should be removed",
        }
        result = model._clean_output_item(item, "function_call")
        assert "id" in result
        assert "name" in result
        assert "invalid_field" not in result

    def test_clean_output_item_reasoning(self, model):
        """Test _clean_output_item for reasoning type."""
        item = {
            "type": "reasoning",
            "content": "reasoning text",
            "status": "should be removed",
            "model_fields": "should be removed",
        }
        result = model._clean_output_item(item, "reasoning")
        assert "content" in result
        assert "status" not in result
        assert "model_fields" not in result

    def test_clean_output_item_other(self, model):
        """Test _clean_output_item for other types."""
        item = {
            "type": "message",
            "content": "text",
            "status": "should be removed",
        }
        result = model._clean_output_item(item, "other")
        assert "content" in result
        assert "status" not in result

    def test_format_files_for_openai_response_no_files(self, model):
        """Test _format_files_for_openai_response with no files."""
        message = {"role": "user", "content": "Hello"}
        result = model._format_files_for_openai_response(message)
        assert result == message

    def test_format_files_for_openai_response_with_image_url(self, model):
        """Test _format_files_for_openai_response with image URL."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        result = model._format_files_for_openai_response(message)
        assert "content" in result
        assert isinstance(result["content"], list)
        assert any(item.get("type") == "input_image" for item in result["content"])

    def test_format_files_for_openai_response_with_image_base64(self, model):
        """Test _format_files_for_openai_response with image base64."""
        file_obj = File(content=b"fake image", file_type=FileType.IMAGE, mime_type="image/jpeg")
        # Mock to_base64 to return a value
        with patch.object(File, "to_base64", return_value="base64data"):
            message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
            result = model._format_files_for_openai_response(message)
            assert "content" in result
            assert isinstance(result["content"], list)
            # Check that base64 data is in the content
            content_str = str(result["content"])
            assert "base64" in content_str or "base64data" in content_str

    def test_format_files_for_openai_response_with_document(self, model):
        """Test _format_files_for_openai_response with document."""
        file_obj = File(url="https://example.com/doc.pdf", file_type=FileType.DOCUMENT)
        message = {"role": "user", "content": "Read this", "_file_objects": [file_obj]}
        result = model._format_files_for_openai_response(message)
        assert any(item.get("type") == "input_file" for item in result["content"])

    def test_format_files_for_openai_response_removes_file_objects(self, model):
        """Test that _format_files_for_openai_response removes _file_objects."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "_file_objects": [file_obj], "files": ["url"]}
        result = model._format_files_for_openai_response(message)
        assert "_file_objects" not in result
        assert "files" not in result

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_basic(self, model):
        """Test ahandle_non_streaming with basic message."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item = MagicMock()
        mock_output_item.type = "message"
        mock_content_part = MagicMock()
        mock_content_part.type = "output_text"
        mock_content_part.text = "Response text"
        mock_output_item.content = [mock_content_part]
        mock_response.output = [mock_output_item]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}
        mock_response.output_parsed = None

        model.async_client.responses.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_tools(self, model):
        """Test ahandle_non_streaming with tools."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item = MagicMock()
        mock_output_item.type = "function_call"
        mock_output_item.id = "call_1"
        mock_output_item.name = "test_tool"
        mock_output_item.arguments = '{"param": "value"}'
        mock_output_item.call_id = "call_1"
        mock_output_item.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_1", "name": "test_tool"})
        mock_response.output = [mock_output_item]
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}

        model.async_client.responses.create = AsyncMock(return_value=mock_response)
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
    async def test_ahandle_non_streaming_with_system_message(self, model):
        """Test ahandle_non_streaming with system message."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item = MagicMock()
        mock_output_item.type = "message"
        mock_content_part = MagicMock()
        mock_content_part.type = "output_text"
        mock_content_part.text = "Response"
        mock_output_item.content = [mock_content_part]
        mock_response.output = [mock_output_item]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}
        mock_response.output_parsed = None

        model.async_client.responses.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_api_error(self, model):
        """Test ahandle_non_streaming with API error."""
        mock_response = MagicMock()
        mock_error = MagicMock()
        mock_error.message = "API Error"
        mock_response.error = mock_error

        model.async_client.responses.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="OpenAI API error"):
            await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_no_output(self, model):
        """Test ahandle_non_streaming with no output."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_response.output = []

        model.async_client.responses.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="No output"):
            await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)

    def test_handle_non_streaming_basic(self, model):
        """Test handle_non_streaming with basic message."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item = MagicMock()
        mock_output_item.type = "message"
        mock_content_part = MagicMock()
        mock_content_part.type = "output_text"
        mock_content_part.text = "Response text"
        mock_output_item.content = [mock_content_part]
        mock_response.output = [mock_output_item]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}
        mock_response.output_parsed = None

        model.client.responses.create = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gpt-4o", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_with_tools(self, model):
        """Test handle_non_streaming with tools."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item = MagicMock()
        mock_output_item.type = "function_call"
        mock_output_item.id = "call_1"
        mock_output_item.name = "test_tool"
        mock_output_item.arguments = '{"param": "value"}'
        mock_output_item.call_id = "call_1"
        mock_output_item.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_1", "name": "test_tool"})
        mock_response.output = [mock_output_item]
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}

        model.client.responses.create = MagicMock(return_value=mock_response)
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
        mock_event1 = MagicMock()
        mock_event1.type = "response.created"
        mock_response = MagicMock()
        mock_response.id = "resp_123"
        mock_event1.response = mock_response
        mock_event2 = MagicMock()
        mock_event2.type = "response.output_text.delta"
        mock_event2.delta = "Response"
        mock_event3 = MagicMock()
        mock_event3.type = "response.completed"

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

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=AsyncIterator([mock_event1, mock_event2, mock_event3]))
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        model.async_client.responses.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gpt-4o", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_rate_limit_error(self, model):
        """Test ahandle_streaming with rate limit error."""
        from openai import RateLimitError

        # Create a proper RateLimitError instance
        mock_response = MagicMock()
        mock_response.status_code = 429
        rate_limit_error = RateLimitError(response=mock_response, body={}, message="Rate limited")

        model.async_client.responses.stream = MagicMock(side_effect=rate_limit_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Rate limit"):
            async for _ in model.ahandle_streaming("gpt-4o", messages, None, tool_executor):
                pass

    @pytest.mark.asyncio
    async def test_ahandle_streaming_connection_error(self, model):
        """Test ahandle_streaming with connection error."""
        from openai import APIConnectionError

        # Create a proper APIConnectionError instance
        connection_error = APIConnectionError(request=MagicMock(), message="Connection failed")

        model.async_client.responses.stream = MagicMock(side_effect=connection_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Connection error"):
            async for _ in model.ahandle_streaming("gpt-4o", messages, None, tool_executor):
                pass

    def test_handle_streaming_basic(self, model):
        """Test handle_streaming with basic message."""
        mock_event1 = MagicMock()
        mock_event1.type = "response.created"
        mock_event2 = MagicMock()
        mock_event2.type = "response.output_item.added"
        mock_event2.output_item = MagicMock()
        mock_event2.output_item.type = "message"
        mock_event2.output_item.content = "Response"
        mock_event3 = MagicMock()
        mock_event3.type = "response.done"

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=None)
        mock_stream.__iter__ = MagicMock(return_value=iter([mock_event1, mock_event2, mock_event3]))

        model.client.responses.stream = MagicMock(return_value=mock_stream)
        model._process_streaming_event_sync = MagicMock(return_value=[])

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gpt-4o", messages, None, tool_executor))
        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_created(self, model):
        """Test _process_streaming_event_async with response.created event."""
        mock_event = MagicMock()
        mock_event.type = "response.created"
        mock_event.response = MagicMock()
        mock_event.response.id = "resp_123"

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        events = await model._process_streaming_event_async(mock_event, tool_executor, messages, tool_use)
        assert len(events) > 0
        assert any(isinstance(e, ResponseCreatedEvent) for e in events)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_content(self, model):
        """Test _process_streaming_event_async with content event."""
        mock_event = MagicMock()
        mock_event.type = "response.output_text.delta"
        mock_event.delta = "Test content"

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        events = await model._process_streaming_event_async(mock_event, tool_executor, messages, tool_use)
        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_tool_calls(self, model):
        """Test _process_streaming_event_async with tool calls."""
        mock_event = MagicMock()
        mock_event.type = "response.output_item.done"
        mock_item = MagicMock()
        mock_item.type = "function_call"
        mock_item.id = "call_1"
        mock_item.name = "test_tool"
        mock_item.arguments = '{"param": "value"}'
        mock_item.call_id = "call_1"
        mock_item.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_1"})
        mock_event.item = mock_item

        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        messages = []
        tool_use = {}

        events = await model._process_streaming_event_async(mock_event, tool_executor, messages, tool_use)
        assert len(events) > 0

    def test_process_streaming_event_sync_created(self, model):
        """Test _process_streaming_event_sync with response.created event."""
        mock_event = MagicMock()
        mock_event.type = "response.created"
        mock_event.response = MagicMock()
        mock_event.response.id = "resp_123"

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        events = model._process_streaming_event_sync(mock_event, tool_executor, messages, tool_use)
        assert len(events) > 0

    def test_process_streaming_event_sync_content(self, model):
        """Test _process_streaming_event_sync with content event."""
        mock_event = MagicMock()
        mock_event.type = "response.output_text.delta"
        mock_event.delta = "Test content"

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        events = model._process_streaming_event_sync(mock_event, tool_executor, messages, tool_use)
        assert len(events) > 0

    def test_process_streaming_event_sync_tool_calls(self, model):
        """Test _process_streaming_event_sync with tool calls."""
        mock_event = MagicMock()
        mock_event.type = "response.output_item.done"
        mock_item = MagicMock()
        mock_item.type = "function_call"
        mock_item.id = "call_1"
        mock_item.name = "test_tool"
        mock_item.arguments = '{"param": "value"}'
        mock_item.call_id = "call_1"
        mock_item.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_1"})
        mock_event.item = mock_item

        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        messages = []
        tool_use = {}

        events = model._process_streaming_event_sync(mock_event, tool_executor, messages, tool_use)
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_tools(self, model):
        """Test ahandle_non_streaming with sequential (non-parallel) tool execution."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item1 = MagicMock()
        mock_output_item1.type = "function_call"
        mock_output_item1.id = "call_1"
        mock_output_item1.name = "test_tool1"
        mock_output_item1.arguments = '{"param": "value1"}'
        mock_output_item1.call_id = "call_1"
        mock_output_item1.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_1", "name": "test_tool1", "call_id": "call_1"})
        mock_output_item2 = MagicMock()
        mock_output_item2.type = "function_call"
        mock_output_item2.id = "call_2"
        mock_output_item2.name = "test_tool2"
        mock_output_item2.arguments = '{"param": "value2"}'
        mock_output_item2.call_id = "call_2"
        mock_output_item2.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_2", "name": "test_tool2", "call_id": "call_2"})
        mock_response.output = [mock_output_item1, mock_output_item2]
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}

        model.async_client.responses.create = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool1": "result1"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool1", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential execution
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("gpt-4o", messages, tools, tool_executor)
        assert result is None  # Returns None when tool calls are present
        model._execute_tools_parallel_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_reasoning(self, model):
        """Test ahandle_non_streaming with reasoning output."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item = MagicMock()
        mock_output_item.type = "reasoning"
        mock_output_item.model_dump = MagicMock(return_value={"type": "reasoning", "content": "thinking..."})
        mock_message_item = MagicMock()
        mock_message_item.type = "message"
        mock_content_part = MagicMock()
        mock_content_part.type = "output_text"
        mock_content_part.text = "Response"
        mock_message_item.content = [mock_content_part]
        mock_response.output = [mock_output_item, mock_message_item]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}
        mock_response.output_parsed = None

        model.async_client.responses.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_structured_output(self, model):
        """Test ahandle_non_streaming with response_format."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item = MagicMock()
        mock_output_item.type = "message"
        mock_content_part = MagicMock()
        mock_content_part.type = "output_text"
        mock_content_part.text = "Response"
        mock_output_item.content = [mock_content_part]
        mock_response.output = [mock_output_item]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}
        mock_response.output_parsed = {"result": "test"}

        model.async_client.responses.parse = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor, response_format=OutputModel)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_content_list(self, model):
        """Test ahandle_non_streaming with list content."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item = MagicMock()
        mock_output_item.type = "message"
        mock_output_item.content = [{"type": "output_text", "text": "Response"}]
        mock_response.output = [mock_output_item]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}
        mock_response.output_parsed = None

        model.async_client.responses.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_instructions_from_executor(self, model):
        """Test ahandle_non_streaming with instructions from tool_executor."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item = MagicMock()
        mock_output_item.type = "message"
        mock_content_part = MagicMock()
        mock_content_part.type = "output_text"
        mock_content_part.text = "Response"
        mock_output_item.content = [mock_content_part]
        mock_response.output = [mock_output_item]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}
        mock_response.output_parsed = None

        model.async_client.responses.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()
        tool_executor.instructions = "You are helpful"

        result = await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_incomplete_details(self, model):
        """Test ahandle_non_streaming with incomplete_details."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_incomplete = MagicMock()
        mock_incomplete.reason = "max_tokens"
        mock_response.incomplete_details = mock_incomplete

        model.async_client.responses.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="incomplete"):
            await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_rate_limit_error(self, model):
        """Test ahandle_non_streaming with rate limit error."""
        from openai import RateLimitError

        mock_response = MagicMock()
        mock_response.status_code = 429
        rate_limit_error = RateLimitError(response=mock_response, body={}, message="Rate limited")

        model.async_client.responses.create = AsyncMock(side_effect=rate_limit_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Rate limit"):
            await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_connection_error(self, model):
        """Test ahandle_non_streaming with connection error."""
        from openai import APIConnectionError

        connection_error = APIConnectionError(request=MagicMock(), message="Connection failed")

        model.async_client.responses.create = AsyncMock(side_effect=connection_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Connection error"):
            await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_status_error(self, model):
        """Test ahandle_non_streaming with status error."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": {"message": "API Error"}})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.async_client.responses.create = AsyncMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Status error"):
            await model.ahandle_non_streaming("gpt-4o", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_reasoning_with_tool_call(self, model):
        """Test _process_streaming_event_async with reasoning before tool call."""
        mock_event = MagicMock()
        mock_event.type = "response.completed"
        mock_response = MagicMock()

        # Create a proper usage object that can be checked for truthiness
        class UsageObj:
            def __init__(self):
                self.input_tokens = 10
                self.output_tokens = 20

        mock_response.usage = UsageObj()
        mock_reasoning_item = MagicMock()
        mock_reasoning_item.type = "reasoning"
        mock_reasoning_item.id = "reason_1"
        mock_reasoning_item.model_dump = MagicMock(return_value={"type": "reasoning", "id": "reason_1"})
        mock_tool_item = MagicMock()
        mock_tool_item.type = "function_call"
        mock_tool_item.id = "call_1"
        mock_tool_item.name = "test_tool"
        mock_tool_item.arguments = '{"param": "value"}'
        mock_tool_item.call_id = "call_1"
        mock_tool_item.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_1"})
        mock_response.output = [mock_reasoning_item, mock_tool_item]
        mock_event.response = mock_response

        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._pending_function_calls = [
            {"function_call": {"id": "call_1", "function": {"name": "test_tool"}}, "item": mock_tool_item, "item_call_id": "call_1"}
        ]
        tool_executor._tool_calls = []
        tool_executor._tool_outputs = {}
        messages = []

        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        events = await model._process_streaming_event_async(mock_event, tool_executor, messages, {})
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_sequential_tools(self, model):
        """Test _process_streaming_event_async with sequential tool execution."""
        mock_event = MagicMock()
        mock_event.type = "response.completed"
        mock_response = MagicMock()

        # Create a proper usage object that can be checked for truthiness
        class UsageObj:
            def __init__(self):
                self.input_tokens = 10
                self.output_tokens = 20

        mock_response.usage = UsageObj()
        mock_tool_item1 = MagicMock()
        mock_tool_item1.type = "function_call"
        mock_tool_item1.id = "call_1"
        mock_tool_item1.name = "test_tool1"
        mock_tool_item1.arguments = '{"param": "value1"}'
        mock_tool_item1.call_id = "call_1"
        mock_tool_item1.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_1"})
        mock_tool_item2 = MagicMock()
        mock_tool_item2.type = "function_call"
        mock_tool_item2.id = "call_2"
        mock_tool_item2.name = "test_tool2"
        mock_tool_item2.arguments = '{"param": "value2"}'
        mock_tool_item2.call_id = "call_2"
        mock_tool_item2.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_2"})
        mock_response.output = [mock_tool_item1, mock_tool_item2]
        mock_event.response = mock_response

        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._pending_function_calls = [
            {"function_call": {"id": "call_1", "function": {"name": "test_tool1"}}, "item": mock_tool_item1, "item_call_id": "call_1"},
            {"function_call": {"id": "call_2", "function": {"name": "test_tool2"}}, "item": mock_tool_item2, "item_call_id": "call_2"},
        ]
        tool_executor._tool_calls = []
        tool_executor._tool_outputs = {}
        messages = []

        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool1": "result1"})

        events = await model._process_streaming_event_async(mock_event, tool_executor, messages, {})
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_output_item_added(self, model):
        """Test _process_streaming_event_async with output_item.added event."""
        mock_event = MagicMock()
        mock_event.type = "response.output_item.added"
        mock_item = MagicMock()
        mock_item.type = "message"
        mock_item.id = "msg_1"
        mock_event.item = mock_item

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        events = await model._process_streaming_event_async(mock_event, tool_executor, messages, tool_use)
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_content_part_added(self, model):
        """Test _process_streaming_event_async with content_part.added event."""
        mock_event = MagicMock()
        mock_event.type = "response.content_part.added"

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        events = await model._process_streaming_event_async(mock_event, tool_executor, messages, tool_use)
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_output_text_done(self, model):
        """Test _process_streaming_event_async with output_text.done event."""
        mock_event = MagicMock()
        mock_event.type = "response.output_text.done"

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        events = await model._process_streaming_event_async(mock_event, tool_executor, messages, tool_use)
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_in_progress(self, model):
        """Test _process_streaming_event_async with in_progress event."""
        mock_event = MagicMock()
        mock_event.type = "response.in_progress"

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        events = await model._process_streaming_event_async(mock_event, tool_executor, messages, tool_use)
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_process_streaming_event_async_completed_no_tool_calls(self, model):
        """Test _process_streaming_event_async with completed event and no tool calls."""
        mock_event = MagicMock()
        mock_event.type = "response.completed"
        mock_response = MagicMock()

        # Create a proper usage object that can be checked for truthiness
        class UsageObj:
            def __init__(self):
                self.input_tokens = 10
                self.output_tokens = 20

        mock_response.usage = UsageObj()
        mock_response.output = []
        mock_event.response = mock_response

        tool_executor = MagicMock()
        tool_executor._streaming_content_buffer = "content"
        messages = []
        tool_use = {}

        events = await model._process_streaming_event_async(mock_event, tool_executor, messages, tool_use)
        assert len(events) > 0
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)

    def test_handle_non_streaming_sequential_tools(self, model):
        """Test handle_non_streaming with sequential (non-parallel) tool execution."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item1 = MagicMock()
        mock_output_item1.type = "function_call"
        mock_output_item1.id = "call_1"
        mock_output_item1.name = "test_tool1"
        mock_output_item1.arguments = '{"param": "value1"}'
        mock_output_item1.call_id = "call_1"
        mock_output_item1.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_1", "name": "test_tool1", "call_id": "call_1"})
        mock_output_item2 = MagicMock()
        mock_output_item2.type = "function_call"
        mock_output_item2.id = "call_2"
        mock_output_item2.name = "test_tool2"
        mock_output_item2.arguments = '{"param": "value2"}'
        mock_output_item2.call_id = "call_2"
        mock_output_item2.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_2", "name": "test_tool2", "call_id": "call_2"})
        mock_response.output = [mock_output_item1, mock_output_item2]
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}

        model.client.responses.create = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool1": "result1"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool1", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential execution
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("gpt-4o", messages, tools, tool_executor)
        assert result is None
        model._execute_tools_parallel_sync.assert_called_once()

    def test_handle_non_streaming_with_reasoning(self, model):
        """Test handle_non_streaming with reasoning output."""
        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item = MagicMock()
        mock_output_item.type = "reasoning"
        mock_output_item.model_dump = MagicMock(return_value={"type": "reasoning", "content": "thinking..."})
        mock_message_item = MagicMock()
        mock_message_item.type = "message"
        mock_content_part = MagicMock()
        mock_content_part.type = "output_text"
        mock_content_part.text = "Response"
        mock_message_item.content = [mock_content_part]
        mock_response.output = [mock_output_item, mock_message_item]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}
        mock_response.output_parsed = None

        model.client.responses.create = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gpt-4o", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_with_structured_output(self, model):
        """Test handle_non_streaming with response_format."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_response = MagicMock()
        mock_response.error = None
        mock_response.incomplete_details = None
        mock_output_item = MagicMock()
        mock_output_item.type = "message"
        mock_content_part = MagicMock()
        mock_content_part.type = "output_text"
        mock_content_part.text = "Response"
        mock_output_item.content = [mock_content_part]
        mock_response.output = [mock_output_item]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {"input_tokens": 10, "output_tokens": 20}
        mock_response.output_parsed = {"result": "test"}

        model.client.responses.parse = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gpt-4o", messages, None, tool_executor, response_format=OutputModel)
        assert result is not None

    def test_handle_non_streaming_rate_limit_error(self, model):
        """Test handle_non_streaming with rate limit error."""
        from openai import RateLimitError

        mock_response = MagicMock()
        mock_response.status_code = 429
        rate_limit_error = RateLimitError(response=mock_response, body={}, message="Rate limited")

        model.client.responses.create = MagicMock(side_effect=rate_limit_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Rate limit"):
            model.handle_non_streaming("gpt-4o", messages, None, tool_executor)

    def test_handle_non_streaming_connection_error(self, model):
        """Test handle_non_streaming with connection error."""
        from openai import APIConnectionError

        connection_error = APIConnectionError(request=MagicMock(), message="Connection failed")

        model.client.responses.create = MagicMock(side_effect=connection_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Connection error"):
            model.handle_non_streaming("gpt-4o", messages, None, tool_executor)

    def test_handle_non_streaming_status_error(self, model):
        """Test handle_non_streaming with status error."""
        from openai import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": {"message": "API Error"}})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.client.responses.create = MagicMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Status error"):
            model.handle_non_streaming("gpt-4o", messages, None, tool_executor)

    def test_process_streaming_event_sync_reasoning_with_tool_call(self, model):
        """Test _process_streaming_event_sync with reasoning before tool call."""
        mock_event = MagicMock()
        mock_event.type = "response.completed"
        mock_response = MagicMock()

        # Create a proper usage object that can be checked for truthiness
        class UsageObj:
            def __init__(self):
                self.input_tokens = 10
                self.output_tokens = 20

        mock_response.usage = UsageObj()
        mock_reasoning_item = MagicMock()
        mock_reasoning_item.type = "reasoning"
        mock_reasoning_item.id = "reason_1"
        mock_reasoning_item.model_dump = MagicMock(return_value={"type": "reasoning", "id": "reason_1"})
        mock_tool_item = MagicMock()
        mock_tool_item.type = "function_call"
        mock_tool_item.id = "call_1"
        mock_tool_item.name = "test_tool"
        mock_tool_item.arguments = '{"param": "value"}'
        mock_tool_item.call_id = "call_1"
        mock_tool_item.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_1"})
        mock_response.output = [mock_reasoning_item, mock_tool_item]
        mock_event.response = mock_response

        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._pending_function_calls = [
            {"function_call": {"id": "call_1", "function": {"name": "test_tool"}}, "item": mock_tool_item, "item_call_id": "call_1"}
        ]
        tool_executor._tool_calls = []
        tool_executor._tool_outputs = {}
        messages = []

        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        events = model._process_streaming_event_sync(mock_event, tool_executor, messages, {})
        assert len(events) > 0

    def test_process_streaming_event_sync_sequential_tools(self, model):
        """Test _process_streaming_event_sync with sequential tool execution."""
        mock_event = MagicMock()
        mock_event.type = "response.completed"
        mock_response = MagicMock()

        # Create a proper usage object that can be checked for truthiness
        class UsageObj:
            def __init__(self):
                self.input_tokens = 10
                self.output_tokens = 20

        mock_response.usage = UsageObj()
        mock_tool_item1 = MagicMock()
        mock_tool_item1.type = "function_call"
        mock_tool_item1.id = "call_1"
        mock_tool_item1.name = "test_tool1"
        mock_tool_item1.arguments = '{"param": "value1"}'
        mock_tool_item1.call_id = "call_1"
        mock_tool_item1.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_1"})
        mock_tool_item2 = MagicMock()
        mock_tool_item2.type = "function_call"
        mock_tool_item2.id = "call_2"
        mock_tool_item2.name = "test_tool2"
        mock_tool_item2.arguments = '{"param": "value2"}'
        mock_tool_item2.call_id = "call_2"
        mock_tool_item2.model_dump = MagicMock(return_value={"type": "function_call", "id": "call_2"})
        mock_response.output = [mock_tool_item1, mock_tool_item2]
        mock_event.response = mock_response

        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._pending_function_calls = [
            {"function_call": {"id": "call_1", "function": {"name": "test_tool1"}}, "item": mock_tool_item1, "item_call_id": "call_1"},
            {"function_call": {"id": "call_2", "function": {"name": "test_tool2"}}, "item": mock_tool_item2, "item_call_id": "call_2"},
        ]
        tool_executor._tool_calls = []
        tool_executor._tool_outputs = {}
        messages = []

        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool1": "result1"})

        events = model._process_streaming_event_sync(mock_event, tool_executor, messages, {})
        assert len(events) > 0

    def test_process_streaming_event_sync_output_item_added(self, model):
        """Test _process_streaming_event_sync with output_item.added event."""
        mock_event = MagicMock()
        mock_event.type = "response.output_item.added"
        mock_item = MagicMock()
        mock_item.type = "message"
        mock_item.id = "msg_1"
        mock_event.item = mock_item

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        events = model._process_streaming_event_sync(mock_event, tool_executor, messages, tool_use)
        assert isinstance(events, list)

    def test_process_streaming_event_sync_content_part_added(self, model):
        """Test _process_streaming_event_sync with content_part.added event."""
        mock_event = MagicMock()
        mock_event.type = "response.content_part.added"

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        events = model._process_streaming_event_sync(mock_event, tool_executor, messages, tool_use)
        assert isinstance(events, list)

    def test_process_streaming_event_sync_output_text_done(self, model):
        """Test _process_streaming_event_sync with output_text.done event."""
        mock_event = MagicMock()
        mock_event.type = "response.output_text.done"

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        events = model._process_streaming_event_sync(mock_event, tool_executor, messages, tool_use)
        assert isinstance(events, list)

    def test_process_streaming_event_sync_in_progress(self, model):
        """Test _process_streaming_event_sync with in_progress event."""
        mock_event = MagicMock()
        mock_event.type = "response.in_progress"

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        events = model._process_streaming_event_sync(mock_event, tool_executor, messages, tool_use)
        assert isinstance(events, list)

    def test_process_streaming_event_sync_completed_no_tool_calls(self, model):
        """Test _process_streaming_event_sync with completed event and no tool calls."""
        mock_event = MagicMock()
        mock_event.type = "response.completed"
        mock_response = MagicMock()

        # Create a proper usage object that can be checked for truthiness
        class UsageObj:
            def __init__(self):
                self.input_tokens = 10
                self.output_tokens = 20

        mock_response.usage = UsageObj()
        mock_response.output = []
        mock_event.response = mock_response

        tool_executor = MagicMock()
        tool_executor._streaming_content_buffer = "content"
        messages = []
        tool_use = {}

        events = model._process_streaming_event_sync(mock_event, tool_executor, messages, tool_use)
        assert len(events) > 0
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)

    def test_process_streaming_event_sync_error(self, model):
        """Test _process_streaming_event_sync with error event."""
        mock_event = MagicMock()
        mock_event.type = "error"
        mock_event.error = {"message": "Test error"}

        tool_executor = MagicMock()
        messages = []
        tool_use = {}

        with pytest.raises(Exception, match="Error from OpenAI"):
            model._process_streaming_event_sync(mock_event, tool_executor, messages, tool_use)

    def test_format_files_for_openai_response_with_document_base64(self, model):
        """Test _format_files_for_openai_response with document base64."""
        file_obj = File(content=b"fake pdf", file_type=FileType.DOCUMENT, mime_type="application/pdf", filename="test.pdf")
        with patch.object(File, "to_base64", return_value="base64data"):
            message = {"role": "user", "content": "Read this", "_file_objects": [file_obj]}
            result = model._format_files_for_openai_response(message)
            assert "content" in result
            assert any("input_file" in str(item) for item in result["content"])

    def test_format_files_for_openai_response_unsupported_file_type(self, model):
        """Test _format_files_for_openai_response with unsupported file type."""
        file_obj = File(content=b"data", file_type=FileType.AUDIO)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        result = model._format_files_for_openai_response(message)
        # Should still return message, just without the unsupported file
        assert "content" in result

    def test_format_files_for_openai_response_with_existing_content_list(self, model):
        """Test _format_files_for_openai_response with existing content list."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": [{"type": "text", "text": "Check this"}], "_file_objects": [file_obj]}
        result = model._format_files_for_openai_response(message)
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 1

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_response_format(self, model):
        """Test ahandle_streaming with response_format."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        async def async_iter_empty():
            return
            yield

        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=async_iter_empty())
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        model.async_client.responses.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gpt-4o", messages, None, tool_executor, response_format=OutputModel):
            events.append(event)
        # Should complete without error

    def test_handle_streaming_with_response_format(self, model):
        """Test handle_streaming with response_format."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=None)
        mock_stream.__iter__ = MagicMock(return_value=iter([]))

        model.client.responses.stream = MagicMock(return_value=mock_stream)
        model._process_streaming_event_sync = MagicMock(return_value=[])

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gpt-4o", messages, None, tool_executor, response_format=OutputModel))
        assert isinstance(events, list)
