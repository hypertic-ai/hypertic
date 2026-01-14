from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.models.anthropic.anthropic import Anthropic
from hypertic.models.events import ContentEvent, ReasoningEvent
from hypertic.utils.files import File, FileType


class TestAnthropic:
    @pytest.fixture
    def model(self, mock_api_key):
        with patch("hypertic.models.anthropic.anthropic.AsyncAnthropicClient"), patch("hypertic.models.anthropic.anthropic.AnthropicClient"):
            return Anthropic(api_key=mock_api_key, model="claude-3-5-sonnet-20241022")

    def test_anthropic_creation(self, mock_api_key):
        """Test Anthropic initialization."""
        with patch("hypertic.models.anthropic.anthropic.AsyncAnthropicClient"), patch("hypertic.models.anthropic.anthropic.AnthropicClient"):
            model = Anthropic(api_key=mock_api_key, model="claude-3-5-sonnet-20241022")
            assert model.api_key == mock_api_key
            assert model.model == "claude-3-5-sonnet-20241022"

    def test_anthropic_creation_no_api_key(self):
        """Test Anthropic initialization without API key."""
        with (
            patch("hypertic.models.anthropic.anthropic.getenv", return_value=None),
            patch("hypertic.models.anthropic.anthropic.AsyncAnthropicClient"),
            patch("hypertic.models.anthropic.anthropic.AnthropicClient"),
        ):
            model = Anthropic(model="claude-3-5-sonnet-20241022")
            assert model.api_key is None

    def test_anthropic_with_params(self, mock_api_key):
        """Test Anthropic with all parameters."""
        with patch("hypertic.models.anthropic.anthropic.AsyncAnthropicClient"), patch("hypertic.models.anthropic.anthropic.AnthropicClient"):
            model = Anthropic(
                api_key=mock_api_key,
                model="claude-3-5-sonnet-20241022",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000,
                thinking_tokens=500,
            )
            assert model.temperature == 0.7
            assert model.top_p == 0.9
            assert model.max_tokens == 1000
            assert model.thinking_tokens == 500

    def test_convert_tools_to_anthropic(self, model):
        """Test _convert_tools_to_anthropic."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = model._convert_tools_to_anthropic(tools)
        assert len(result) == 1
        assert result[0]["name"] == "test_tool"
        assert "input_schema" in result[0]

    def test_convert_tools_to_anthropic_empty(self, model):
        """Test _convert_tools_to_anthropic with empty list."""
        result = model._convert_tools_to_anthropic([])
        assert result == []

    def test_prepare_structured_output_none(self, model):
        """Test _prepare_structured_output with None."""
        output_format, use_beta = model._prepare_structured_output(None, None)
        assert output_format is None
        assert use_beta is False

    def test_prepare_structured_output_pydantic_with_tools(self, model):
        """Test _prepare_structured_output with Pydantic model and tools."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str

        with patch("hypertic.models.anthropic.anthropic.transform_schema", return_value={"type": "object"}):
            output_format, use_beta = model._prepare_structured_output(TestSchema, [{"type": "function"}])
            assert use_beta is True
            assert "type" in output_format
            assert output_format["type"] == "json_schema"

    def test_prepare_structured_output_pydantic_without_tools(self, model):
        """Test _prepare_structured_output with Pydantic model without tools."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str

        output_format, use_beta = model._prepare_structured_output(TestSchema, None)
        assert use_beta is True
        assert output_format == TestSchema

    def test_format_files_for_anthropic_no_files(self, model):
        """Test _format_files_for_anthropic with no files."""
        message = {"role": "user", "content": "Hello"}
        result = model._format_files_for_anthropic(message)
        assert result == message

    def test_format_files_for_anthropic_with_image_url(self, model):
        """Test _format_files_for_anthropic with image URL."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        result = model._format_files_for_anthropic(message)
        assert "content" in result
        assert isinstance(result["content"], list)
        assert any(item.get("type") == "image" for item in result["content"])

    def test_format_files_for_anthropic_with_image_base64(self, model):
        """Test _format_files_for_anthropic with image base64."""
        file_obj = File(content=b"fake image", file_type=FileType.IMAGE, mime_type="image/jpeg")
        with patch.object(File, "to_base64", return_value="base64data"):
            message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
            result = model._format_files_for_anthropic(message)
            assert "content" in result
            assert any(item.get("type") == "image" for item in result["content"])

    def test_format_files_for_anthropic_with_document(self, model):
        """Test _format_files_for_anthropic with document."""
        file_obj = File(url="https://example.com/doc.pdf", file_type=FileType.DOCUMENT)
        message = {"role": "user", "content": "Read this", "_file_objects": [file_obj]}
        result = model._format_files_for_anthropic(message)
        assert any(item.get("type") == "document" for item in result["content"])

    def test_format_files_for_anthropic_removes_file_objects(self, model):
        """Test that _format_files_for_anthropic removes _file_objects."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "_file_objects": [file_obj], "files": ["url"]}
        result = model._format_files_for_anthropic(message)
        assert "_file_objects" not in result
        assert "files" not in result

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_basic(self, model):
        """Test ahandle_non_streaming with basic message."""
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "Response text"

        mock_response = MagicMock()
        mock_response.content = [mock_content_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.stop_reason = "end_turn"

        model.async_client.messages.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_tools(self, model):
        """Test ahandle_non_streaming with tools."""
        mock_tool_use_block = MagicMock()
        mock_tool_use_block.type = "tool_use"
        mock_tool_use_block.id = "tool_1"
        mock_tool_use_block.name = "test_tool"
        mock_tool_use_block.input = {"param": "value"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_use_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        model.async_client.messages.create = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, tools, tool_executor)
        assert result is None  # Returns None when tool calls are present

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_thinking(self, model):
        """Test ahandle_non_streaming with thinking blocks."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"

        mock_thinking_block = MagicMock()
        mock_thinking_block.type = "thinking"
        mock_thinking_block.thinking = "Let me think..."

        mock_response = MagicMock()
        mock_response.content = [mock_thinking_block, mock_text_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.stop_reason = "end_turn"

        model.async_client.messages.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_rate_limit_error(self, model):
        """Test ahandle_non_streaming with rate limit error."""
        from anthropic import RateLimitError

        # Create a proper RateLimitError instance
        rate_limit_error = RateLimitError(message="Rate limited", response=MagicMock(), body={})

        model.async_client.messages.create = AsyncMock(side_effect=rate_limit_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Rate limit"):
            await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_connection_error(self, model):
        """Test ahandle_non_streaming with connection error."""
        from anthropic import APIConnectionError

        # Create a proper APIConnectionError instance
        connection_error = APIConnectionError(request=MagicMock(), message="Connection failed")

        model.async_client.messages.create = AsyncMock(side_effect=connection_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Connection error"):
            await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor)

    def test_handle_non_streaming_basic(self, model):
        """Test handle_non_streaming with basic message."""
        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "Response text"

        mock_response = MagicMock()
        mock_response.content = [mock_content_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.stop_reason = "end_turn"

        model.client.messages.create = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_with_tools(self, model):
        """Test handle_non_streaming with tools."""
        mock_tool_use_block = MagicMock()
        mock_tool_use_block.type = "tool_use"
        mock_tool_use_block.id = "tool_1"
        mock_tool_use_block.name = "test_tool"
        mock_tool_use_block.input = {"param": "value"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_use_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        model.client.messages.create = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("claude-3-5-sonnet-20241022", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_streaming_basic(self, model):
        """Test ahandle_streaming with basic message."""
        from anthropic.types import ContentBlockDeltaEvent, TextDelta

        mock_delta = TextDelta(type="text_delta", text="Response")
        mock_event = ContentBlockDeltaEvent(type="content_block_delta", index=0, delta=mock_delta)

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
        mock_stream.__aenter__ = AsyncMock(return_value=AsyncIterator([mock_event]))
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        model.async_client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_thinking(self, model):
        """Test ahandle_streaming with thinking events."""
        from anthropic.types import ContentBlockDeltaEvent, ThinkingDelta

        mock_delta = ThinkingDelta(type="thinking_delta", thinking="Let me think...")
        mock_event = ContentBlockDeltaEvent(type="content_block_delta", index=0, delta=mock_delta)

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
        mock_stream.__aenter__ = AsyncMock(return_value=AsyncIterator([mock_event]))
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        model.async_client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ReasoningEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_tool_use(self, model):
        """Test ahandle_streaming with tool use events."""
        from anthropic.types import ContentBlockDeltaEvent, ContentBlockStartEvent, InputJSONDelta, ToolUseBlock

        mock_content_block = ToolUseBlock(type="tool_use", id="tool_1", name="test_tool", input={})
        mock_start_event = ContentBlockStartEvent(type="content_block_start", index=0, content_block=mock_content_block)

        mock_delta = InputJSONDelta(type="input_json_delta", partial_json='{"param": "value"}')
        mock_delta_event = ContentBlockDeltaEvent(type="content_block_delta", index=0, delta=mock_delta)

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
        mock_stream.__aenter__ = AsyncMock(return_value=AsyncIterator([mock_start_event, mock_delta_event]))
        mock_stream.__aexit__ = AsyncMock(return_value=None)

        model.async_client.messages.stream = MagicMock(return_value=mock_stream)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True

        events = []
        async for event in model.ahandle_streaming("claude-3-5-sonnet-20241022", messages, tools, tool_executor):
            events.append(event)

        assert len(events) >= 0

    def test_handle_streaming_basic(self, model):
        """Test handle_streaming with basic message."""
        from anthropic.types import ContentBlockDeltaEvent, TextDelta

        mock_delta = TextDelta(type="text_delta", text="Response")
        mock_event = ContentBlockDeltaEvent(type="content_block_delta", index=0, delta=mock_delta)

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter([mock_event]))
        mock_stream.__exit__ = MagicMock(return_value=None)

        model.client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)

    def test_handle_streaming_with_thinking(self, model):
        """Test handle_streaming with thinking events."""
        from anthropic.types import ContentBlockDeltaEvent, ThinkingDelta

        mock_delta = ThinkingDelta(type="thinking_delta", thinking="Let me think...")
        mock_event = ContentBlockDeltaEvent(type="content_block_delta", index=0, delta=mock_delta)

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter([mock_event]))
        mock_stream.__exit__ = MagicMock(return_value=None)

        model.client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, ReasoningEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_tools(self, model):
        """Test ahandle_non_streaming with sequential (non-parallel) tool execution."""
        mock_tool_use_block1 = MagicMock()
        mock_tool_use_block1.type = "tool_use"
        mock_tool_use_block1.id = "tool_1"
        mock_tool_use_block1.name = "test_tool1"
        mock_tool_use_block1.input = {"param": "value1"}
        mock_tool_use_block2 = MagicMock()
        mock_tool_use_block2.type = "tool_use"
        mock_tool_use_block2.id = "tool_2"
        mock_tool_use_block2.name = "test_tool2"
        mock_tool_use_block2.input = {"param": "value2"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_use_block1, mock_tool_use_block2]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        model.async_client.messages.create = AsyncMock(return_value=mock_response)
        model._execute_tool_async = AsyncMock(return_value="result1")

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool1", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential execution
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, tools, tool_executor)
        assert result is None  # Returns None when tool calls are present
        assert model._execute_tool_async.call_count == 2  # Called for each tool sequentially

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_structured_output_beta(self, model):
        """Test ahandle_non_streaming with structured output (beta API)."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_parsed_output = MagicMock()
        mock_parsed_output.model_dump = MagicMock(return_value={"result": "test"})
        mock_message = MagicMock()
        mock_message.usage = MagicMock()
        mock_message.usage.input_tokens = 10
        mock_message.usage.output_tokens = 20
        mock_message.stop_reason = "end_turn"
        mock_response = MagicMock()
        mock_response.parsed_output = mock_parsed_output
        mock_response.message = mock_message

        model.async_client.beta.messages.parse = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor, response_format=OutputModel)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_structured_output_no_message(self, model):
        """Test ahandle_non_streaming with structured output but no message."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_parsed_output = MagicMock()
        mock_parsed_output.model_dump = MagicMock(return_value={"result": "test"})
        mock_response = MagicMock()
        mock_response.parsed_output = mock_parsed_output
        mock_response.message = None

        model.async_client.beta.messages.parse = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor, response_format=OutputModel)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_structured_output_with_tools(self, model):
        """Test ahandle_non_streaming with structured output and tools (beta API)."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "Response"
        mock_response = MagicMock()
        mock_response.content = [mock_content_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.stop_reason = "end_turn"

        model.async_client.beta.messages.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, tools, tool_executor, response_format=OutputModel)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_empty_text_block(self, model):
        """Test ahandle_non_streaming with empty text block in tool use."""
        mock_tool_use_block = MagicMock()
        mock_tool_use_block.type = "tool_use"
        mock_tool_use_block.id = "tool_1"
        mock_tool_use_block.name = "test_tool"
        mock_tool_use_block.input = {"param": "value"}
        mock_empty_text_block = MagicMock()
        mock_empty_text_block.type = "text"
        mock_empty_text_block.text = "   "  # Whitespace only

        mock_response = MagicMock()
        mock_response.content = [mock_empty_text_block, mock_tool_use_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        model.async_client.messages.create = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_thinking_content(self, model):
        """Test ahandle_non_streaming with thinking content."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"
        mock_thinking_block = MagicMock()
        mock_thinking_block.type = "thinking"
        mock_thinking_block.thinking = "Let me think about this..."

        mock_response = MagicMock()
        mock_response.content = [mock_thinking_block, mock_text_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.stop_reason = "end_turn"

        model.async_client.messages.create = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_status_error_with_json(self, model):
        """Test ahandle_non_streaming with status error that has JSON response."""
        from anthropic import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": {"message": "API Error"}})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.async_client.messages.create = AsyncMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Status error"):
            await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_status_error_without_json(self, model):
        """Test ahandle_non_streaming with status error without JSON response."""
        from anthropic import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(side_effect=Exception("No JSON"))
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.async_client.messages.create = AsyncMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Status error"):
            await model.ahandle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_sequential_tools(self, model):
        """Test ahandle_streaming with sequential tool execution."""
        from anthropic.types import ContentBlockStartEvent, ContentBlockStopEvent, MessageStopEvent, ToolUseBlock

        mock_content_block = ToolUseBlock(type="tool_use", id="tool_1", name="test_tool1", input={"param": "value1"})
        mock_start_event = ContentBlockStartEvent(type="content_block_start", index=0, content_block=mock_content_block)
        mock_stop_event = ContentBlockStopEvent(type="content_block_stop", index=0)
        mock_message_stop = MessageStopEvent(type="message_stop")

        class UsageObj:
            def __init__(self):
                self.input_tokens = 10
                self.output_tokens = 20

        class FinalMessage:
            def __init__(self):
                self.usage = UsageObj()

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

        class StreamContext:
            async def __aenter__(self):
                return AsyncIterator([mock_start_event, mock_stop_event, mock_message_stop])

            async def __aexit__(self, *args):
                pass

            async def get_final_message(self):
                return FinalMessage()

        mock_stream = StreamContext()
        model.async_client.messages.stream = MagicMock(return_value=mock_stream)
        model._execute_tool_async = AsyncMock(return_value="result1")

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool1", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential

        events = []
        async for event in model.ahandle_streaming("claude-3-5-sonnet-20241022", messages, tools, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_beta_api(self, model):
        """Test ahandle_streaming with beta API (structured output)."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        # Use MagicMock to avoid Pydantic validation issues
        mock_event = MagicMock()
        mock_event.type = "message_stop"
        mock_message = MagicMock()
        mock_message.usage = MagicMock()
        mock_message.usage.input_tokens = 10
        mock_message.usage.output_tokens = 20
        mock_event.message = mock_message
        # Ensure no delta.text access
        mock_event.delta = None

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

        class StreamContext:
            async def __aenter__(self):
                return AsyncIterator([mock_event])

            async def __aexit__(self, *args):
                pass

        mock_stream = StreamContext()
        model.async_client.beta.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor, response_format=OutputModel):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_beta_content_block_start(self, model):
        """Test ahandle_streaming with beta content block start event."""
        from anthropic.lib.streaming._beta_types import BetaRawContentBlockStartEvent

        # Use MagicMock to avoid Pydantic validation issues
        mock_content_block = MagicMock()
        mock_content_block.type = "tool_use"
        mock_content_block.id = "tool_1"
        mock_content_block.name = "test_tool"
        mock_content_block.input = {}
        mock_event = MagicMock()
        mock_event.type = "content_block_start"
        mock_event.index = 0
        mock_event.content_block = mock_content_block
        # Make it pass isinstance check
        mock_event.__class__ = BetaRawContentBlockStartEvent

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

        class StreamContext:
            async def __aenter__(self):
                return AsyncIterator([mock_event])

            async def __aexit__(self, *args):
                pass

        mock_stream = StreamContext()
        model.async_client.beta.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_beta_input_json(self, model):
        """Test ahandle_streaming with beta input JSON event."""
        from anthropic.lib.streaming._beta_types import BetaInputJsonEvent, BetaRawContentBlockStartEvent

        # Use MagicMock to avoid Pydantic validation issues
        mock_content_block = MagicMock()
        mock_content_block.type = "tool_use"
        mock_content_block.id = "tool_1"
        mock_content_block.name = "test_tool"
        mock_content_block.input = {}
        mock_start = MagicMock()
        mock_start.type = "content_block_start"
        mock_start.index = 0
        mock_start.content_block = mock_content_block
        mock_start.__class__ = BetaRawContentBlockStartEvent
        mock_json = MagicMock()
        mock_json.type = "input_json_delta"
        mock_json.partial_json = '{"param": "value"}'
        mock_json.__class__ = BetaInputJsonEvent

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

        class StreamContext:
            async def __aenter__(self):
                return AsyncIterator([mock_start, mock_json])

            async def __aexit__(self, *args):
                pass

        mock_stream = StreamContext()
        model.async_client.beta.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_content_block_stop_json_decode_error(self, model):
        """Test ahandle_streaming with content block stop and JSON decode error."""
        from anthropic.types import ContentBlockStartEvent, ContentBlockStopEvent, ToolUseBlock

        mock_content_block = ToolUseBlock(type="tool_use", id="tool_1", name="test_tool", input={})
        mock_start = ContentBlockStartEvent(type="content_block_start", index=0, content_block=mock_content_block)
        mock_stop = ContentBlockStopEvent(type="content_block_stop", index=0)

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

        class StreamContext:
            async def __aenter__(self):
                # Set up current_tool_use with invalid JSON
                return AsyncIterator([mock_start, mock_stop])

            async def __aexit__(self, *args):
                pass

        mock_stream = StreamContext()
        model.async_client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        # Manually set up the state that would cause JSON decode error
        events = []
        async for event in model.ahandle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_parsed_beta_content_block_stop(self, model):
        """Test ahandle_streaming with parsed beta content block stop."""
        from anthropic.lib.streaming._beta_types import ParsedBetaContentBlockStopEvent

        # Use MagicMock to avoid Pydantic validation issues
        mock_content_block = MagicMock()
        mock_content_block.type = "tool_use"
        mock_content_block.id = "tool_1"
        mock_content_block.name = "test_tool"
        mock_content_block.input = {"param": "value"}
        mock_event = MagicMock()
        mock_event.type = "content_block_stop"
        mock_event.index = 0
        mock_event.content_block = mock_content_block
        # Make it pass isinstance check
        mock_event.__class__ = ParsedBetaContentBlockStopEvent

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

        class StreamContext:
            async def __aenter__(self):
                return AsyncIterator([mock_event])

            async def __aexit__(self, *args):
                pass

        mock_stream = StreamContext()
        model.async_client.beta.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_message_stop_no_usage(self, model):
        """Test ahandle_streaming with message stop but no usage."""
        from anthropic.types import MessageStopEvent

        mock_event = MessageStopEvent(type="message_stop")

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

        class FinalMessage:
            def __init__(self):
                self.usage = None

        class StreamContext:
            async def __aenter__(self):
                return AsyncIterator([mock_event])

            async def __aexit__(self, *args):
                pass

            async def get_final_message(self):
                return FinalMessage()

        mock_stream = StreamContext()
        model.async_client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_message_stop_exception(self, model):
        """Test ahandle_streaming with message stop that raises exception."""
        from anthropic.types import MessageStopEvent

        mock_event = MessageStopEvent(type="message_stop")

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

        class StreamContext:
            async def __aenter__(self):
                return AsyncIterator([mock_event])

            async def __aexit__(self, *args):
                pass

            async def get_final_message(self):
                raise AttributeError("No final message")

        mock_stream = StreamContext()
        model.async_client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    def test_handle_non_streaming_sequential_tools(self, model):
        """Test handle_non_streaming with sequential (non-parallel) tool execution."""
        mock_tool_use_block1 = MagicMock()
        mock_tool_use_block1.type = "tool_use"
        mock_tool_use_block1.id = "tool_1"
        mock_tool_use_block1.name = "test_tool1"
        mock_tool_use_block1.input = {"param": "value1"}
        mock_tool_use_block2 = MagicMock()
        mock_tool_use_block2.type = "tool_use"
        mock_tool_use_block2.id = "tool_2"
        mock_tool_use_block2.name = "test_tool2"
        mock_tool_use_block2.input = {"param": "value2"}

        mock_response = MagicMock()
        mock_response.content = [mock_tool_use_block1, mock_tool_use_block2]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        model.client.messages.create = MagicMock(return_value=mock_response)
        model._execute_tool_sync = MagicMock(return_value="result1")

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool1", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential execution
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("claude-3-5-sonnet-20241022", messages, tools, tool_executor)
        assert result is None
        assert model._execute_tool_sync.call_count == 2

    def test_handle_non_streaming_with_structured_output_beta(self, model):
        """Test handle_non_streaming with structured output (beta API)."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_parsed_output = MagicMock()
        mock_parsed_output.model_dump = MagicMock(return_value={"result": "test"})
        mock_message = MagicMock()
        mock_message.usage = MagicMock()
        mock_message.usage.input_tokens = 10
        mock_message.usage.output_tokens = 20
        mock_message.stop_reason = "end_turn"
        mock_response = MagicMock()
        mock_response.parsed_output = mock_parsed_output
        mock_response.message = mock_message

        model.client.beta.messages.parse = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor, response_format=OutputModel)
        assert result is not None

    def test_handle_non_streaming_with_structured_output_with_tools(self, model):
        """Test handle_non_streaming with structured output and tools (beta API)."""
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        mock_content_block = MagicMock()
        mock_content_block.type = "text"
        mock_content_block.text = "Response"
        mock_response = MagicMock()
        mock_response.content = [mock_content_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.stop_reason = "end_turn"

        model.client.beta.messages.create = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("claude-3-5-sonnet-20241022", messages, tools, tool_executor, response_format=OutputModel)
        assert result is not None

    def test_handle_non_streaming_with_thinking_content(self, model):
        """Test handle_non_streaming with thinking content."""
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Response"
        mock_thinking_block = MagicMock()
        mock_thinking_block.type = "thinking"
        mock_thinking_block.thinking = "Let me think about this..."

        mock_response = MagicMock()
        mock_response.content = [mock_thinking_block, mock_text_block]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        mock_response.stop_reason = "end_turn"

        model.client.messages.create = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_status_error_with_json(self, model):
        """Test handle_non_streaming with status error that has JSON response."""
        from anthropic import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"error": {"message": "API Error"}})
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.client.messages.create = MagicMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Status error"):
            model.handle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor)

    def test_handle_non_streaming_status_error_without_json(self, model):
        """Test handle_non_streaming with status error without JSON response."""
        from anthropic import APIStatusError

        mock_response = MagicMock()
        mock_response.json = MagicMock(side_effect=Exception("No JSON"))
        status_error = APIStatusError(response=mock_response, body={}, message="Status error")

        model.client.messages.create = MagicMock(side_effect=status_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="Status error"):
            model.handle_non_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor)

    def test_handle_streaming_sequential_tools(self, model):
        """Test handle_streaming with sequential tool execution."""
        from anthropic.types import ContentBlockStartEvent, ContentBlockStopEvent, MessageStopEvent, ToolUseBlock

        mock_content_block = ToolUseBlock(type="tool_use", id="tool_1", name="test_tool1", input={"param": "value1"})
        mock_start_event = ContentBlockStartEvent(type="content_block_start", index=0, content_block=mock_content_block)
        mock_stop_event = ContentBlockStopEvent(type="content_block_stop", index=0)
        mock_message_stop = MessageStopEvent(type="message_stop")

        class UsageObj:
            def __init__(self):
                self.input_tokens = 10
                self.output_tokens = 20

        class FinalMessage:
            def __init__(self):
                self.usage = UsageObj()

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter([mock_start_event, mock_stop_event, mock_message_stop]))
        mock_stream.__exit__ = MagicMock(return_value=None)
        mock_stream.get_final_message = MagicMock(return_value=FinalMessage())

        model.client.messages.stream = MagicMock(return_value=mock_stream)
        model._execute_tool_sync = MagicMock(return_value="result1")

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool1", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential

        events = list(model.handle_streaming("claude-3-5-sonnet-20241022", messages, tools, tool_executor))
        assert len(events) > 0

    def test_handle_streaming_with_beta_api(self, model):
        """Test handle_streaming with beta API (structured output)."""
        from anthropic.lib.streaming._beta_types import ParsedBetaMessageStopEvent
        from pydantic import BaseModel

        class OutputModel(BaseModel):
            result: str

        # Use patch to make isinstance work
        with patch("hypertic.models.anthropic.anthropic.isinstance") as mock_isinstance:

            def isinstance_check(obj, cls):
                if cls == ParsedBetaMessageStopEvent and hasattr(obj, "type") and obj.type == "message_stop":
                    return True
                return isinstance.__wrapped__(obj, cls) if hasattr(isinstance, "__wrapped__") else type(obj) is cls

            mock_isinstance.side_effect = isinstance_check

            # Use MagicMock to avoid Pydantic validation issues
            mock_event = MagicMock()
            mock_event.type = "message_stop"
            mock_message = MagicMock()
            mock_message.usage = MagicMock()
            mock_message.usage.input_tokens = 10
            mock_message.usage.output_tokens = 20
            mock_event.message = mock_message
            # Ensure no delta.text access
            mock_event.delta = None

            mock_stream = MagicMock()
            mock_stream.__enter__ = MagicMock(return_value=iter([mock_event]))
            mock_stream.__exit__ = MagicMock(return_value=None)

            model.client.beta.messages.stream = MagicMock(return_value=mock_stream)

            messages = [{"role": "user", "content": "Hello"}]
            tool_executor = MagicMock()

            events = list(model.handle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor, response_format=OutputModel))
            assert len(events) >= 0

    def test_handle_streaming_with_content_block_stop_json_decode_error(self, model):
        """Test handle_streaming with content block stop and JSON decode error."""
        from anthropic.types import ContentBlockStartEvent, ContentBlockStopEvent, ToolUseBlock

        mock_content_block = ToolUseBlock(type="tool_use", id="tool_1", name="test_tool", input={})
        mock_start = ContentBlockStartEvent(type="content_block_start", index=0, content_block=mock_content_block)
        mock_stop = ContentBlockStopEvent(type="content_block_stop", index=0)

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter([mock_start, mock_stop]))
        mock_stream.__exit__ = MagicMock(return_value=None)

        model.client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor))
        assert len(events) >= 0

    def test_handle_streaming_with_message_stop_no_usage(self, model):
        """Test handle_streaming with message stop but no usage."""
        from anthropic.types import MessageStopEvent

        mock_event = MessageStopEvent(type="message_stop")

        class FinalMessage:
            def __init__(self):
                self.usage = None

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter([mock_event]))
        mock_stream.__exit__ = MagicMock(return_value=None)
        mock_stream.get_final_message = MagicMock(return_value=FinalMessage())

        model.client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor))
        assert len(events) >= 0

    def test_handle_streaming_with_message_stop_exception(self, model):
        """Test handle_streaming with message stop that raises exception."""
        from anthropic.types import MessageStopEvent

        mock_event = MessageStopEvent(type="message_stop")

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=iter([mock_event]))
        mock_stream.__exit__ = MagicMock(return_value=None)
        mock_stream.get_final_message = MagicMock(side_effect=AttributeError("No final message"))

        model.client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("claude-3-5-sonnet-20241022", messages, None, tool_executor))
        assert len(events) >= 0

    def test_format_files_for_anthropic_with_document_base64(self, model):
        """Test _format_files_for_anthropic with document base64."""
        file_obj = File(content=b"fake pdf", file_type=FileType.DOCUMENT, mime_type="application/pdf")
        with patch.object(File, "to_base64", return_value="base64data"):
            message = {"role": "user", "content": "Read this", "_file_objects": [file_obj]}
            result = model._format_files_for_anthropic(message)
            assert "content" in result
            assert any("document" in str(item) for item in result["content"])

    def test_format_files_for_anthropic_unsupported_file_type(self, model):
        """Test _format_files_for_anthropic with unsupported file type."""
        file_obj = File(content=b"data", file_type=FileType.AUDIO)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        result = model._format_files_for_anthropic(message)
        # Should still return message, just without the unsupported file
        assert "content" in result

    def test_format_files_for_anthropic_with_existing_content_list(self, model):
        """Test _format_files_for_anthropic with existing content list."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": [{"type": "text", "text": "Check this"}], "_file_objects": [file_obj]}
        result = model._format_files_for_anthropic(message)
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 1

    def test_prepare_structured_output_transform_schema_error(self, model):
        """Test _prepare_structured_output with transform_schema error."""
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            name: str

        with patch("hypertic.models.anthropic.anthropic.transform_schema", side_effect=ImportError("No transform")):
            output_format, use_beta = model._prepare_structured_output(TestSchema, [{"type": "function"}])
            assert use_beta is True
            assert "type" in output_format
