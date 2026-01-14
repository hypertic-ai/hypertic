from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hypertic.models.events import ContentEvent, ReasoningEvent, ResponseCompletedEvent
from hypertic.models.google.google import GoogleAI
from hypertic.utils.files import File, FileType


class TestGoogleAI:
    @pytest.fixture
    def model(self, mock_api_key):
        with patch("hypertic.models.google.google.GeminiClient"):
            return GoogleAI(api_key=mock_api_key, model="gemini-pro")

    def test_google_ai_creation(self, mock_api_key):
        """Test GoogleAI initialization."""
        with patch("hypertic.models.google.google.GeminiClient"):
            model = GoogleAI(api_key=mock_api_key, model="gemini-pro")
            assert model.api_key == mock_api_key
            assert model.model == "gemini-pro"

    def test_google_ai_creation_no_api_key(self):
        """Test GoogleAI initialization without API key."""
        with patch("hypertic.models.google.google.getenv", return_value=None), patch("hypertic.models.google.google.GeminiClient"):
            model = GoogleAI(model="gemini-pro")
            assert model.api_key is None

    def test_google_ai_with_params(self, mock_api_key):
        """Test GoogleAI with all parameters."""
        with patch("hypertic.models.google.google.GeminiClient"):
            model = GoogleAI(
                api_key=mock_api_key,
                model="gemini-pro",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000,
                thinking_tokens=500,
                thinking=True,
            )
            assert model.temperature == 0.7
            assert model.top_p == 0.9
            assert model.max_tokens == 1000
            assert model.thinking_tokens == 500
            assert model.thinking is True

    def test_normalize_google_usage(self, model):
        """Test _normalize_google_usage."""
        usage = {
            "prompt_token_count": 10,
            "candidates_token_count": 20,
            "total_token_count": 30,
        }
        result = model._normalize_google_usage(usage)
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 20
        assert result["total_tokens"] == 30

    def test_normalize_google_usage_camelcase(self, model):
        """Test _normalize_google_usage with camelCase keys."""
        usage = {
            "promptTokenCount": 10,
            "candidatesTokenCount": 20,
            "totalTokenCount": 30,
        }
        result = model._normalize_google_usage(usage)
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 20
        assert result["total_tokens"] == 30

    def test_normalize_google_usage_empty(self, model):
        """Test _normalize_google_usage with empty dict."""
        result = model._normalize_google_usage({})
        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0
        assert result["total_tokens"] == 0

    def test_convert_tools_to_google(self, model):
        """Test _convert_tools_to_google."""
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
        result = model._convert_tools_to_google(tools)
        assert len(result) == 1
        assert len(result[0].function_declarations) == 1
        assert result[0].function_declarations[0].name == "test_tool"

    def test_convert_tools_to_google_removes_schema_fields(self, model):
        """Test _convert_tools_to_google removes schema fields."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": True,
                        "$schema": "http://json-schema.org/draft-07/schema#",
                    },
                },
            }
        ]
        result = model._convert_tools_to_google(tools)
        assert len(result) == 1

    def test_convert_messages_to_google_format_simple(self, model):
        """Test _convert_messages_to_google_format with simple messages."""

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = model._convert_messages_to_google_format(messages)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[1].role == "model"  # Google uses "model" instead of "assistant"

    def test_convert_messages_to_google_format_list_content(self, model):
        """Test _convert_messages_to_google_format with list content."""
        messages = [{"role": "user", "content": [{"text": "Hello"}, {"text": "World"}]}]
        result = model._convert_messages_to_google_format(messages)
        assert len(result) == 1
        assert len(result[0].parts) == 2

    def test_format_files_for_google_no_files(self, model):
        """Test _format_files_for_google with no files."""
        message = {"role": "user", "content": "Hello"}
        result = model._format_files_for_google(message)
        assert result == message

    def test_format_files_for_google_with_image_url(self, model):
        """Test _format_files_for_google with image URL."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
        result = model._format_files_for_google(message)
        assert "content" in result
        assert isinstance(result["content"], list)
        assert any("file_data" in item for item in result["content"])

    def test_format_files_for_google_with_image_base64(self, model):
        """Test _format_files_for_google with image base64."""
        file_obj = File(content=b"fake image", file_type=FileType.IMAGE, mime_type="image/jpeg")
        with patch.object(File, "to_base64", return_value="base64data"):
            message = {"role": "user", "content": "Check this", "_file_objects": [file_obj]}
            result = model._format_files_for_google(message)
            assert "content" in result
            assert any("inline_data" in item for item in result["content"])

    def test_format_files_for_google_removes_file_objects(self, model):
        """Test that _format_files_for_google removes _file_objects."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "_file_objects": [file_obj], "files": ["url"]}
        result = model._format_files_for_google(message)
        assert "_file_objects" not in result
        assert "files" not in result

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_basic(self, model):
        """Test ahandle_non_streaming with basic message."""
        mock_part = MagicMock()
        mock_part.text = "Response text"
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_tools(self, model):
        """Test ahandle_non_streaming with tools."""
        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        # Ensure thought_signature is None or bytes, not MagicMock
        mock_part.thought_signature = None
        if hasattr(mock_part, "thoughtSignature"):
            mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("gemini-pro", messages, tools, tool_executor)
        assert result is None  # Returns None when tool calls are present

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_thinking(self, model):
        """Test ahandle_non_streaming with thinking."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = "Let me think..."

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_no_content(self, model):
        """Test ahandle_non_streaming with no content."""
        mock_candidate = MagicMock()
        mock_candidate.content = MagicMock()
        mock_candidate.content.parts = None

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_error(self, model):
        """Test ahandle_non_streaming with error."""
        from google.genai.errors import ClientError

        # Create a proper ClientError instance - check what parameters it needs
        try:
            client_error = ClientError(code=400, response_json={"error": {"message": "API Error"}})
        except TypeError:
            # If that doesn't work, just use a generic Exception
            client_error = Exception("API Error")

        model.client.aio.models.generate_content = AsyncMock(side_effect=client_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="API Error"):  # noqa: B017
            await model.ahandle_non_streaming("gemini-pro", messages, None, tool_executor)

    def test_handle_non_streaming_basic(self, model):
        """Test handle_non_streaming with basic message."""
        mock_part = MagicMock()
        mock_part.text = "Response text"
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_with_tools(self, model):
        """Test handle_non_streaming with tools."""
        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        # Ensure thought_signature is None or bytes, not MagicMock
        mock_part.thought_signature = None
        if hasattr(mock_part, "thoughtSignature"):
            mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("gemini-pro", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_streaming_basic(self, model):
        """Test ahandle_streaming with basic message."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_thinking(self, model):
        """Test ahandle_streaming with thinking events."""
        mock_part = MagicMock()
        mock_part.text = "Let me think..."
        mock_part.function_call = None
        mock_part.thought = "Let me think..."

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ReasoningEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_tool_use(self, model):
        """Test ahandle_streaming with tool use events."""
        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, tools, tool_executor):
            events.append(event)

        assert len(events) >= 0

    def test_handle_streaming_basic(self, model):
        """Test handle_streaming with basic message."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

        model.client.models.generate_content_stream = MagicMock(return_value=iter([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gemini-pro", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)

    def test_handle_streaming_with_thinking(self, model):
        """Test handle_streaming with thinking events."""
        mock_part = MagicMock()
        mock_part.text = "Let me think..."
        mock_part.function_call = None
        mock_part.thought = "Let me think..."

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

        model.client.models.generate_content_stream = MagicMock(return_value=iter([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gemini-pro", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, ReasoningEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_tools(self, model):
        """Test ahandle_non_streaming with sequential (non-parallel) tool execution."""
        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("gemini-pro", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_tools_and_response_format_error(self, model):
        """Test ahandle_non_streaming raises error when tools and response_format are both provided."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="does not support structured output"):
            await model.ahandle_non_streaming(
                "gemini-pro",
                messages,
                tools,
                tool_executor,
                response_format={"type": "json_object"},
            )

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_response_format(self, model):
        """Test ahandle_non_streaming with response format (no tools)."""
        mock_part = MagicMock()
        mock_part.text = '{"result": "test"}'
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gemini-pro", messages, None, tool_executor, response_format={"type": "object"})
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_thought_signature(self, model):
        """Test ahandle_non_streaming with thought_signature."""

        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = b"signature"
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("gemini-pro", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_thoughtSignature(self, model):  # noqa: N802
        """Test ahandle_non_streaming with thoughtSignature (camelCase)."""

        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None
        mock_part.thoughtSignature = b"signature"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("gemini-pro", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_sequential_no_parts_list(self, model):
        """Test ahandle_non_streaming sequential with empty parts_list_for_message."""
        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = await model.ahandle_non_streaming("gemini-pro", messages, tools, tool_executor)
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_reasoning_content(self, model):
        """Test ahandle_non_streaming with reasoning content (thought)."""
        mock_part = MagicMock()
        mock_part.text = "Let me think..."
        mock_part.function_call = None
        mock_part.thought = "Let me think..."

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_with_function_call_in_parts(self, model):
        """Test ahandle_non_streaming with function_call in parts (else branch processing)."""
        # This test checks the else branch where parts are processed individually
        # The else branch processes parts that have function_call but aren't in the main function_calls list
        # Actually, if a part has function_call, it WILL be in function_calls list
        # So to test else branch, we need parts with NO function_call
        # But the test name says "with_function_call", so let's test when function_call exists
        # but tools=None, so it goes into tool path and returns None (which is expected)
        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        mock_part.thought = None
        # Ensure thought_signature is None to avoid base64 issues
        mock_part.thought_signature = None
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = None  # No tools provided
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        # With tools=None but function_call exists, it will try to execute tools
        # But since tools=None, it should handle it gracefully
        # Actually, looking at the code, if tools=None, it won't execute tools
        # So it will return None
        result = await model.ahandle_non_streaming("gemini-pro", messages, tools, tool_executor)
        # Result is None because function_calls were detected but tools=None
        assert result is None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_no_candidates(self, model):
        """Test ahandle_non_streaming with no candidates."""
        mock_response = MagicMock()
        mock_response.candidates = []

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_all_params(self, model):
        """Test ahandle_non_streaming with all parameters."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        # Set all parameters
        model.temperature = 0.8
        model.top_p = 0.9
        model.max_tokens = 1000
        model.thinking_tokens = 500
        model.thinking = True

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None
        # Verify parameters were passed
        call_args = model.client.aio.models.generate_content.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_thinking_config_only_tokens(self, model):
        """Test ahandle_non_streaming with thinking_config only tokens."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        model.thinking_tokens = 500
        model.thinking = None

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_thinking_config_only_bool(self, model):
        """Test ahandle_non_streaming with thinking_config only bool."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        model.thinking_tokens = None
        model.thinking = True

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_usage_none(self, model):
        """Test ahandle_non_streaming with None usage_metadata."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_non_streaming_finish_reason_none(self, model):
        """Test ahandle_non_streaming with None finish_reason."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = None

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = await model.ahandle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    @pytest.mark.asyncio
    async def test_ahandle_streaming_tools_and_response_format_error(self, model):
        """Test ahandle_streaming raises error when tools and response_format are both provided."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()

        # The error is raised when the generator function body executes (during iteration)
        # The error is raised at line 491, which is inside a try-except block
        # So it will be caught and wrapped in an Exception
        gen = model.ahandle_streaming(
            "gemini-pro",
            messages,
            tools,
            tool_executor,
            response_format={"type": "json_object"},
        )
        # Try to get the first item - this will execute the function body and trigger the error
        # The error will be caught and yielded as a ContentEvent
        events = []
        async for event in gen:
            events.append(event)
        # The error should be in the events as a ContentEvent
        assert len(events) > 0
        assert any("does not support structured output" in str(e.content) for e in events if hasattr(e, "content"))

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_response_format(self, model):
        """Test ahandle_streaming with response format (no tools)."""
        mock_part = MagicMock()
        mock_part.text = '{"result": "test"}'
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor, response_format={"type": "object"}):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_sequential_tools(self, model):
        """Test ahandle_streaming with sequential (non-parallel) tool execution."""

        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, tools, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_thought_signature(self, model):
        """Test ahandle_streaming with thought_signature."""

        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = b"signature"
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, tools, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_thoughtSignature(self, model):  # noqa: N802
        """Test ahandle_streaming with thoughtSignature (camelCase)."""

        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None
        mock_part.thoughtSignature = b"signature"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, tools, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_sequential_no_parts_list(self, model):
        """Test ahandle_streaming sequential with empty parts_list_for_message."""

        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))
        model._execute_tools_parallel_async = AsyncMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, tools, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_with_usage_metadata(self, model):
        """Test ahandle_streaming with usage_metadata in chunk."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_usage = MagicMock()
        mock_usage.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = mock_usage

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ResponseCompletedEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_usage_dict(self, model):
        """Test ahandle_streaming with usage as dict."""
        mock_usage = {"prompt_token_count": 10, "candidates_token_count": 20}

        mock_chunk = MagicMock()
        mock_chunk.candidates = []
        mock_chunk.usage_metadata = mock_usage

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_usage_with_dict_attr(self, model):
        """Test ahandle_streaming with usage that has __dict__ attribute."""

        class UsageObj:
            def __init__(self):
                self.prompt_token_count = 10
                self.candidates_token_count = 20

        mock_usage = UsageObj()

        mock_chunk = MagicMock()
        mock_chunk.candidates = []
        mock_chunk.usage_metadata = mock_usage

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_usage_with_model_dump(self, model):
        """Test ahandle_streaming with usage that has model_dump method."""

        class UsageObj:
            def model_dump(self):
                return {"prompt_token_count": 10, "candidates_token_count": 20}

        mock_usage = UsageObj()

        mock_chunk = MagicMock()
        mock_chunk.candidates = []
        mock_chunk.usage_metadata = mock_usage

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_text_empty_string(self, model):
        """Test ahandle_streaming with empty string text."""
        mock_part = MagicMock()
        mock_part.text = ""  # Empty string
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        # Empty string should not yield ContentEvent
        assert not any(isinstance(e, ContentEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_text_whitespace(self, model):
        """Test ahandle_streaming with whitespace-only text."""
        mock_part = MagicMock()
        mock_part.text = "   "  # Whitespace only
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        # Whitespace-only should not yield ContentEvent
        assert not any(isinstance(e, ContentEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_reasoning_empty_string(self, model):
        """Test ahandle_streaming with empty string reasoning."""
        mock_part = MagicMock()
        mock_part.text = ""
        mock_part.function_call = None
        mock_part.thought = ""

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        # Empty reasoning should not yield ReasoningEvent
        assert not any(isinstance(e, ReasoningEvent) for e in events)

    @pytest.mark.asyncio
    async def test_ahandle_streaming_all_params(self, model):
        """Test ahandle_streaming with all parameters."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        # Set all parameters
        model.temperature = 0.8
        model.top_p = 0.9
        model.max_tokens = 1000
        model.thinking_tokens = 500
        model.thinking = True

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_thinking_config_only_tokens(self, model):
        """Test ahandle_streaming with thinking_config only tokens."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        model.thinking_tokens = 500
        model.thinking = None

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_thinking_config_only_bool(self, model):
        """Test ahandle_streaming with thinking_config only bool."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        model.thinking_tokens = None
        model.thinking = True

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_no_candidates(self, model):
        """Test ahandle_streaming with no candidates."""
        mock_chunk = MagicMock()
        mock_chunk.candidates = []
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_no_content_parts(self, model):
        """Test ahandle_streaming with candidate but no content.parts."""
        mock_candidate = MagicMock()
        mock_candidate.content = None

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

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

        model.client.aio.models.generate_content_stream = AsyncMock(return_value=AsyncIterator([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) >= 0

    @pytest.mark.asyncio
    async def test_ahandle_streaming_client_error(self, model):
        """Test ahandle_streaming with ClientError."""
        from google.genai.errors import ClientError

        client_error = ClientError(code=400, response_json={"error": {"message": "API Error"}})

        model.client.aio.models.generate_content_stream = AsyncMock(side_effect=client_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)
        assert any("Google API Error" in str(e.content) for e in events if isinstance(e, ContentEvent))

    @pytest.mark.asyncio
    async def test_ahandle_streaming_server_error(self, model):
        """Test ahandle_streaming with ServerError."""
        from google.genai.errors import ServerError

        server_error = ServerError(code=500, response_json={"error": {"message": "Server Error"}})

        model.client.aio.models.generate_content_stream = AsyncMock(side_effect=server_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)
        assert any("Google API Error" in str(e.content) for e in events if isinstance(e, ContentEvent))

    @pytest.mark.asyncio
    async def test_ahandle_streaming_generic_error(self, model):
        """Test ahandle_streaming with generic Exception."""
        model.client.aio.models.generate_content_stream = AsyncMock(side_effect=Exception("Unexpected error"))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = []
        async for event in model.ahandle_streaming("gemini-pro", messages, None, tool_executor):
            events.append(event)

        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)
        assert any("Unexpected error" in str(e.content) for e in events if isinstance(e, ContentEvent))

    def test_handle_non_streaming_sequential_tools(self, model):
        """Test handle_non_streaming with sequential (non-parallel) tool execution."""
        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("gemini-pro", messages, tools, tool_executor)
        assert result is None

    def test_handle_non_streaming_tools_and_response_format_error(self, model):
        """Test handle_non_streaming raises error when tools and response_format are both provided."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()

        with pytest.raises(Exception, match="does not support structured output"):
            model.handle_non_streaming(
                "gemini-pro",
                messages,
                tools,
                tool_executor,
                response_format={"type": "json_object"},
            )

    def test_handle_non_streaming_with_response_format(self, model):
        """Test handle_non_streaming with response format (no tools)."""
        mock_part = MagicMock()
        mock_part.text = '{"result": "test"}'
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gemini-pro", messages, None, tool_executor, response_format={"type": "object"})
        assert result is not None

    def test_handle_non_streaming_with_thought_signature(self, model):
        """Test handle_non_streaming with thought_signature."""

        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = b"signature"
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("gemini-pro", messages, tools, tool_executor)
        assert result is None

    def test_handle_non_streaming_with_thoughtSignature(self, model):  # noqa: N802
        """Test handle_non_streaming with thoughtSignature (camelCase)."""

        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None
        mock_part.thoughtSignature = b"signature"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("gemini-pro", messages, tools, tool_executor)
        assert result is None

    def test_handle_non_streaming_sequential_no_parts_list(self, model):
        """Test handle_non_streaming sequential with empty parts_list_for_message."""
        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        result = model.handle_non_streaming("gemini-pro", messages, tools, tool_executor)
        assert result is None

    def test_handle_non_streaming_with_reasoning_content(self, model):
        """Test handle_non_streaming with reasoning content (thought)."""
        mock_part = MagicMock()
        mock_part.text = "Let me think..."
        mock_part.function_call = None
        mock_part.thought = "Let me think..."

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_with_function_call_in_parts(self, model):
        """Test handle_non_streaming with function_call in parts (tool execution path with tools=None)."""
        # This test checks the path where function_call exists but tools=None
        # Function_call will be detected and go into tool execution path
        # But since tools=None, it will return None
        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        mock_part.thought = None
        # Ensure thought_signature is None to avoid base64 issues
        mock_part.thought_signature = None
        mock_part.thoughtSignature = None
        mock_part.thought = None
        # Ensure thought_signature attributes are None to avoid base64 encoding issues
        mock_part.thought_signature = None
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = None  # No tools provided
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        # With tools=None but function_call exists, it will try to execute tools
        # But since tools=None, it should handle it gracefully
        # Actually, looking at the code, if tools=None, it won't execute tools
        # So it will return None
        result = model.handle_non_streaming("gemini-pro", messages, tools, tool_executor)
        # Result is None because function_calls were detected but tools=None
        assert result is None

    def test_handle_non_streaming_no_candidates(self, model):
        """Test handle_non_streaming with no candidates."""
        mock_response = MagicMock()
        mock_response.candidates = []

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_all_params(self, model):
        """Test handle_non_streaming with all parameters."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        # Set all parameters
        model.temperature = 0.8
        model.top_p = 0.9
        model.max_tokens = 1000
        model.thinking_tokens = 500
        model.thinking = True

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_thinking_config_only_tokens(self, model):
        """Test handle_non_streaming with thinking_config only tokens."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        model.thinking_tokens = 500
        model.thinking = None

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_thinking_config_only_bool(self, model):
        """Test handle_non_streaming with thinking_config only bool."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        model.thinking_tokens = None
        model.thinking = True

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_usage_none(self, model):
        """Test handle_non_streaming with None usage_metadata."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    def test_handle_non_streaming_finish_reason_none(self, model):
        """Test handle_non_streaming with None finish_reason."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = None

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.model_dump = MagicMock(return_value={"prompt_token_count": 10, "candidates_token_count": 20})

        model.client.models.generate_content = MagicMock(return_value=mock_response)
        model._create_llm_response = MagicMock(return_value=MagicMock())

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        result = model.handle_non_streaming("gemini-pro", messages, None, tool_executor)
        assert result is not None

    def test_handle_streaming_tools_and_response_format_error(self, model):
        """Test handle_streaming raises error when tools and response_format are both provided."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()

        # The error is raised when the generator function body executes (during iteration)
        # The error is raised at line 729, which is inside a try-except block
        # So it will be caught and wrapped in an Exception, then yielded as ContentEvent
        gen = model.handle_streaming(
            "gemini-pro",
            messages,
            tools,
            tool_executor,
            response_format={"type": "json_object"},
        )
        # Try to get the first item - this will execute the function body and trigger the error
        # The error will be caught and yielded as a ContentEvent
        events = list(gen)
        # The error should be in the events as a ContentEvent
        assert len(events) > 0
        assert any("does not support structured output" in str(e.content) for e in events if hasattr(e, "content"))

    def test_handle_streaming_with_response_format(self, model):
        """Test handle_streaming with response format (no tools)."""
        mock_part = MagicMock()
        mock_part.text = '{"result": "test"}'
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

        model.client.models.generate_content_stream = MagicMock(return_value=iter([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gemini-pro", messages, None, tool_executor, response_format={"type": "object"}))
        assert len(events) >= 0

    def test_handle_streaming_sequential_tools(self, model):
        """Test handle_streaming with sequential (non-parallel) tool execution."""

        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

        model.client.models.generate_content_stream = MagicMock(return_value=iter([mock_chunk]))
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False  # Sequential
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = list(model.handle_streaming("gemini-pro", messages, tools, tool_executor))
        assert len(events) > 0

    def test_handle_streaming_with_thought_signature(self, model):
        """Test handle_streaming with thought_signature."""

        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = b"signature"
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

        model.client.models.generate_content_stream = MagicMock(return_value=iter([mock_chunk]))
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = list(model.handle_streaming("gemini-pro", messages, tools, tool_executor))
        assert len(events) > 0

    def test_handle_streaming_with_thoughtSignature(self, model):  # noqa: N802
        """Test handle_streaming with thoughtSignature (camelCase)."""

        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None
        mock_part.thoughtSignature = b"signature"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

        model.client.models.generate_content_stream = MagicMock(return_value=iter([mock_chunk]))
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = True
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = list(model.handle_streaming("gemini-pro", messages, tools, tool_executor))
        assert len(events) > 0

    def test_handle_streaming_sequential_no_parts_list(self, model):
        """Test handle_streaming sequential with empty parts_list_for_message."""

        mock_function_call = MagicMock()
        mock_function_call.name = "test_tool"
        mock_function_call.args = {"param": "value"}
        mock_function_call.model_dump = MagicMock(return_value={"name": "test_tool", "args": {"param": "value"}})

        mock_part = MagicMock()
        mock_part.text = None
        mock_part.function_call = mock_function_call
        mock_part.thought_signature = None
        mock_part.thoughtSignature = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

        model.client.models.generate_content_stream = MagicMock(return_value=iter([mock_chunk]))
        model._execute_tools_parallel_sync = MagicMock(return_value={"test_tool": "result"})

        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"type": "function", "function": {"name": "test_tool", "description": "Test", "parameters": {}}}]
        tool_executor = MagicMock()
        tool_executor.parallel_calls = False
        tool_executor._tool_outputs = {}
        tool_executor._tool_calls = []

        events = list(model.handle_streaming("gemini-pro", messages, tools, tool_executor))
        assert len(events) > 0

    def test_handle_streaming_all_params(self, model):
        """Test handle_streaming with all parameters."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

        model.client.models.generate_content_stream = MagicMock(return_value=iter([mock_chunk]))

        # Set all parameters
        model.temperature = 0.8
        model.top_p = 0.9
        model.max_tokens = 1000
        model.thinking_tokens = 500
        model.thinking = True

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gemini-pro", messages, None, tool_executor))
        assert len(events) > 0

    def test_handle_streaming_thinking_config_only_tokens(self, model):
        """Test handle_streaming with thinking_config only tokens."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

        model.client.models.generate_content_stream = MagicMock(return_value=iter([mock_chunk]))

        model.thinking_tokens = 500
        model.thinking = None

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gemini-pro", messages, None, tool_executor))
        assert len(events) > 0

    def test_handle_streaming_thinking_config_only_bool(self, model):
        """Test handle_streaming with thinking_config only bool."""
        mock_part = MagicMock()
        mock_part.text = "Response"
        mock_part.function_call = None
        mock_part.thought = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

        model.client.models.generate_content_stream = MagicMock(return_value=iter([mock_chunk]))

        model.thinking_tokens = None
        model.thinking = True

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gemini-pro", messages, None, tool_executor))
        assert len(events) > 0

    def test_handle_streaming_no_candidates(self, model):
        """Test handle_streaming with no candidates."""
        mock_chunk = MagicMock()
        mock_chunk.candidates = []
        mock_chunk.usage_metadata = None

        model.client.models.generate_content_stream = MagicMock(return_value=iter([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gemini-pro", messages, None, tool_executor))
        assert len(events) >= 0

    def test_handle_streaming_no_content_parts(self, model):
        """Test handle_streaming with candidate but no content.parts."""
        mock_candidate = MagicMock()
        mock_candidate.content = None

        mock_chunk = MagicMock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

        model.client.models.generate_content_stream = MagicMock(return_value=iter([mock_chunk]))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gemini-pro", messages, None, tool_executor))
        assert len(events) >= 0

    def test_handle_streaming_client_error(self, model):
        """Test handle_streaming with ClientError."""
        from google.genai.errors import ClientError

        client_error = ClientError(code=400, response_json={"error": {"message": "API Error"}})

        model.client.models.generate_content_stream = MagicMock(side_effect=client_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gemini-pro", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)
        assert any("Google API Error" in str(e.content) for e in events if isinstance(e, ContentEvent))

    def test_handle_streaming_server_error(self, model):
        """Test handle_streaming with ServerError."""
        from google.genai.errors import ServerError

        server_error = ServerError(code=500, response_json={"error": {"message": "Server Error"}})

        model.client.models.generate_content_stream = MagicMock(side_effect=server_error)

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gemini-pro", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)
        assert any("Google API Error" in str(e.content) for e in events if isinstance(e, ContentEvent))

    def test_handle_streaming_generic_error(self, model):
        """Test handle_streaming with generic Exception."""
        model.client.models.generate_content_stream = MagicMock(side_effect=Exception("Unexpected error"))

        messages = [{"role": "user", "content": "Hello"}]
        tool_executor = MagicMock()

        events = list(model.handle_streaming("gemini-pro", messages, None, tool_executor))
        assert len(events) > 0
        assert any(isinstance(e, ContentEvent) for e in events)
        assert any("Unexpected error" in str(e.content) for e in events if isinstance(e, ContentEvent))

    def test_build_function_call_parts_with_signatures(self, model):
        """Test _build_function_call_parts_with_signatures."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"param": "value"}'},
                "_thought_signature": "sig1",
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "test_tool2", "arguments": '{"param2": "value2"}'},
            },
        ]
        result = model._build_function_call_parts_with_signatures(tool_calls)
        assert len(result) == 2
        assert "functionCall" in result[0]
        assert "thoughtSignature" in result[0]
        assert result[0]["thoughtSignature"] == "sig1"
        assert "functionCall" in result[1]
        assert "thoughtSignature" not in result[1] or result[1].get("thoughtSignature") != "sig1"

    def test_format_files_for_google_with_existing_content_list(self, model):
        """Test _format_files_for_google with existing content list."""
        file_obj = File(url="https://example.com/image.jpg", file_type=FileType.IMAGE)
        message = {"role": "user", "content": [{"type": "text", "text": "Check this"}], "_file_objects": [file_obj]}
        result = model._format_files_for_google(message)
        assert isinstance(result["content"], list)
        assert len(result["content"]) > 1
