import pytest

from hypertic.models.base import Base, LLMResponse
from hypertic.models.metrics import Metrics


class TestLLMResponse:
    def test_llm_response_creation(self):
        metrics = Metrics(
            input_tokens=10,
            output_tokens=20,
        )
        response = LLMResponse(
            response_text="Test response",
            metrics=metrics,
            model="test-model",
        )
        assert response.content == "Test response"
        assert response.response_text == "Test response"
        assert response.model == "test-model"
        assert response.metrics == metrics

    def test_llm_response_metadata(self):
        metrics = Metrics(
            input_tokens=10,
            output_tokens=20,
        )
        response = LLMResponse(
            response_text="Test",
            metrics=metrics,
            model="test-model",
            finish_reason="stop",
        )
        assert "model" in response.metadata
        assert "finish_reason" in response.metadata
        assert "input_tokens" in response.metadata
        assert "output_tokens" in response.metadata
        assert response.metadata["model"] == "test-model"
        assert response.metadata["finish_reason"] == "stop"

    def test_llm_response_tool_calls(self):
        metrics = Metrics(
            input_tokens=10,
            output_tokens=20,
        )
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "test_tool"}}]
        response = LLMResponse(
            response_text="Test",
            metrics=metrics,
            model="test-model",
            tool_calls=tool_calls,
        )
        assert response.tool_calls == tool_calls

    def test_llm_response_str_repr(self):
        metrics = Metrics(
            input_tokens=10,
            output_tokens=20,
        )
        response = LLMResponse(
            response_text="Test response",
            metrics=metrics,
            model="test-model",
        )
        str_repr = str(response)
        assert "content" in str_repr
        assert "metadata" in str_repr
        assert "tool_calls" in str_repr

    def test_llm_response_with_params(self):
        """Test LLMResponse with custom params."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        params = {"temperature": 0.7, "top_p": 0.9}
        response = LLMResponse(
            response_text="Test",
            metrics=metrics,
            model="test-model",
            params=params,
        )
        assert response.params == params
        assert response.metadata["params"] == params

    def test_llm_response_with_additional_metadata(self):
        """Test LLMResponse with additional metadata."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        additional_metadata = {"custom_field": "value", "another_field": 123}
        response = LLMResponse(
            response_text="Test",
            metrics=metrics,
            model="test-model",
            additional_metadata=additional_metadata,
        )
        assert response.additional_metadata == additional_metadata
        assert response.metadata["custom_field"] == "value"
        assert response.metadata["another_field"] == 123

    def test_llm_response_with_tool_outputs(self):
        """Test LLMResponse with tool outputs."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        tool_outputs = {"call_1": "result1", "call_2": "result2"}
        response = LLMResponse(
            response_text="Test",
            metrics=metrics,
            model="test-model",
            tool_outputs=tool_outputs,
        )
        assert response.tool_outputs == tool_outputs

    def test_llm_response_reasoning_from_metadata(self):
        """Test LLMResponse extracts reasoning from metadata."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        additional_metadata = {"reasoning_content": "This is reasoning"}
        response = LLMResponse(
            response_text="Test",
            metrics=metrics,
            model="test-model",
            additional_metadata=additional_metadata,
        )
        assert response.reasoning == "This is reasoning"

    def test_llm_response_reasoning_none_when_missing(self):
        """Test LLMResponse reasoning is None when not in metadata."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        response = LLMResponse(
            response_text="Test",
            metrics=metrics,
            model="test-model",
        )
        assert response.reasoning is None

    def test_llm_response_structured_output(self):
        """Test LLMResponse with structured output."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        structured_output = {"key": "value"}
        response = LLMResponse(
            response_text="Test",
            metrics=metrics,
            model="test-model",
        )
        # structured_output is set via additional_metadata
        response.additional_metadata["structured_output"] = structured_output
        # Note: The model_validator doesn't set structured_output from metadata,
        # so we test the field exists
        assert hasattr(response, "structured_output")

    def test_llm_response_default_values(self):
        """Test LLMResponse default values."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        response = LLMResponse(
            response_text="Test",
            metrics=metrics,
        )
        assert response.model == "unknown"
        assert response.finish_reason == "stop"
        assert response.params == {}
        assert response.tool_calls == []
        assert response.tool_outputs == {}
        assert response.content == "Test"
        assert response.metadata["model"] == "unknown"

    def test_llm_response_metadata_computation(self):
        """Test LLMResponse metadata is computed correctly."""
        metrics = Metrics(input_tokens=100, output_tokens=200)
        response = LLMResponse(
            response_text="Test response",
            metrics=metrics,
            model="gpt-4",
            finish_reason="length",
            params={"temperature": 0.7},
        )
        assert response.metadata["model"] == "gpt-4"
        assert response.metadata["finish_reason"] == "length"
        assert response.metadata["params"] == {"temperature": 0.7}
        assert response.metadata["input_tokens"] == 100
        assert response.metadata["output_tokens"] == 200

    def test_llm_response_repr_equals_str(self):
        """Test LLMResponse __repr__ equals __str__."""
        metrics = Metrics(input_tokens=10, output_tokens=20)
        response = LLMResponse(
            response_text="Test",
            metrics=metrics,
            model="test-model",
        )
        assert str(response) == repr(response)


class TestBase:
    def test_base_is_abstract(self):
        with pytest.raises(TypeError):
            Base(api_key="test_key", model="test-model")  # type: ignore[abstract]

    def test_base_has_required_methods(self):
        assert hasattr(Base, "ahandle_non_streaming")
        assert hasattr(Base, "ahandle_streaming")
        assert hasattr(Base, "handle_non_streaming")
        assert hasattr(Base, "handle_streaming")
        assert hasattr(Base, "set_mcp_servers")
        assert hasattr(Base, "_accumulate_metrics")

    def test_base_initialization(self):
        """Test Base class can be subclassed and initialized."""

        class ConcreteBase(Base):
            async def ahandle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            async def ahandle_streaming(self, model, messages, tools, tool_executor):
                yield None

            def handle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            def handle_streaming(self, model, messages, tools, tool_executor):
                yield None

        instance = ConcreteBase(
            api_key="test_key",
            model="test-model",
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            max_tokens=100,
        )
        assert instance.api_key == "test_key"
        assert instance.model == "test-model"
        assert instance.temperature == 0.7
        assert instance.top_p == 0.9
        assert instance.presence_penalty == 0.1
        assert instance.frequency_penalty == 0.1
        assert instance.max_tokens == 100
        assert instance.supports_mcp is False
        assert instance.mcp_servers is None

    def test_base_set_mcp_servers(self):
        """Test setting MCP servers."""
        from unittest.mock import Mock

        class ConcreteBase(Base):
            async def ahandle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            async def ahandle_streaming(self, model, messages, tools, tool_executor):
                yield None

            def handle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            def handle_streaming(self, model, messages, tools, tool_executor):
                yield None

        instance = ConcreteBase()
        mcp_servers = Mock()
        instance.set_mcp_servers(mcp_servers)
        assert instance.mcp_servers == mcp_servers
        assert instance.supports_mcp is True

    def test_base_accumulate_metrics(self):
        """Test metrics accumulation."""

        class ConcreteBase(Base):
            async def ahandle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            async def ahandle_streaming(self, model, messages, tools, tool_executor):
                yield None

            def handle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            def handle_streaming(self, model, messages, tools, tool_executor):
                yield None

        instance = ConcreteBase()
        usage1 = {"prompt_tokens": 10, "completion_tokens": 20}
        usage2 = {"prompt_tokens": 5, "completion_tokens": 15}

        instance._accumulate_metrics(usage1)
        metrics = instance._get_cumulative_metrics()
        assert metrics.input_tokens == 10
        assert metrics.output_tokens == 20

        instance._accumulate_metrics(usage2)
        metrics = instance._get_cumulative_metrics()
        assert metrics.input_tokens == 15
        assert metrics.output_tokens == 35

    def test_base_reset_metrics(self):
        """Test metrics reset."""

        class ConcreteBase(Base):
            async def ahandle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            async def ahandle_streaming(self, model, messages, tools, tool_executor):
                yield None

            def handle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            def handle_streaming(self, model, messages, tools, tool_executor):
                yield None

        instance = ConcreteBase()
        instance._accumulate_metrics({"prompt_tokens": 10, "completion_tokens": 20})
        instance._reset_metrics()
        metrics = instance._get_cumulative_metrics()
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0

    def test_base_create_streaming_metadata(self):
        """Test streaming metadata creation."""

        class ConcreteBase(Base):
            async def ahandle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            async def ahandle_streaming(self, model, messages, tools, tool_executor):
                yield None

            def handle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            def handle_streaming(self, model, messages, tools, tool_executor):
                yield None

        instance = ConcreteBase(
            model="test-model",
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            max_tokens=100,
        )
        instance._accumulate_metrics({"prompt_tokens": 10, "completion_tokens": 20})

        metadata = instance._create_streaming_metadata()
        assert metadata["model"] == "test-model"
        assert metadata["finish_reason"] == "stop"
        assert metadata["params"]["temperature"] == 0.7
        assert metadata["params"]["top_p"] == 0.9
        assert metadata["params"]["presence_penalty"] == 0.1
        assert metadata["params"]["frequency_penalty"] == 0.2
        assert metadata["params"]["max_tokens"] == 100
        assert metadata["input_tokens"] == 10
        assert metadata["output_tokens"] == 20

    def test_base_create_streaming_metadata_with_custom_params(self):
        """Test streaming metadata with custom params."""

        class ConcreteBase(Base):
            async def ahandle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            async def ahandle_streaming(self, model, messages, tools, tool_executor):
                yield None

            def handle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            def handle_streaming(self, model, messages, tools, tool_executor):
                yield None

        instance = ConcreteBase(model="test-model")
        custom_params = {"custom_param": "value"}
        additional_metrics = {"custom_metric": 123}

        metadata = instance._create_streaming_metadata(
            finish_reason="length",
            params=custom_params,
            additional_metrics=additional_metrics,
        )
        assert metadata["finish_reason"] == "length"
        assert metadata["params"] == custom_params
        assert metadata["custom_metric"] == 123

    def test_base_create_llm_response(self):
        """Test LLM response creation."""

        class ConcreteBase(Base):
            async def ahandle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            async def ahandle_streaming(self, model, messages, tools, tool_executor):
                yield None

            def handle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            def handle_streaming(self, model, messages, tools, tool_executor):
                yield None

        instance = ConcreteBase(model="test-model")
        instance._accumulate_metrics({"prompt_tokens": 10, "completion_tokens": 20})

        usage = {
            "model": "test-model-v2",
            "finish_reason": "stop",
            "params": {"temperature": 0.7},
            "input_tokens": 10,
            "output_tokens": 20,
            "custom_field": "value",
        }

        response = instance._create_llm_response(
            content="Test response",
            usage=usage,
            tool_calls=[{"id": "call_1"}],
            tool_outputs={"call_1": "result"},
        )

        assert response.response_text == "Test response"
        assert response.model == "test-model-v2"
        assert response.finish_reason == "stop"
        assert response.params == {"temperature": 0.7}
        assert response.tool_calls == [{"id": "call_1"}]
        assert response.tool_outputs == {"call_1": "result"}
        assert response.additional_metadata["custom_field"] == "value"
        assert response.metrics.input_tokens == 10
        assert response.metrics.output_tokens == 20

    def test_base_execute_tool_sync(self):
        """Test synchronous tool execution."""
        from unittest.mock import Mock

        class ConcreteBase(Base):
            async def ahandle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            async def ahandle_streaming(self, model, messages, tools, tool_executor):
                yield None

            def handle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            def handle_streaming(self, model, messages, tools, tool_executor):
                yield None

        instance = ConcreteBase()
        tool_executor = Mock()
        tool_executor._execute_tool.return_value = "result"

        result = instance._execute_tool_sync(tool_executor, "test_tool", {"param": "value"})
        assert result == "result"
        tool_executor._execute_tool.assert_called_once_with("test_tool", {"param": "value"})

    def test_base_execute_tools_parallel_sync(self):
        """Test parallel synchronous tool execution."""
        from unittest.mock import Mock

        class ConcreteBase(Base):
            async def ahandle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            async def ahandle_streaming(self, model, messages, tools, tool_executor):
                yield None

            def handle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            def handle_streaming(self, model, messages, tools, tool_executor):
                yield None

        instance = ConcreteBase()
        tool_executor = Mock()
        tool_executor._execute_tools_parallel.return_value = {"tool1": "result1", "tool2": "result2"}

        tool_calls = [{"name": "tool1"}, {"name": "tool2"}]
        result = instance._execute_tools_parallel_sync(tool_executor, tool_calls)
        assert result == {"tool1": "result1", "tool2": "result2"}

    @pytest.mark.asyncio
    async def test_base_execute_tool_async(self):
        """Test asynchronous tool execution."""
        from unittest.mock import AsyncMock, Mock

        class ConcreteBase(Base):
            async def ahandle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            async def ahandle_streaming(self, model, messages, tools, tool_executor):
                yield None

            def handle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            def handle_streaming(self, model, messages, tools, tool_executor):
                yield None

        instance = ConcreteBase()
        tool_executor = Mock()
        tool_executor._aexecute_tool = AsyncMock(return_value="result")

        result = await instance._execute_tool_async(tool_executor, "test_tool", {"param": "value"})
        assert result == "result"
        tool_executor._aexecute_tool.assert_called_once_with("test_tool", {"param": "value"})

    @pytest.mark.asyncio
    async def test_base_execute_tools_parallel_async(self):
        """Test parallel asynchronous tool execution."""
        from unittest.mock import AsyncMock, Mock

        class ConcreteBase(Base):
            async def ahandle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            async def ahandle_streaming(self, model, messages, tools, tool_executor):
                yield None

            def handle_non_streaming(self, model, messages, tools, tool_executor):
                return None

            def handle_streaming(self, model, messages, tools, tool_executor):
                yield None

        instance = ConcreteBase()
        tool_executor = Mock()
        tool_executor._aexecute_tools_parallel = AsyncMock(return_value={"tool1": "result1", "tool2": "result2"})

        tool_calls = [{"name": "tool1"}, {"name": "tool2"}]
        result = await instance._execute_tools_parallel_async(tool_executor, tool_calls)
        assert result == {"tool1": "result1", "tool2": "result2"}
