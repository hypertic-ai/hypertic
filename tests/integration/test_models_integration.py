from unittest.mock import Mock

import pytest

from hypertic.models.anthropic import Anthropic
from hypertic.models.base import LLMResponse
from hypertic.models.cohere import Cohere
from hypertic.models.deepseek import DeepSeek
from hypertic.models.google import GoogleAI
from hypertic.models.groq import Groq
from hypertic.models.metrics import Metrics
from hypertic.models.mistral import MistralAI
from hypertic.models.moonshot import MoonshotAI
from hypertic.models.openai import OpenAIChat
from hypertic.models.qwen import Qwen
from hypertic.models.xai import XAI


class TestModelProvidersIntegration:
    @pytest.mark.parametrize(
        "model_class,model_name",
        [
            (Anthropic, "claude-3-5-sonnet-20241022"),
            (OpenAIChat, "gpt-4o"),
            (Cohere, "command-r-plus"),
            (GoogleAI, "gemini-pro"),
            (Groq, "llama-3.1-70b-versatile"),
            (MistralAI, "mistral-large-latest"),
            (DeepSeek, "deepseek-chat"),
            (Qwen, "qwen-plus"),
            (MoonshotAI, "moonshot-v1-8k"),
            (XAI, "grok-beta"),
        ],
    )
    def test_model_provider_initialization(self, model_class, model_name, mock_api_key):
        model = model_class(api_key=mock_api_key, model=model_name)
        assert model.api_key == mock_api_key
        assert model.model == model_name

    @pytest.mark.parametrize(
        "model_class",
        [
            Anthropic,
            OpenAIChat,
            Cohere,
            GoogleAI,
            Groq,
            MistralAI,
            DeepSeek,
            Qwen,
            MoonshotAI,
            XAI,
        ],
    )
    def test_model_provider_metrics_accumulation(self, model_class, mock_api_key):
        model = model_class(api_key=mock_api_key, model="test-model")
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        model._accumulate_metrics(usage)
        metrics = model._get_cumulative_metrics()
        assert metrics.input_tokens == 10
        assert metrics.output_tokens == 20

    @pytest.mark.parametrize(
        "model_class",
        [
            Anthropic,
            OpenAIChat,
            Cohere,
            GoogleAI,
            Groq,
            MistralAI,
            DeepSeek,
            Qwen,
            MoonshotAI,
            XAI,
        ],
    )
    def test_model_provider_mcp_support(self, model_class, mock_api_key):
        model = model_class(api_key=mock_api_key, model="test-model")
        mock_mcp = Mock()
        model.set_mcp_servers([mock_mcp])
        assert model.supports_mcp is True
        assert model.mcp_servers == [mock_mcp]


class TestModelHandlerIntegration:
    @pytest.fixture
    def mock_model(self, mock_api_key):
        model = Anthropic(api_key=mock_api_key, model="claude-3-5-sonnet-20241022")
        return model

    def test_model_handler_initialization(self, mock_model):
        assert hasattr(mock_model, "ahandle_non_streaming")
        assert hasattr(mock_model, "handle_non_streaming")

    def test_model_handler_has_required_methods(self, mock_model):
        assert hasattr(mock_model, "ahandle_non_streaming")
        assert hasattr(mock_model, "ahandle_streaming")
        assert hasattr(mock_model, "handle_non_streaming")
        assert hasattr(mock_model, "handle_streaming")

    @pytest.mark.asyncio
    async def test_model_handler_async_methods(self, mock_model):
        assert callable(mock_model.ahandle_non_streaming)
        assert callable(mock_model.ahandle_streaming)

    def test_model_handler_sync_methods(self, mock_model):
        assert callable(mock_model.handle_non_streaming)
        assert callable(mock_model.handle_streaming)


class TestModelResponseIntegration:
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
        assert response.metrics == metrics

    def test_llm_response_with_tool_calls(self):
        metrics = Metrics(
            input_tokens=10,
            output_tokens=20,
        )
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "test"}}]
        response = LLMResponse(
            response_text="Test",
            metrics=metrics,
            model="test-model",
            tool_calls=tool_calls,
        )
        assert response.tool_calls == tool_calls

    def test_llm_response_metadata(self):
        metrics = Metrics(
            input_tokens=10,
            output_tokens=20,
        )
        response = LLMResponse(
            response_text="Test",
            metrics=metrics,
            model="test-model",
        )
        assert "model" in response.metadata
        assert "input_tokens" in response.metadata
        assert "output_tokens" in response.metadata
