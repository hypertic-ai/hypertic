from unittest.mock import patch

import pytest

from hypertic.models.anthropic import Anthropic
from hypertic.models.cohere import Cohere
from hypertic.models.deepseek import DeepSeek
from hypertic.models.google import GoogleAI
from hypertic.models.groq import Groq
from hypertic.models.mistral import MistralAI
from hypertic.models.moonshot import MoonshotAI
from hypertic.models.openai import OpenAIChat
from hypertic.models.qwen import Qwen
from hypertic.models.xai import XAI


class TestModelInitialization:
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
    def test_model_creation(self, model_class, model_name, mock_api_key):
        model = model_class(api_key=mock_api_key, model=model_name)
        assert model.api_key == mock_api_key
        assert model.model == model_name

    def test_model_with_optional_params(self, mock_api_key):
        model = Anthropic(
            api_key=mock_api_key,
            model="claude-haiku-4-5",
            temperature=0.7,
            max_tokens=1000,
        )
        assert model.temperature == 0.7
        assert model.max_tokens == 1000

    def test_model_set_mcp_servers(self, mock_api_key):
        model = Anthropic(api_key=mock_api_key, model="claude-haiku-4-5")
        mock_servers = [object()]
        model.set_mcp_servers(mock_servers)
        assert model.mcp_servers == mock_servers
        assert model.supports_mcp is True


class TestModelMetrics:
    def test_metrics_accumulation(self, mock_api_key):
        model = Anthropic(api_key=mock_api_key, model="claude-3-5-sonnet-20241022")
        usage1 = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        usage2 = {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
        model._accumulate_metrics(usage1)
        model._accumulate_metrics(usage2)
        metrics = model._get_cumulative_metrics()
        assert metrics.input_tokens == 25
        assert metrics.output_tokens == 45

    def test_metrics_reset(self, mock_api_key):
        model = Anthropic(api_key=mock_api_key, model="claude-3-5-sonnet-20241022")
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        model._accumulate_metrics(usage)
        model._reset_metrics()
        metrics = model._get_cumulative_metrics()
        assert metrics.input_tokens == 0
        assert metrics.output_tokens == 0


class TestAnthropic:
    @pytest.fixture
    def model(self, mock_api_key):
        return Anthropic(api_key=mock_api_key, model="claude-3-5-sonnet-20241022")

    def test_anthropic_creation(self, model, mock_api_key):
        assert model.api_key == mock_api_key
        assert isinstance(model, Anthropic)

    @pytest.mark.asyncio
    @patch("hypertic.models.anthropic.anthropic.AsyncAnthropicClient")
    async def test_anthropic_ahandle_non_streaming(self, mock_client_class, model):
        assert hasattr(model, "ahandle_non_streaming")
        assert callable(model.ahandle_non_streaming)


class TestOpenAI:
    @pytest.fixture
    def model(self, mock_api_key):
        return OpenAIChat(api_key=mock_api_key, model="gpt-4o")

    def test_openai_creation(self, model, mock_api_key):
        assert model.api_key == mock_api_key
        assert isinstance(model, OpenAIChat)

    @pytest.mark.asyncio
    async def test_openai_has_required_methods(self, model):
        assert hasattr(model, "ahandle_non_streaming")
        assert hasattr(model, "ahandle_streaming")
        assert hasattr(model, "handle_non_streaming")
        assert hasattr(model, "handle_streaming")


class TestCohere:
    @pytest.fixture
    def model(self, mock_api_key):
        return Cohere(api_key=mock_api_key, model="command-r-plus")

    def test_cohere_creation(self, model, mock_api_key):
        assert model.api_key == mock_api_key
        assert isinstance(model, Cohere)


class TestGoogleAI:
    @pytest.fixture
    def model(self, mock_api_key):
        return GoogleAI(api_key=mock_api_key, model="gemini-pro")

    def test_google_creation(self, model, mock_api_key):
        assert model.api_key == mock_api_key
        assert isinstance(model, GoogleAI)


class TestGroq:
    @pytest.fixture
    def model(self, mock_api_key):
        return Groq(api_key=mock_api_key, model="llama-3.1-70b-versatile")

    def test_groq_creation(self, model, mock_api_key):
        assert model.api_key == mock_api_key
        assert isinstance(model, Groq)


class TestMistralAI:
    @pytest.fixture
    def model(self, mock_api_key):
        return MistralAI(api_key=mock_api_key, model="mistral-large-latest")

    def test_mistral_creation(self, model, mock_api_key):
        assert model.api_key == mock_api_key
        assert isinstance(model, MistralAI)


class TestDeepSeek:
    @pytest.fixture
    def model(self, mock_api_key):
        return DeepSeek(api_key=mock_api_key, model="deepseek-chat")

    def test_deepseek_creation(self, model, mock_api_key):
        assert model.api_key == mock_api_key
        assert isinstance(model, DeepSeek)


class TestQwen:
    @pytest.fixture
    def model(self, mock_api_key):
        return Qwen(api_key=mock_api_key, model="qwen-plus")

    def test_qwen_creation(self, model, mock_api_key):
        assert model.api_key == mock_api_key
        assert isinstance(model, Qwen)


class TestMoonshotAI:
    @pytest.fixture
    def model(self, mock_api_key):
        return MoonshotAI(api_key=mock_api_key, model="moonshot-v1-8k")

    def test_moonshot_creation(self, model, mock_api_key):
        assert model.api_key == mock_api_key
        assert isinstance(model, MoonshotAI)


class TestXAI:
    @pytest.fixture
    def model(self, mock_api_key):
        return XAI(api_key=mock_api_key, model="grok-beta")

    def test_xai_creation(self, model, mock_api_key):
        assert model.api_key == mock_api_key
        assert isinstance(model, XAI)
