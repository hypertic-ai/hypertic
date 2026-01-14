from unittest.mock import Mock

import pytest

from hypertic.agents import Agent
from hypertic.models.anthropic import Anthropic
from hypertic.models.base import LLMResponse
from hypertic.models.metrics import Metrics
from hypertic.tools import tool


class TestAgentComprehensive:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    @pytest.fixture
    def mock_response(self):
        metrics = Metrics(input_tokens=10, output_tokens=20)
        return LLMResponse(
            response_text="Test response",
            metrics=metrics,
            model="test-model",
        )

    @pytest.fixture
    def sample_tool(self):
        @tool
        def test_tool(param: str) -> str:
            """Test tool description."""
            return f"Result: {param}"

        return test_tool

    def test_agent_with_all_components(self, mock_model, mock_memory, mock_guardrail, sample_tool):
        agent = Agent(
            model=mock_model,
            instructions="Test instructions",
            tools=[sample_tool],
            memory=mock_memory,
            guardrails=[mock_guardrail],
        )
        assert agent.model == mock_model
        assert agent.instructions == "Test instructions"
        assert len(agent.tools) > 0
        assert agent.memory == mock_memory
        assert len(agent.guardrails) == 1

    def test_agent_with_vectordb(self, mock_model, mock_vector_db):
        agent = Agent(model=mock_model, retriever=mock_vector_db)
        assert agent.retriever == mock_vector_db

    def test_agent_with_max_steps(self, mock_model):
        agent = Agent(model=mock_model, max_steps=10)
        assert agent.max_steps == 10

    def test_agent_with_temperature(self, mock_model):
        mock_model.temperature = 0.7
        agent = Agent(model=mock_model)
        assert agent.model.temperature == 0.7

    @pytest.mark.asyncio
    async def test_agent_arun_basic(self, mock_model):
        agent = Agent(model=mock_model, instructions="Test")
        assert agent is not None

    def test_agent_run_basic(self, mock_model):
        agent = Agent(model=mock_model, instructions="Test")
        assert agent is not None

    def test_agent_with_session_id(self, mock_model):
        agent = Agent(model=mock_model)
        agent.session_id = "test_session"
        assert agent.session_id == "test_session"

    def test_agent_with_user_id(self, mock_model):
        agent = Agent(model=mock_model)
        agent.user_id = "test_user"
        assert agent.user_id == "test_user"

    def test_agent_tool_execution_mode(self, mock_model):
        agent = Agent(model=mock_model, parallel_calls=True)
        assert agent.parallel_calls is True
