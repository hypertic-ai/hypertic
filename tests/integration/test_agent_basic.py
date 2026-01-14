from unittest.mock import Mock

import pytest

from hypertic.agents import Agent
from hypertic.models.base import Base, LLMResponse
from hypertic.models.metrics import Metrics
from hypertic.tools import tool


class TestAgentBasic:
    """Basic integration tests for Agent."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = Mock(spec=Base)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    @pytest.fixture
    def mock_response(self):
        """Create a mock LLM response."""
        metrics = Metrics(
            input_tokens=10,
            output_tokens=20,
        )
        return LLMResponse(
            response_text="Test response",
            metrics=metrics,
            model="test-model",
        )

    def test_agent_creation(self, mock_model):
        """Test creating an Agent."""
        agent = Agent(model=mock_model, instructions="Test instructions")
        assert agent.model == mock_model
        assert agent.instructions == "Test instructions"

    def test_agent_with_tools(self, mock_model):
        """Test creating an Agent with tools."""

        @tool
        def sample_tool() -> str:
            """Sample tool."""
            return "ok"

        agent = Agent(model=mock_model, tools=[sample_tool])
        assert len(agent.tools) > 0

    def test_agent_with_memory(self, mock_model, mock_memory):
        """Test creating an Agent with memory."""
        agent = Agent(model=mock_model, memory=mock_memory)
        assert agent.memory == mock_memory

    def test_agent_with_guardrails(self, mock_model):
        """Test creating an Agent with guardrails."""
        from hypertic.guardrails import Guardrail

        guardrail = Guardrail()
        agent = Agent(model=mock_model, guardrails=[guardrail])
        assert len(agent.guardrails) == 1
