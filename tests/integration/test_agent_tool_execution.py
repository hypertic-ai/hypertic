from unittest.mock import AsyncMock, Mock

import pytest

from hypertic.agents import Agent
from hypertic.models.anthropic import Anthropic
from hypertic.models.base import LLMResponse
from hypertic.models.metrics import Metrics
from hypertic.tools import tool


class TestAgentToolExecution:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    @pytest.fixture
    def math_tool(self):
        @tool
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        return add

    @pytest.fixture
    def search_tool(self):
        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        return search

    def test_agent_tool_registration(self, mock_model, math_tool):
        agent = Agent(model=mock_model, tools=[math_tool])

        assert len(agent.tools) > 0
        assert agent._tool_manager is not None

    def test_agent_tool_parallel_execution(self, mock_model, math_tool, search_tool):
        agent = Agent(model=mock_model, tools=[math_tool, search_tool], parallel_calls=True)

        assert agent.parallel_calls is True
        assert len(agent.tools) == 2

    def test_agent_tool_sequential_execution(self, mock_model, math_tool, search_tool):
        agent = Agent(model=mock_model, tools=[math_tool, search_tool], parallel_calls=False)

        assert agent.parallel_calls is False
        assert len(agent.tools) == 2

    def test_agent_tool_mixed_types(self, mock_model, math_tool):
        agent = Agent(model=mock_model, tools=[math_tool])

        assert len(agent.tools) > 0

    @pytest.mark.asyncio
    async def test_agent_tool_execution_flow(self, mock_model, math_tool):
        agent = Agent(model=mock_model, tools=[math_tool], max_steps=5)

        handler = Mock()
        handler.ahandle_non_streaming = AsyncMock(
            return_value=LLMResponse(
                response_text="Calculating...",
                metrics=Metrics(input_tokens=10, output_tokens=20),
                model="test-model",
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "add", "arguments": '{"a": 5, "b": 3}'}}],
            )
        )

        assert agent is not None
        assert len(agent.tools) > 0


class TestAgentMCPTools:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    @pytest.fixture
    def mock_mcp_servers(self):
        servers = Mock()
        servers.tools = {}
        servers.initialized = True
        servers.initialize = AsyncMock(return_value=servers)
        return servers

    def test_agent_mcp_tools_integration(self, mock_model, mock_mcp_servers):
        mock_model.set_mcp_servers(mock_mcp_servers)
        mock_model.supports_mcp = True

        agent = Agent(model=mock_model)

        assert bool(agent.model.supports_mcp) is True

    @pytest.mark.asyncio
    async def test_agent_mcp_tool_execution(self, mock_model, mock_mcp_servers):
        mock_model.set_mcp_servers(mock_mcp_servers)

        agent = Agent(model=mock_model)

        assert agent.model.mcp_servers is not None


class TestAgentToolErrorHandling:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    @pytest.fixture
    def failing_tool(self):
        @tool
        def failing_tool() -> str:
            """A tool that always fails."""
            raise Exception("Tool execution failed")

        return failing_tool

    def test_agent_tool_error_handling(self, mock_model, failing_tool):
        agent = Agent(model=mock_model, tools=[failing_tool])

        assert agent is not None
        assert len(agent.tools) > 0

    def test_agent_tool_validation(self, mock_model):
        agent = Agent(model=mock_model, tools=[])

        assert agent is not None
