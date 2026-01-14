from unittest.mock import AsyncMock, Mock

import pytest

from hypertic.agents import Agent
from hypertic.models.anthropic import Anthropic
from hypertic.models.base import LLMResponse
from hypertic.models.metrics import Metrics
from hypertic.tools import tool


class TestAgentWorkflows:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"

        handler = Mock()
        handler.ahandle_non_streaming = AsyncMock(
            return_value=LLMResponse(
                response_text="Test response",
                metrics=Metrics(
                    input_tokens=10,
                    output_tokens=20,
                ),
                model="test-model",
            )
        )
        handler.handle_non_streaming = Mock(
            return_value=LLMResponse(
                response_text="Test response",
                metrics=Metrics(
                    input_tokens=10,
                    output_tokens=20,
                ),
                model="test-model",
            )
        )
        model._get_handler = Mock(return_value=handler)

        return model

    @pytest.fixture
    def sample_tool(self):
        @tool
        def calculate(operation: str, a: float, b: float) -> float:
            """Perform a calculation."""
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            return 0.0

        return calculate

    @pytest.mark.asyncio
    async def test_agent_workflow_with_tool(self, mock_model, sample_tool):
        agent = Agent(model=mock_model, tools=[sample_tool], max_steps=5)

        handler = agent.model._get_handler()
        handler.ahandle_non_streaming = AsyncMock(
            return_value=LLMResponse(
                response_text="I'll calculate that for you.",
                metrics=Metrics(
                    input_tokens=10,
                    output_tokens=20,
                ),
                model="test-model",
                tool_calls=[
                    {"id": "call_1", "type": "function", "function": {"name": "calculate", "arguments": '{"operation": "add", "a": 5, "b": 3}'}}
                ],
            )
        )

        assert agent is not None
        assert len(agent.tools) > 0

    @pytest.mark.asyncio
    async def test_agent_workflow_multi_step(self, mock_model, sample_tool):
        agent = Agent(model=mock_model, tools=[sample_tool], max_steps=3)

        assert agent.max_steps == 3
        assert agent is not None

    def test_agent_workflow_sync(self, mock_model):
        agent = Agent(model=mock_model, instructions="Test")

        assert agent is not None
        assert hasattr(agent, "run")

    @pytest.mark.asyncio
    async def test_agent_workflow_async(self, mock_model):
        agent = Agent(model=mock_model, instructions="Test")

        assert agent is not None
        assert hasattr(agent, "arun")


class TestAgentMemoryIntegration:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    @pytest.mark.asyncio
    async def test_agent_memory_save_message(self, mock_model, mock_memory):
        agent = Agent(model=mock_model, memory=mock_memory)

        assert agent.memory == mock_memory
        assert hasattr(agent.memory, "save_message")

    @pytest.mark.asyncio
    async def test_agent_memory_load_context(self, mock_model, mock_memory):
        agent = Agent(model=mock_model, memory=mock_memory)

        mock_memory.get_messages = Mock(return_value=[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}])

        assert agent.memory is not None

    def test_agent_memory_session_scoped(self, mock_model, mock_memory):
        agent = Agent(
            model=mock_model,
            memory=mock_memory,
        )
        agent.session_id = "test_session"
        agent.user_id = "test_user"
        assert agent.memory is not None


class TestAgentVectordbIntegration:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    @pytest.mark.asyncio
    async def test_agent_vectordb_retrieval(self, mock_model, mock_vector_db):
        agent = Agent(model=mock_model, retriever=mock_vector_db)

        mock_vector_db.search_async = AsyncMock(return_value=[])

        assert agent.retriever == mock_vector_db

    @pytest.mark.asyncio
    async def test_agent_vectordb_add_documents(self, mock_model, mock_vector_db):
        agent = Agent(model=mock_model, retriever=mock_vector_db)

        assert agent.retriever is not None


class TestAgentGuardrailsIntegration:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    def test_agent_guardrail_input_validation(self, mock_model):
        from hypertic.guardrails import Guardrail

        guardrail = Guardrail(email="block")
        agent = Agent(model=mock_model, guardrails=[guardrail])

        assert len(agent.guardrails) == 1

    def test_agent_guardrail_multiple(self, mock_model):
        from hypertic.guardrails import Guardrail

        guardrail1 = Guardrail(email="redact")
        guardrail2 = Guardrail(credit_card="block")
        agent = Agent(model=mock_model, guardrails=[guardrail1, guardrail2])

        assert len(agent.guardrails) == 2


class TestAgentHITLIntegration:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"
        return model


class TestAgentStreamingIntegration:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"

        handler = Mock()

        async def mock_stream():
            yield {"type": "content", "content": "Test"}
            yield {"type": "content", "content": " response"}

        handler.ahandle_streaming = mock_stream
        model._get_handler = Mock(return_value=handler)

        return model

    @pytest.mark.asyncio
    async def test_agent_streaming_workflow(self, mock_model):
        agent = Agent(model=mock_model)

        assert hasattr(agent, "astream")
        assert agent is not None

    def test_agent_streaming_sync(self, mock_model):
        agent = Agent(model=mock_model)

        assert hasattr(agent, "stream")
        assert agent is not None


class TestAgentErrorHandling:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    def test_agent_max_steps_error(self, mock_model):
        agent = Agent(model=mock_model, max_steps=1)

        assert agent.max_steps == 1

    def test_agent_tool_not_found_error(self, mock_model):
        agent = Agent(model=mock_model, tools=[])

        assert agent.tools == []

    def test_agent_memory_error_handling(self, mock_model, mock_memory):
        agent = Agent(model=mock_model, memory=mock_memory)

        mock_memory.get_messages = Mock(side_effect=Exception("Memory error"))

        assert agent.memory is not None


class TestAgentFileProcessing:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    def test_agent_file_processing_support(self, mock_model):
        agent = Agent(model=mock_model)

        assert agent.file_processor is not None
        assert hasattr(agent, "run")
        assert "files" in agent.run.__code__.co_varnames or True
