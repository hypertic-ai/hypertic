from unittest.mock import AsyncMock, Mock

import pytest

from hypertic.agents import Agent
from hypertic.embedders.openai import OpenAIEmbedder
from hypertic.guardrails import Guardrail
from hypertic.memory.postgres.postgres import PostgresServer
from hypertic.models.anthropic import Anthropic
from hypertic.tools import tool
from hypertic.vectordb.chroma.chroma import ChromaDB


@pytest.fixture
def mock_model():
    model = Mock(spec=Anthropic)
    model.api_key = "test_key"
    model.model = "test-model"
    return model


@pytest.fixture
def mock_embedder():
    embedder = Mock(spec=OpenAIEmbedder)
    embedder.embed = AsyncMock(return_value=[0.1] * 100)
    embedder.embed_sync = Mock(return_value=[0.1] * 100)
    return embedder


@pytest.fixture
def mock_vectordb(mock_embedder):
    db = Mock(spec=ChromaDB)
    db.embedder = mock_embedder
    db.search_async = AsyncMock(return_value=[])
    db.add_documents_async = AsyncMock(return_value=None)
    return db


@pytest.fixture
def mock_memory():
    memory = Mock(spec=PostgresServer)
    memory.get_messages = Mock(return_value=[])
    memory.save_message = Mock(return_value=None)
    memory.aget_messages = AsyncMock(return_value=[])
    memory.asave_message = AsyncMock(return_value=None)
    return memory


@pytest.fixture
def sample_tool():
    @tool
    def test_tool(param: str) -> str:
        """Test tool."""
        return f"Result: {param}"

    return test_tool


class TestEndToEndWorkflows:
    def test_agent_with_all_components(self, mock_model, mock_vectordb, mock_memory, sample_tool):
        guardrail = Guardrail(email="redact")

        agent = Agent(
            model=mock_model,
            instructions="Test instructions",
            tools=[sample_tool],
            retriever=mock_vectordb,
            memory=mock_memory,
            guardrails=[guardrail],
            max_steps=5,
        )

        assert agent.model == mock_model
        assert agent.retriever == mock_vectordb
        assert agent.memory == mock_memory
        assert len(agent.guardrails) == 1
        assert len(agent.tools) > 0

    def test_rag_with_memory_workflow(self, mock_model, mock_vectordb, mock_memory):
        agent = Agent(
            model=mock_model,
            retriever=mock_vectordb,
            memory=mock_memory,
        )

        assert agent.retriever is not None
        assert agent.memory is not None

    def test_agent_with_guardrails_and_tools(self, mock_model, sample_tool):
        guardrail = Guardrail(email="block")

        agent = Agent(
            model=mock_model,
            tools=[sample_tool],
            guardrails=[guardrail],
        )

        assert len(agent.tools) > 0
        assert len(agent.guardrails) == 1

    def test_agent_multi_step_with_memory(self, mock_model, mock_memory, sample_tool):
        agent = Agent(
            model=mock_model,
            tools=[sample_tool],
            memory=mock_memory,
            max_steps=3,
        )

        assert agent.max_steps == 3
        assert agent.memory is not None
        assert len(agent.tools) > 0

    def test_agent_rag_guardrails_memory(self, mock_model, mock_vectordb, mock_memory):
        guardrail = Guardrail(email="redact")

        agent = Agent(
            model=mock_model,
            retriever=mock_vectordb,
            memory=mock_memory,
            guardrails=[guardrail],
        )

        assert agent.retriever is not None
        assert agent.memory is not None
        assert len(agent.guardrails) == 1


class TestComponentIntegration:
    def test_embedder_with_vectordb(self, mock_embedder):
        db = ChromaDB(embedder=mock_embedder, collection="test")

        assert db.embedder == mock_embedder

    def test_memory_with_agent(self, mock_model):
        memory = PostgresServer(db_url="postgresql://localhost/test")
        agent = Agent(model=mock_model, memory=memory)

        assert agent.memory == memory

    def test_guardrails_with_agent(self, mock_model):
        guardrail = Guardrail(email="block")
        agent = Agent(model=mock_model, guardrails=[guardrail])

        assert len(agent.guardrails) == 1

    def test_tools_with_agent(self, mock_model):
        @tool
        def test_tool(param: str) -> str:
            """Test tool."""
            return "result"

        agent = Agent(model=mock_model, tools=[test_tool])

        assert len(agent.tools) > 0

    def test_vectordb_with_agent(self, mock_model, mock_vectordb):
        agent = Agent(model=mock_model, retriever=mock_vectordb)

        assert agent.retriever == mock_vectordb
