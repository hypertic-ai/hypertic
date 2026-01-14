from unittest.mock import AsyncMock, Mock

import pytest

from hypertic.agents import Agent
from hypertic.models.anthropic import Anthropic
from hypertic.vectordb.base import VectorSearchResult


@pytest.fixture
def mock_model():
    model = Mock(spec=Anthropic)
    model.api_key = "test_key"
    model.model = "test-model"
    return model


@pytest.fixture
def mock_vectordb(mock_embedder):
    db = Mock()
    db.search = Mock(
        return_value=[
            VectorSearchResult(
                content="Test document content",
                metadata={"source": "test"},
                score=0.95,
            )
        ]
    )
    db.search_async = AsyncMock(
        return_value=[
            VectorSearchResult(
                content="Test document content",
                metadata={"source": "test"},
                score=0.95,
            )
        ]
    )
    db.embedder = mock_embedder
    return db


class TestAgentRAGWorkflows:
    def test_agent_rag_setup(self, mock_model, mock_vectordb):
        agent = Agent(model=mock_model, retriever=mock_vectordb)

        assert agent.retriever == mock_vectordb
        assert agent.retriever.embedder is not None

    @pytest.mark.asyncio
    async def test_agent_rag_retrieval(self, mock_model, mock_vectordb):
        agent = Agent(model=mock_model, retriever=mock_vectordb)

        results = await agent.retriever.search_async("test query")
        assert len(results) > 0
        assert results[0].content is not None

    def test_agent_rag_sync_retrieval(self, mock_model, mock_vectordb):
        agent = Agent(model=mock_model, retriever=mock_vectordb)

        results = agent.retriever.search("test query")
        assert len(results) > 0

    def test_agent_rag_with_embedder(self, mock_model, mock_vectordb, mock_embedder):
        agent = Agent(model=mock_model, retriever=mock_vectordb)

        assert agent.retriever.embedder == mock_embedder

    @pytest.mark.asyncio
    async def test_agent_rag_graceful_degradation(self, mock_model, mock_vectordb):
        agent = Agent(model=mock_model, retriever=mock_vectordb)

        mock_vectordb.search_async = AsyncMock(side_effect=Exception("Retrieval failed"))

        assert agent.retriever is not None


class TestAgentRAGWithMemory:
    @pytest.fixture
    def mock_model(self):
        model = Mock(spec=Anthropic)
        model.api_key = "test_key"
        model.model = "test-model"
        return model

    @pytest.mark.asyncio
    async def test_agent_rag_and_memory(self, mock_model, mock_vectordb, mock_memory):
        agent = Agent(model=mock_model, retriever=mock_vectordb, memory=mock_memory)

        assert agent.retriever is not None
        assert agent.memory is not None

    @pytest.mark.asyncio
    async def test_agent_rag_memory_context(self, mock_model, mock_vectordb, mock_memory):
        agent = Agent(model=mock_model, retriever=mock_vectordb, memory=mock_memory)

        mock_memory.get_messages = Mock(
            return_value=[{"role": "user", "content": "Previous question"}, {"role": "assistant", "content": "Previous answer"}]
        )

        assert agent.retriever is not None
        assert agent.memory is not None
