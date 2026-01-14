"""
Shared pytest fixtures and configuration for Hypertic tests.
"""

import os
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")


@pytest.fixture
def mock_api_key() -> str:
    """Mock API key for testing."""
    return "test_api_key_12345"


@pytest.fixture
def mock_model_name() -> str:
    """Mock model name for testing."""
    return "test-model-v1"


@pytest.fixture
def sample_text() -> str:
    """Sample text for embedding/testing."""
    return "This is a sample text for testing purposes."


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample batch of texts for embedding/testing."""
    return [
        "First sample text.",
        "Second sample text.",
        "Third sample text.",
    ]


@pytest.fixture
def mock_llm_response() -> dict[str, Any]:
    """Mock LLM API response structure."""
    return {
        "content": "This is a test response",
        "model": "test-model",
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_tool_call() -> dict[str, Any]:
    """Mock tool call structure."""
    return {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "test_tool",
            "arguments": '{"param1": "value1"}',
        },
    }


@pytest.fixture
def mock_embedding() -> list[float]:
    """Mock embedding vector."""
    return [0.1, 0.2, 0.3, 0.4, 0.5] * 20  # 100-dim vector


@pytest.fixture
def mock_embeddings() -> list[list[float]]:
    """Mock batch of embedding vectors."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
        [0.2, 0.3, 0.4, 0.5, 0.6] * 20,
        [0.3, 0.4, 0.5, 0.6, 0.7] * 20,
    ]


@pytest.fixture
def mock_vector_document() -> dict[str, Any]:
    """Mock vector document structure."""
    return {
        "id": "doc_123",
        "content": "Sample document content",
        "metadata": {"source": "test", "page": 1},
        "embedding": [0.1, 0.2, 0.3] * 33,  # 99-dim, close to 100
    }


@pytest.fixture
def mock_memory_entry() -> dict[str, Any]:
    """Mock memory entry structure."""
    return {
        "role": "user",
        "content": "Hello, how are you?",
        "timestamp": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def mock_agent_config() -> dict[str, Any]:
    """Mock agent configuration."""
    return {
        "instructions": "You are a helpful assistant.",
        "max_steps": 5,
        "temperature": 0.7,
    }


@pytest.fixture
def mock_guardrail() -> Mock:
    """Mock guardrail instance."""
    guardrail = Mock()
    guardrail.check.return_value = True
    guardrail.check_async = AsyncMock(return_value=True)
    return guardrail


@pytest.fixture
def mock_tool() -> Mock:
    """Mock tool instance."""
    tool = Mock()
    tool.name = "test_tool"
    tool.description = "A test tool"
    tool.execute = Mock(return_value="Tool result")
    tool.execute_async = AsyncMock(return_value="Tool result")
    return tool


@pytest.fixture
def mock_vector_db() -> Mock:
    """Mock vector database instance."""
    db = Mock()
    db.add_documents = Mock()
    db.add_documents_async = AsyncMock()
    db.search = Mock(return_value=[])
    db.search_async = AsyncMock(return_value=[])
    return db


@pytest.fixture
def mock_memory() -> Mock:
    """Mock memory instance."""
    memory = Mock()
    memory.add = Mock()
    memory.add_async = AsyncMock()
    memory.get = Mock(return_value=[])
    memory.get_async = AsyncMock(return_value=[])
    return memory


@pytest.fixture
def mock_model_handler() -> Mock:
    """Mock model handler instance."""
    handler = Mock()
    handler.handle_non_streaming = Mock()
    handler.handle_streaming = Mock()
    handler.ahandle_non_streaming = AsyncMock()
    handler.ahandle_streaming = AsyncMock()
    return handler


@pytest.fixture
def env_vars(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Fixture to manage environment variables for tests."""
    # Store original values
    original_env = {}

    def set_env(key: str, value: str | None) -> None:
        original_env[key] = os.environ.get(key)
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    yield set_env

    # Restore original values
    for key, value in original_env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


@pytest.fixture
def temp_dir(tmp_path: pytest.TempPathFactory) -> str:
    """Create a temporary directory for file operations."""
    return str(tmp_path)


@pytest.fixture
def sample_file_content() -> str:
    """Sample file content for testing."""
    return "This is sample file content for testing.\nLine 2\nLine 3"


@pytest.fixture
def sample_json_data() -> dict[str, Any]:
    """Sample JSON data for testing."""
    return {
        "key1": "value1",
        "key2": 123,
        "key3": [1, 2, 3],
        "key4": {"nested": "data"},
    }


@pytest.fixture
def mock_embedder() -> Mock:
    """Mock embedder instance."""
    embedder = Mock()
    embedder.embed = AsyncMock(return_value=[0.1] * 100)
    embedder.embed_sync = Mock(return_value=[0.1] * 100)
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * 100] * 3)
    embedder.embed_batch_sync = Mock(return_value=[[0.1] * 100] * 3)
    embedder.initialize = AsyncMock(return_value=True)
    embedder.initialize_sync = Mock(return_value=True)
    return embedder
