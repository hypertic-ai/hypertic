"""
Mock utilities for testing.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock


def create_mock_llm_response(
    content: str = "Test response",
    model: str = "test-model",
    tokens: int = 100,
) -> dict[str, Any]:
    """Create a mock LLM response dictionary."""
    return {
        "content": content,
        "model": model,
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": tokens // 2,
            "completion_tokens": tokens // 2,
            "total_tokens": tokens,
        },
    }


def create_mock_embedding(dim: int = 100) -> list[float]:
    """Create a mock embedding vector of specified dimension."""
    return [0.1 * (i % 10) for i in range(dim)]


def create_mock_async_client() -> MagicMock:
    """Create a mock async client."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


def create_mock_sync_client() -> Mock:
    """Create a mock sync client."""
    return Mock()
