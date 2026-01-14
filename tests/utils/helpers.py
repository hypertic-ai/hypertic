"""
Helper utilities for testing.
"""

import asyncio
from collections.abc import Coroutine
from typing import Any


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async function synchronously in tests."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def assert_dict_subset(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Assert that actual dict contains all keys from expected dict with matching values."""
    for key, value in expected.items():
        assert key in actual, f"Key '{key}' not found in actual dict"
        assert actual[key] == value, f"Value mismatch for key '{key}': expected {value}, got {actual[key]}"


def assert_embedding_valid(embedding: list[float], expected_dim: int | None = None) -> None:
    """Assert that an embedding vector is valid."""
    assert isinstance(embedding, list), "Embedding must be a list"
    assert len(embedding) > 0, "Embedding must not be empty"
    assert all(isinstance(x, (int, float)) for x in embedding), "Embedding must contain only numbers"
    if expected_dim is not None:
        assert len(embedding) == expected_dim, f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}"


def assert_metrics_valid(metrics: Any) -> None:
    """Assert that metrics object is valid."""
    assert hasattr(metrics, "input_tokens"), "Metrics must have input_tokens"
    assert hasattr(metrics, "output_tokens"), "Metrics must have output_tokens"
    assert isinstance(metrics.input_tokens, int), "input_tokens must be int"
    assert isinstance(metrics.output_tokens, int), "output_tokens must be int"
    assert metrics.input_tokens >= 0, "input_tokens must be non-negative"
    assert metrics.output_tokens >= 0, "output_tokens must be non-negative"
