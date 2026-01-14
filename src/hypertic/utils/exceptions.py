from typing import Any


class HyperticError(Exception):
    """Base exception for all Hypertic errors."""

    pass


class RetrieverError(HyperticError):
    """Error occurred during knowledge retrieval from vector database."""

    pass


class ToolExecutionError(HyperticError):
    """Error occurred while executing a tool."""

    pass


class ToolNotFoundError(HyperticError):
    """Tool was not found."""

    pass


class APIError(HyperticError):
    """Base exception for API-related errors."""

    pass


class RateLimitError(APIError):
    """API rate limit exceeded."""

    pass


class AuthenticationError(APIError):
    """API authentication failed (invalid API key, etc.)."""

    pass


class ConnectionError(APIError):
    """Connection error when calling API."""

    pass


class ValidationError(HyperticError):
    """Input validation error."""

    pass


class SchemaConversionError(HyperticError):
    """Error converting schema for structured output."""

    pass


class ConfigurationError(HyperticError):
    """Configuration error (invalid settings, etc.)."""

    pass


class MaxStepsError(HyperticError):
    """Maximum number of agent steps reached."""

    pass


class GuardrailViolationError(HyperticError):
    """Guardrail violation detected - operation blocked."""

    def __init__(self, reason: str, violation_type: str | None = None, details: dict[str, Any] | None = None):
        """Initialize guardrail violation error.

        Args:
            reason: Human-readable reason for the violation
            violation_type: Type of violation (e.g., 'pii', 'content_moderation', 'tool_safety')
            details: Additional details about the violation
        """
        super().__init__(reason)
        self.reason = reason
        self.violation_type = violation_type
        self.details = details or {}
