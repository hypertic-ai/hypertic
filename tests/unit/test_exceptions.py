import pytest

from hypertic.utils.exceptions import (
    APIError,
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    GuardrailViolationError,
    HyperticError,
    MaxStepsError,
    RateLimitError,
    RetrieverError,
    SchemaConversionError,
    ToolExecutionError,
    ToolNotFoundError,
    ValidationError,
)


@pytest.mark.unit
class TestHyperticError:
    @pytest.mark.parametrize(
        "error_message",
        ["Test error", "Another error", "Error with details"],
    )
    def test_hypertic_error_creation(self, error_message):
        """Test HyperticError creation with various messages."""
        error = HyperticError(error_message)
        assert str(error) == error_message
        assert isinstance(error, Exception)


@pytest.mark.unit
class TestRetrieverError:
    def test_retriever_error_creation(self):
        """Test RetrieverError creation and inheritance."""
        error = RetrieverError("Retrieval failed")
        assert str(error) == "Retrieval failed"
        assert isinstance(error, HyperticError)


@pytest.mark.unit
class TestToolExecutionError:
    def test_tool_execution_error_creation(self):
        """Test ToolExecutionError creation and inheritance."""
        error = ToolExecutionError("Tool execution failed")
        assert str(error) == "Tool execution failed"
        assert isinstance(error, HyperticError)


@pytest.mark.unit
class TestToolNotFoundError:
    def test_tool_not_found_error_creation(self):
        """Test ToolNotFoundError creation and inheritance."""
        error = ToolNotFoundError("Tool not found")
        assert str(error) == "Tool not found"
        assert isinstance(error, HyperticError)


@pytest.mark.unit
class TestAPIError:
    def test_api_error_creation(self):
        """Test APIError creation and inheritance."""
        error = APIError("API call failed")
        assert str(error) == "API call failed"
        assert isinstance(error, HyperticError)


@pytest.mark.unit
class TestRateLimitError:
    def test_rate_limit_error_creation(self):
        """Test RateLimitError creation and inheritance chain."""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, APIError)
        assert isinstance(error, HyperticError)


@pytest.mark.unit
class TestAuthenticationError:
    def test_authentication_error_creation(self):
        """Test AuthenticationError creation and inheritance chain."""
        error = AuthenticationError("Invalid API key")
        assert str(error) == "Invalid API key"
        assert isinstance(error, APIError)
        assert isinstance(error, HyperticError)


@pytest.mark.unit
class TestConnectionError:
    def test_connection_error_creation(self):
        """Test ConnectionError creation and inheritance chain."""
        error = ConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, APIError)
        assert isinstance(error, HyperticError)


@pytest.mark.unit
class TestValidationError:
    def test_validation_error_creation(self):
        """Test ValidationError creation and inheritance."""
        error = ValidationError("Invalid input")
        assert str(error) == "Invalid input"
        assert isinstance(error, HyperticError)


@pytest.mark.unit
class TestSchemaConversionError:
    def test_schema_conversion_error_creation(self):
        """Test SchemaConversionError creation and inheritance."""
        error = SchemaConversionError("Schema conversion failed")
        assert str(error) == "Schema conversion failed"
        assert isinstance(error, HyperticError)


@pytest.mark.unit
class TestConfigurationError:
    def test_configuration_error_creation(self):
        """Test ConfigurationError creation and inheritance."""
        error = ConfigurationError("Invalid configuration")
        assert str(error) == "Invalid configuration"
        assert isinstance(error, HyperticError)


@pytest.mark.unit
class TestMaxStepsError:
    def test_max_steps_error_creation(self):
        """Test MaxStepsError creation and inheritance."""
        error = MaxStepsError("Max steps reached")
        assert str(error) == "Max steps reached"
        assert isinstance(error, HyperticError)


class TestGuardrailViolationError:
    def test_guardrail_violation_error_creation(self):
        error = GuardrailViolationError("Content blocked")
        assert str(error) == "Content blocked"
        assert error.reason == "Content blocked"
        assert error.violation_type is None
        assert error.details == {}
        assert isinstance(error, HyperticError)

    def test_guardrail_violation_error_with_type(self):
        error = GuardrailViolationError(
            "PII detected",
            violation_type="pii",
        )
        assert error.reason == "PII detected"
        assert error.violation_type == "pii"
        assert error.details == {}

    def test_guardrail_violation_error_with_details(self):
        details = {"field": "email", "value": "test@example.com"}
        error = GuardrailViolationError(
            "PII detected",
            violation_type="pii",
            details=details,
        )
        assert error.reason == "PII detected"
        assert error.violation_type == "pii"
        assert error.details == details
