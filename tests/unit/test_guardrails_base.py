import pytest

from hypertic.guardrails.base import BaseGuardrail, GuardrailResult


class TestGuardrailResult:
    def test_guardrail_result_creation(self):
        """Test GuardrailResult creation."""
        result = GuardrailResult(allowed=True)
        assert result.allowed is True
        assert result.reason is None
        assert result.violation_type is None
        assert result.details is None
        assert result.modified_input is None

    def test_guardrail_result_with_all_fields(self):
        """Test GuardrailResult with all fields."""
        result = GuardrailResult(
            allowed=False,
            reason="Contains prohibited content",
            violation_type="content_filter",
            details={"matched_pattern": "spam"},
            modified_input="Filtered content",
        )
        assert result.allowed is False
        assert result.reason == "Contains prohibited content"
        assert result.violation_type == "content_filter"
        assert result.details == {"matched_pattern": "spam"}
        assert result.modified_input == "Filtered content"

    def test_guardrail_result_allowed_true(self):
        """Test GuardrailResult with allowed=True."""
        result = GuardrailResult(allowed=True, reason="Valid input")
        assert result.allowed is True
        assert result.reason == "Valid input"

    def test_guardrail_result_allowed_false(self):
        """Test GuardrailResult with allowed=False."""
        result = GuardrailResult(allowed=False, reason="Invalid input")
        assert result.allowed is False
        assert result.reason == "Invalid input"

    @pytest.mark.parametrize(
        "violation_type",
        ["content_filter", "pii_detection", "toxicity", "custom"],
    )
    def test_guardrail_result_violation_types(self, violation_type):
        """Test GuardrailResult with various violation types."""
        result = GuardrailResult(allowed=False, violation_type=violation_type)
        assert result.violation_type == violation_type

    def test_guardrail_result_with_modified_input(self):
        """Test GuardrailResult with modified input."""
        result = GuardrailResult(
            allowed=True,
            modified_input="Sanitized version of input",
        )
        assert result.modified_input == "Sanitized version of input"


class TestBaseGuardrail:
    def test_base_guardrail_is_abstract(self):
        """Test that BaseGuardrail is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseGuardrail()  # type: ignore[abstract]

    def test_base_guardrail_has_required_methods(self):
        """Test that BaseGuardrail has required abstract methods."""
        assert hasattr(BaseGuardrail, "validate_input")

    def test_base_guardrail_can_be_subclassed(self):
        """Test that BaseGuardrail can be subclassed."""

        class ConcreteGuardrail(BaseGuardrail):
            def validate_input(
                self,
                query: str,
                user_id: str | None = None,
                session_id: str | None = None,
            ) -> GuardrailResult:
                return GuardrailResult(allowed=True)

        guardrail = ConcreteGuardrail()
        assert isinstance(guardrail, BaseGuardrail)
        result = guardrail.validate_input("test query")
        assert result.allowed is True

    def test_base_guardrail_validate_input_with_user_id(self):
        """Test validate_input with user_id."""

        class ConcreteGuardrail(BaseGuardrail):
            def validate_input(
                self,
                query: str,
                user_id: str | None = None,
                session_id: str | None = None,
            ) -> GuardrailResult:
                if user_id == "blocked_user":
                    return GuardrailResult(allowed=False, reason="User blocked")
                return GuardrailResult(allowed=True)

        guardrail = ConcreteGuardrail()
        result = guardrail.validate_input("test", user_id="blocked_user")
        assert result.allowed is False
        assert result.reason == "User blocked"

    def test_base_guardrail_validate_input_with_session_id(self):
        """Test validate_input with session_id."""

        class ConcreteGuardrail(BaseGuardrail):
            def validate_input(
                self,
                query: str,
                user_id: str | None = None,
                session_id: str | None = None,
            ) -> GuardrailResult:
                if session_id == "blocked_session":
                    return GuardrailResult(allowed=False, reason="Session blocked")
                return GuardrailResult(allowed=True)

        guardrail = ConcreteGuardrail()
        result = guardrail.validate_input("test", session_id="blocked_session")
        assert result.allowed is False

    def test_base_guardrail_validate_input_with_none_params(self):
        """Test validate_input with None parameters."""

        class ConcreteGuardrail(BaseGuardrail):
            def validate_input(
                self,
                query: str,
                user_id: str | None = None,
                session_id: str | None = None,
            ) -> GuardrailResult:
                return GuardrailResult(allowed=True)

        guardrail = ConcreteGuardrail()
        result = guardrail.validate_input("test", user_id=None, session_id=None)
        assert result.allowed is True
