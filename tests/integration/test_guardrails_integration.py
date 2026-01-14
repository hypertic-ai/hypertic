from hypertic.guardrails import Guardrail
from hypertic.guardrails.base import GuardrailResult


class TestGuardrailsIntegration:
    def test_guardrail_pii_detection_email(self):
        guardrail = Guardrail(email="block")

        result = guardrail.validate_input("Contact me at test@example.com")
        assert isinstance(result, GuardrailResult)
        assert result.allowed is False
        assert result.violation_type == "pii"

    def test_guardrail_pii_redaction_email(self):
        guardrail = Guardrail(email="redact")

        result = guardrail.validate_input("Contact me at test@example.com")
        assert isinstance(result, GuardrailResult)
        assert result.allowed is True
        assert result.modified_input is not None
        assert "test@example.com" not in result.modified_input

    def test_guardrail_credit_card_detection(self):
        guardrail = Guardrail(credit_card="block")

        result = guardrail.validate_input("My card is 4532-1234-5678-9010")
        assert isinstance(result, GuardrailResult)
        assert isinstance(result.allowed, bool)

    def test_guardrail_phone_detection(self):
        guardrail = Guardrail(phone="block")

        result = guardrail.validate_input("Call me at 555-123-4567")
        assert isinstance(result, GuardrailResult)
        assert isinstance(result.allowed, bool)

    def test_guardrail_ssn_detection(self):
        guardrail = Guardrail(ssn="block")

        result = guardrail.validate_input("SSN: 123-45-6789")
        assert isinstance(result, GuardrailResult)
        assert isinstance(result.allowed, bool)

    def test_guardrail_multiple_pii_types(self):
        guardrail = Guardrail(
            email="redact",
            phone="block",
            credit_card="block",
        )

        result = guardrail.validate_input("Email: test@example.com, Phone: 555-1234")
        assert isinstance(result, GuardrailResult)
        assert isinstance(result.allowed, bool)

    def test_guardrail_safe_content(self):
        guardrail = Guardrail(email="block", phone="block")

        result = guardrail.validate_input("This is safe content without any PII")
        assert isinstance(result, GuardrailResult)
        assert result.allowed is True

    def test_guardrail_custom_patterns(self):
        import re

        guardrail = Guardrail(
            custom_patterns={
                "customer_id": re.compile(r"CUST-\d{6}"),
            },
            email="redact",
        )

        result = guardrail.validate_input("Customer ID: CUST-123456")
        assert isinstance(result, GuardrailResult)
        assert isinstance(result.allowed, bool)


class TestGuardrailsWithAgent:
    def test_agent_guardrail_input_validation(self):
        from hypertic.agents import Agent
        from hypertic.models.anthropic import Anthropic

        guardrail = Guardrail(email="block")
        model = Anthropic(api_key="test", model="test-model")
        agent = Agent(model=model, guardrails=[guardrail])

        assert len(agent.guardrails) == 1
