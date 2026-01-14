from unittest.mock import MagicMock, patch

import pytest

from hypertic.guardrails.base import BaseGuardrail, GuardrailResult
from hypertic.guardrails.guardrails import Guardrail, _luhn_check


class TestLuhnCheck:
    def test_luhn_check_valid_card(self):
        """Test _luhn_check with valid credit card."""
        # Valid test card number
        assert _luhn_check("4532015112830366") is True

    def test_luhn_check_invalid_card(self):
        """Test _luhn_check with invalid credit card."""
        assert _luhn_check("4532015112830367") is False

    def test_luhn_check_too_short(self):
        """Test _luhn_check with too short number."""
        assert _luhn_check("123456") is False

    def test_luhn_check_with_dashes(self):
        """Test _luhn_check with formatted card number."""
        assert _luhn_check("4532-0151-1283-0366") is True


class TestBaseGuardrail:
    def test_base_guardrail_is_abstract(self):
        with pytest.raises(TypeError):
            BaseGuardrail()  # type: ignore[abstract]


class TestGuardrail:
    def test_guardrail_creation(self):
        guardrail = Guardrail()
        assert guardrail.email is None
        assert guardrail.credit_card is None
        assert guardrail.block_toxic is False

    def test_guardrail_creation_with_pii_strategies(self):
        """Test Guardrail with PII strategies."""
        guardrail = Guardrail(email="block", credit_card="redact", ssn="mask")
        assert guardrail.email == "block"
        assert guardrail.credit_card == "redact"
        assert guardrail.ssn == "mask"
        assert "email" in guardrail.pii_patterns
        assert "credit_card" in guardrail.pii_patterns
        assert "ssn" in guardrail.pii_patterns

    @patch("hypertic.guardrails.guardrails.getenv")
    def test_guardrail_with_env_api_key(self, mock_getenv):
        """Test Guardrail with API key from environment."""
        mock_getenv.return_value = "env_key"
        guardrail = Guardrail(block_toxic=True)
        assert guardrail.api_key == "env_key"

    def test_guardrail_with_custom_patterns_regex(self):
        """Test Guardrail with custom regex patterns."""
        import re

        custom_pattern = re.compile(r"\b\d{4}\b")
        guardrail = Guardrail(custom_patterns={"custom": custom_pattern})
        assert "custom" in guardrail.pii_patterns

    def test_guardrail_with_custom_patterns_detector(self):
        """Test Guardrail with custom detector function."""

        def detector(text: str) -> list[str]:
            return ["found"] if "secret" in text else []

        guardrail = Guardrail(custom_patterns={"secret": (detector, "block")})
        assert "secret" in guardrail.custom_detector_patterns

    def test_guardrail_validate_input_safe(self):
        guardrail = Guardrail()
        result = guardrail.validate_input("This is safe content")
        assert isinstance(result, GuardrailResult)
        assert result.allowed is True

    def test_guardrail_validate_input_email_block(self):
        guardrail = Guardrail(email="block")
        result = guardrail.validate_input("Contact me at test@example.com")
        assert isinstance(result, GuardrailResult)
        assert result.allowed is False
        assert result.violation_type == "pii"
        assert "email" in result.reason.lower() if result.reason else False

    def test_guardrail_validate_input_email_redact(self):
        guardrail = Guardrail(email="redact")
        result = guardrail.validate_input("Contact me at test@example.com")
        assert isinstance(result, GuardrailResult)
        assert result.allowed is True
        assert result.modified_input is not None
        assert "test@example.com" not in result.modified_input
        assert "[REDACTED" in result.modified_input or "REDACTED" in result.modified_input

    def test_guardrail_validate_input_email_mask(self):
        """Test email masking strategy."""
        guardrail = Guardrail(email="mask")
        result = guardrail.validate_input("Contact me at test@example.com")
        assert result.allowed is True
        assert result.modified_input is not None
        assert "test@example.com" not in result.modified_input
        assert "@example.com" in result.modified_input

    def test_guardrail_validate_input_email_hash(self):
        """Test email hashing strategy."""
        guardrail = Guardrail(email="hash")
        result = guardrail.validate_input("Contact me at test@example.com")
        assert result.allowed is True
        assert result.modified_input is not None
        assert "test@example.com" not in result.modified_input
        assert "email_hash:" in result.modified_input

    def test_guardrail_validate_input_credit_card_block(self):
        """Test credit card blocking."""
        guardrail = Guardrail(credit_card="block")
        # Valid Luhn card number
        result = guardrail.validate_input("My card is 4532015112830366")
        assert result.allowed is False
        assert result.violation_type == "pii"

    def test_guardrail_validate_input_credit_card_redact(self):
        """Test credit card redaction."""
        guardrail = Guardrail(credit_card="redact")
        result = guardrail.validate_input("My card is 4532015112830366")
        assert result.allowed is True
        assert result.modified_input is not None
        assert "4532015112830366" not in result.modified_input

    def test_guardrail_validate_input_credit_card_mask(self):
        """Test credit card masking."""
        guardrail = Guardrail(credit_card="mask")
        result = guardrail.validate_input("My card is 4532015112830366")
        assert result.allowed is True
        assert result.modified_input is not None
        assert "****-****-****-0366" in result.modified_input

    def test_guardrail_validate_input_ssn_block(self):
        """Test SSN blocking."""
        guardrail = Guardrail(ssn="block")
        result = guardrail.validate_input("My SSN is 123-45-6789")
        assert result.allowed is False
        assert result.violation_type == "pii"

    def test_guardrail_validate_input_ssn_mask(self):
        """Test SSN masking."""
        guardrail = Guardrail(ssn="mask")
        result = guardrail.validate_input("My SSN is 123-45-6789")
        assert result.allowed is True
        assert result.modified_input is not None
        assert "***-**-6789" in result.modified_input

    def test_guardrail_validate_input_phone_block(self):
        """Test phone blocking."""
        guardrail = Guardrail(phone="block")
        result = guardrail.validate_input("Call me at 555-123-4567")
        assert result.allowed is False
        assert result.violation_type == "pii"

    def test_guardrail_validate_input_phone_mask(self):
        """Test phone masking."""
        guardrail = Guardrail(phone="mask")
        result = guardrail.validate_input("Call me at 555-123-4567")
        assert result.allowed is True
        assert result.modified_input is not None
        assert "(***) ***-4567" in result.modified_input

    def test_guardrail_validate_input_ip_address_block(self):
        """Test IP address blocking."""
        guardrail = Guardrail(ip_address="block")
        result = guardrail.validate_input("Server at 192.168.1.1")
        assert result.allowed is False
        assert result.violation_type == "pii"

    def test_guardrail_validate_input_ip_address_invalid(self):
        """Test IP address validation filters invalid IPs."""
        guardrail = Guardrail(ip_address="block")
        # 999.999.999.999 is not a valid IP
        result = guardrail.validate_input("Invalid IP 999.999.999.999")
        assert result.allowed is True  # Should not block invalid IPs

    def test_guardrail_validate_input_url_block(self):
        """Test URL blocking."""
        guardrail = Guardrail(url="block")
        result = guardrail.validate_input("Visit https://example.com")
        assert result.allowed is False
        assert result.violation_type == "pii"

    def test_guardrail_validate_input_url_redact(self):
        """Test URL redaction."""
        guardrail = Guardrail(url="redact")
        result = guardrail.validate_input("Visit https://example.com")
        assert result.allowed is True
        assert result.modified_input is not None
        assert "https://example.com" not in result.modified_input

    def test_guardrail_validate_input_multiple_pii(self):
        """Test multiple PII types."""
        guardrail = Guardrail(email="redact", phone="mask")
        result = guardrail.validate_input("Contact test@example.com or call 555-123-4567")
        assert result.allowed is True
        assert result.modified_input is not None
        assert "test@example.com" not in result.modified_input
        assert "555-123-4567" not in result.modified_input

    def test_guardrail_validate_input_custom_detector(self):
        """Test custom detector function."""

        def detector(text: str) -> list[str]:
            return ["secret"] if "secret" in text else []

        guardrail = Guardrail(custom_patterns={"secret": (detector, "block")})
        result = guardrail.validate_input("This is a secret message")
        assert result.allowed is False
        assert result.violation_type == "pii"

    def test_guardrail_validate_input_custom_detector_redact(self):
        """Test custom detector with redact strategy."""

        def detector(text: str) -> list[str]:
            return ["secret"] if "secret" in text else []

        guardrail = Guardrail(custom_patterns={"secret": (detector, "redact")})
        result = guardrail.validate_input("This is a secret message")
        assert result.allowed is True
        assert result.modified_input is not None
        assert "secret" not in result.modified_input

    def test_guardrail_check_toxic_content_no_api_key(self):
        """Test toxic content check without API key."""
        guardrail = Guardrail(block_toxic=True, api_key=None)
        with patch("hypertic.guardrails.guardrails.getenv", return_value=None):
            result = guardrail._check_toxic_content("test content")
            assert result.allowed is True

    @patch("openai.OpenAI")
    def test_guardrail_check_toxic_content_flagged(self, mock_openai_class):
        """Test toxic content check with flagged content."""
        guardrail = Guardrail(block_toxic=True, api_key="test_key")
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.flagged = True
        mock_categories = MagicMock()
        mock_categories.model_dump.return_value = {"hate": True, "harassment": False}
        mock_category_scores = MagicMock()
        mock_category_scores.model_dump.return_value = {"hate": 0.9, "harassment": 0.1}
        mock_result.categories = mock_categories
        mock_result.category_scores = mock_category_scores
        mock_response = MagicMock()
        mock_response.results = [mock_result]
        mock_client.moderations.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        result = guardrail._check_toxic_content("toxic content")
        assert result.allowed is False
        assert result.violation_type == "content_moderation"

    @patch("openai.OpenAI")
    def test_guardrail_check_toxic_content_not_flagged(self, mock_openai_class):
        """Test toxic content check with safe content."""
        guardrail = Guardrail(block_toxic=True, api_key="test_key")
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.flagged = False
        mock_response = MagicMock()
        mock_response.results = [mock_result]
        mock_client.moderations.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        result = guardrail._check_toxic_content("safe content")
        assert result.allowed is True

    @patch("openai.OpenAI")
    def test_guardrail_check_toxic_content_error(self, mock_openai_class):
        """Test toxic content check handles errors."""
        guardrail = Guardrail(block_toxic=True, api_key="test_key")
        mock_client = MagicMock()
        mock_client.moderations.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        result = guardrail._check_toxic_content("test content")
        assert result.allowed is True  # Should allow on error

    def test_guardrail_validate_input_with_toxic_check(self):
        """Test validate_input with toxic content check."""
        guardrail = Guardrail(block_toxic=True, api_key="test_key")
        with patch.object(guardrail, "_check_toxic_content", return_value=GuardrailResult(allowed=False, reason="Toxic")):
            result = guardrail.validate_input("test")
            assert result.allowed is False

    def test_guardrail_mask_pii_credit_card(self):
        """Test _mask_pii for credit card."""
        guardrail = Guardrail()
        masked = guardrail._mask_pii("4532-0151-1283-0366", "credit_card")
        assert masked == "****-****-****-0366"

    def test_guardrail_mask_pii_ssn(self):
        """Test _mask_pii for SSN."""
        guardrail = Guardrail()
        masked = guardrail._mask_pii("123-45-6789", "ssn")
        assert masked == "***-**-6789"

    def test_guardrail_mask_pii_phone(self):
        """Test _mask_pii for phone."""
        guardrail = Guardrail()
        masked = guardrail._mask_pii("555-123-4567", "phone")
        assert masked == "(***) ***-4567"

    def test_guardrail_mask_pii_email(self):
        """Test _mask_pii for email."""
        guardrail = Guardrail()
        masked = guardrail._mask_pii("test@example.com", "email")
        assert masked == "t***@example.com"

    def test_guardrail_mask_pii_default(self):
        """Test _mask_pii default behavior."""
        guardrail = Guardrail()
        masked = guardrail._mask_pii("1234567890", "unknown")
        assert len(masked) == len("1234567890")
        # Should show last 25% of characters (2-3 chars for 10 char string)
        assert len(masked.replace("*", "")) >= 1

    def test_guardrail_apply_strategy_block(self):
        """Test _apply_strategy with block."""
        guardrail = Guardrail()
        result = guardrail._apply_strategy("text with match", ["match"], "block", "test", "")
        assert result == "text with match"

    def test_guardrail_apply_strategy_redact(self):
        """Test _apply_strategy with redact."""
        guardrail = Guardrail()
        result = guardrail._apply_strategy("text with match", ["match"], "redact", "test", "")
        assert "[REDACTED_TEST]" in result
        assert "match" not in result

    def test_guardrail_apply_strategy_mask(self):
        """Test _apply_strategy with mask."""
        guardrail = Guardrail()
        result = guardrail._apply_strategy("text with 123-45-6789", ["123-45-6789"], "mask", "ssn", "")
        assert "123-45-6789" not in result
        assert "***-**-6789" in result

    def test_guardrail_apply_strategy_hash(self):
        """Test _apply_strategy with hash."""
        guardrail = Guardrail()
        result = guardrail._apply_strategy("text with match", ["match"], "hash", "test", "")
        assert "match" not in result
        assert "test_hash:" in result

    def test_guardrail_validate_input_with_user_id(self):
        """Test validate_input with user_id."""
        guardrail = Guardrail()
        result = guardrail.validate_input("test", user_id="user1")
        assert result.allowed is True

    def test_guardrail_validate_input_with_session_id(self):
        """Test validate_input with session_id."""
        guardrail = Guardrail()
        result = guardrail.validate_input("test", session_id="session1")
        assert result.allowed is True
