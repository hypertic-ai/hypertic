"""
Guardrail: Comprehensive guardrail that handles PII and content moderation.
"""

import hashlib
import ipaddress
import re
from collections.abc import Callable
from os import getenv
from re import Pattern

from hypertic.guardrails.base import BaseGuardrail, GuardrailResult
from hypertic.utils.log import get_logger

logger = get_logger(__name__)


def _luhn_check(card_number: str) -> bool:
    digits = re.sub(r"\D", "", card_number)
    if len(digits) < 13 or len(digits) > 19:
        return False

    total = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n

    return total % 10 == 0


class Guardrail(BaseGuardrail):
    """Comprehensive guardrail that handles PII and content moderation.

    PII detection supports: email, credit_card (Luhn validated), ssn, phone,
    ip_address (validated), url. Custom detectors can be added.

    Strategies: "block" (raise error), "redact" ([REDACTED]), "mask" (****1234),
    """

    def __init__(
        self,
        # PII strategies
        email: str | None = None,
        credit_card: str | None = None,
        ssn: str | None = None,
        phone: str | None = None,
        ip_address: str | None = None,
        url: str | None = None,
        # Custom PII patterns: Dict mapping PII type name to compiled regex Pattern or (detector_func, strategy) tuple
        custom_patterns: dict[str, Pattern[str] | tuple[Callable[[str], list[str]], str]] | None = None,
        # Content moderation
        block_toxic: bool = False,
        api_key: str | None = None,
    ):
        self.email = email
        self.credit_card = credit_card
        self.ssn = ssn
        self.phone = phone
        self.ip_address = ip_address
        self.url = url
        self.block_toxic = block_toxic
        self.moderation_model = "omni-moderation-latest"
        self.api_key = api_key or getenv("OPENAI_API_KEY")

        self.pii_patterns: dict[str, Pattern[str]] = {}

        if email:
            self.pii_patterns["email"] = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE)

        if credit_card:
            self.pii_patterns["credit_card"] = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")

        if ssn:
            self.pii_patterns["ssn"] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

        if phone:
            self.pii_patterns["phone"] = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

        if ip_address:
            self.pii_patterns["ip_address"] = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b")

        if url:
            self.pii_patterns["url"] = re.compile(
                r'https?://[^\s<>"{}|\\^`\[\]]+|(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}[^\s<>"{}|\\^`\[\]]*'
            )

        self.custom_detector_patterns = {}

        if custom_patterns:
            for pii_type, pattern_info in custom_patterns.items():
                if isinstance(pattern_info, Pattern):
                    self.pii_patterns[pii_type] = pattern_info
                elif isinstance(pattern_info, tuple):
                    self.custom_detector_patterns[pii_type] = pattern_info

    def validate_input(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> GuardrailResult:
        pii_result = self._check_pii(query)
        if not pii_result.allowed:
            return pii_result

        if self.block_toxic:
            toxic_result = self._check_toxic_content(query)
            if not toxic_result.allowed:
                return toxic_result

        if pii_result.modified_input:
            return GuardrailResult(allowed=True, modified_input=pii_result.modified_input)

        return GuardrailResult(allowed=True)

    def _check_pii(self, text: str) -> GuardrailResult:
        detected_pii = []
        modified_text = text

        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                strategy = getattr(self, pii_type, None)

                if strategy:
                    if pii_type == "credit_card":
                        matches = [m for m in matches if _luhn_check(m)]
                    elif pii_type == "ip_address":
                        valid_matches = []
                        for m in matches:
                            try:
                                ipaddress.ip_address(m)
                                valid_matches.append(m)
                            except ValueError:
                                continue
                        matches = valid_matches

                    if matches:
                        detected_pii.append((pii_type, matches, strategy))
                        modified_text = self._apply_strategy(modified_text, matches, strategy, pii_type, "")

        for pii_type, (detector_func, strategy) in self.custom_detector_patterns.items():
            matches = detector_func(text)
            if matches:
                detected_pii.append((pii_type, matches, strategy))
                modified_text = self._apply_strategy(modified_text, matches, strategy, pii_type, f"[{pii_type.upper()}]")

        if detected_pii:
            for pii_info in detected_pii:
                if len(pii_info) == 3:
                    pii_type, values, strategy = pii_info
                else:
                    pii_type, values = pii_info
                    strategy = getattr(self, pii_type, None)

                if strategy == "block":
                    return GuardrailResult(
                        allowed=False,
                        reason=f"PII detected: {pii_type}",
                        violation_type="pii",
                        details={
                            "pii_type": pii_type,
                            "values": values[:3],
                        },
                    )

        if detected_pii and modified_text != text:
            return GuardrailResult(allowed=True, modified_input=modified_text)

        return GuardrailResult(allowed=True)

    def _apply_strategy(self, text: str, matches: list[str], strategy: str, pii_type: str, default_replacement: str) -> str:
        if strategy == "block":
            return text

        modified = text
        for match in matches:
            if strategy == "redact":
                modified = modified.replace(match, f"[REDACTED_{pii_type.upper()}]")
            elif strategy == "mask":
                masked = self._mask_pii(match, pii_type)
                modified = modified.replace(match, masked)
            elif strategy == "hash":
                hash_digest = hashlib.sha256(match.encode()).hexdigest()[:8]
                modified = modified.replace(match, f"<{pii_type}_hash:{hash_digest}>")

        return modified

    def _mask_pii(self, value: str, pii_type: str) -> str:
        digits_only = re.sub(r"\D", "", value)

        if pii_type == "credit_card":
            if len(digits_only) >= 4:
                last_4 = digits_only[-4:]
                return f"****-****-****-{last_4}"
            return "****-****-****-****"
        elif pii_type == "ssn":
            if len(digits_only) >= 4:
                last_4 = digits_only[-4:]
                return f"***-**-{last_4}"
            return "***-**-****"
        elif pii_type == "phone":
            if len(digits_only) >= 4:
                last_4 = digits_only[-4:]
                return f"(***) ***-{last_4}"
            return "(***) ***-****"
        elif pii_type == "email":
            if "@" in value:
                local, domain = value.split("@", 1)
                masked_local = local[0] + "*" * (len(local) - 1) if len(local) > 1 else "*"
                return f"{masked_local}@{domain}"
            return "*" * len(value)
        else:
            if len(value) > 4:
                show_count = max(1, len(value) // 4)
                return "*" * (len(value) - show_count) + value[-show_count:]
            return "*" * len(value)

    def _check_toxic_content(self, text: str) -> GuardrailResult:
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError("`openai` not installed. Please install using `pip install openai`") from err

        if not self.api_key:
            logger.warning("OpenAI API key not found. Skipping moderation check.")
            return GuardrailResult(allowed=True)

        try:
            if self.api_key is not None:
                client = OpenAI(api_key=self.api_key)
            else:
                client = OpenAI()
            response = client.moderations.create(model=self.moderation_model, input=text)
            result = response.results[0]

            if result.flagged:
                categories = result.categories.model_dump()
                category_scores = result.category_scores.model_dump()

                flagged_categories = [cat for cat, flagged in categories.items() if flagged]

                return GuardrailResult(
                    allowed=False,
                    reason=f"Content moderation violation detected: {', '.join(flagged_categories[:3])}",
                    violation_type="content_moderation",
                    details={
                        "categories": categories,
                        "category_scores": category_scores,
                        "flagged_categories": flagged_categories,
                    },
                )

            return GuardrailResult(allowed=True)
        except Exception as e:
            logger.error(f"Error checking content moderation: {e}", exc_info=True)
            return GuardrailResult(allowed=True)
