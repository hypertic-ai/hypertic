"""
Base guardrail interface for Hypertic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    violation_type: str | None = None
    details: dict[str, Any] | None = None
    modified_input: str | None = None


class BaseGuardrail(ABC):
    @abstractmethod
    def validate_input(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> GuardrailResult:
        pass
