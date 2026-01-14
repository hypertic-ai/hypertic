from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Metrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    provider_metrics: dict[str, Any] | None = None

    additional_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        metrics_dict = asdict(self)
        metrics_dict = {
            k: v
            for k, v in metrics_dict.items()
            if v is not None and (not isinstance(v, int | float) or v != 0) and (not isinstance(v, dict) or len(v) > 0)
        }
        return metrics_dict

    def __add__(self, other: "Metrics") -> "Metrics":
        if not isinstance(other, Metrics):
            return self

        result_class = type(self)
        result = result_class(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

        if self.provider_metrics or other.provider_metrics:
            result.provider_metrics = {}
            if self.provider_metrics:
                result.provider_metrics.update(self.provider_metrics)
            if other.provider_metrics:
                result.provider_metrics.update(other.provider_metrics)

        if self.additional_metrics or other.additional_metrics:
            result.additional_metrics = {}
            if self.additional_metrics:
                result.additional_metrics.update(self.additional_metrics)
            if other.additional_metrics:
                result.additional_metrics.update(other.additional_metrics)

        return result

    def __radd__(self, other: Any) -> "Metrics":
        if isinstance(other, int) and other == 0:
            return self
        if isinstance(other, Metrics):
            return self + other
        return self

    def __iadd__(self, other: "Metrics") -> "Metrics":
        if not isinstance(other, Metrics):
            return self

        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens

        if other.provider_metrics:
            if self.provider_metrics is None:
                self.provider_metrics = {}
            self.provider_metrics.update(other.provider_metrics)

        if other.additional_metrics:
            if self.additional_metrics is None:
                self.additional_metrics = {}
            self.additional_metrics.update(other.additional_metrics)

        return self

    @classmethod
    def from_api_response(cls, usage: dict[str, Any], cost_per_token: dict[str, float] | None = None) -> "Metrics":
        input_tokens = usage.get(
            "input_tokens",
            usage.get("prompt_tokens", usage.get("inputTokens", usage.get("promptTokenCount", 0))),
        )
        output_tokens = usage.get(
            "output_tokens",
            usage.get("completion_tokens", usage.get("outputTokens", usage.get("candidatesTokenCount", 0))),
        )

        if not input_tokens and not output_tokens:
            billed_units = usage.get("billed_units", {})
            if billed_units:
                input_tokens = billed_units.get("input_tokens", 0)
                output_tokens = billed_units.get("output_tokens", 0)
            else:
                tokens = usage.get("tokens", {})
                input_tokens = tokens.get("input_tokens", 0)
                output_tokens = tokens.get("output_tokens", 0)

        total_tokens = input_tokens + output_tokens

        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            provider_metrics=usage,
        )
