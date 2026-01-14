from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, model_validator

from hypertic.models.metrics import Metrics


class LLMResponse(BaseModel):
    response_text: str
    metrics: Metrics
    model: str = "unknown"
    finish_reason: str = "stop"
    params: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[Any] = Field(default_factory=list)
    tool_outputs: dict[str, Any] = Field(default_factory=dict)
    additional_metadata: dict[str, Any] = Field(default_factory=dict)

    content: str = Field(init=False, default="")
    metadata: dict[str, Any] = Field(init=False, default_factory=dict)
    reasoning: str | None = Field(init=False, default=None)
    structured_output: Any | None = Field(init=False, default=None)

    @model_validator(mode="after")
    def _compute_fields(self) -> "LLMResponse":
        object.__setattr__(self, "content", self.response_text)

        metadata = {
            "model": self.model,
            "params": self.params,
            "finish_reason": self.finish_reason,
            "input_tokens": self.metrics.input_tokens,
            "output_tokens": self.metrics.output_tokens,
        }

        metadata.update(self.additional_metadata)
        object.__setattr__(self, "metadata", metadata)

        reasoning = metadata.get("reasoning_content")
        object.__setattr__(self, "reasoning", reasoning)

        return self

    def __str__(self) -> str:
        return f"{{'content': '{self.content}', 'metadata': {self.metadata}, 'tool_calls': {self.tool_calls}, 'tool_outputs': {self.tool_outputs}}}"

    def __repr__(self) -> str:
        return f"{{'content': '{self.content}', 'metadata': {self.metadata}, 'tool_calls': {self.tool_calls}, 'tool_outputs': {self.tool_outputs}}}"


@dataclass
class Base(ABC):
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None

    supports_mcp: bool = field(default=False, init=False)
    mcp_servers: Any = field(default=None, init=False)
    _cumulative_metrics: Metrics = field(default_factory=Metrics, init=False)

    def __post_init__(self) -> None:
        return None

    def set_mcp_servers(self, mcp_servers):
        self.mcp_servers = mcp_servers
        self.supports_mcp = True

    def _accumulate_metrics(self, usage: dict[str, Any]):
        metrics = Metrics.from_api_response(usage)
        self._cumulative_metrics += metrics

    def _get_cumulative_metrics(self) -> Metrics:
        return self._cumulative_metrics

    def _reset_metrics(self):
        self._cumulative_metrics = Metrics()

    def _create_streaming_metadata(
        self,
        finish_reason: str = "stop",
        params: dict[str, Any] | None = None,
        additional_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if params is None:
            params = {}
            if self.temperature is not None:
                params["temperature"] = self.temperature
            if self.top_p is not None:
                params["top_p"] = self.top_p
            if self.presence_penalty is not None:
                params["presence_penalty"] = self.presence_penalty
            if self.frequency_penalty is not None:
                params["frequency_penalty"] = self.frequency_penalty
            if self.max_tokens is not None:
                params["max_tokens"] = self.max_tokens

        cumulative_metrics = self._get_cumulative_metrics()

        metadata = {
            "model": self.model,
            "params": params,
            "finish_reason": finish_reason,
            "input_tokens": cumulative_metrics.input_tokens,
            "output_tokens": cumulative_metrics.output_tokens,
        }

        if additional_metrics:
            metadata.update(additional_metrics)

        return metadata

    @abstractmethod
    async def ahandle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
    ) -> Any | None:
        pass

    @abstractmethod
    async def ahandle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
    ):
        pass

    @abstractmethod
    def handle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
    ) -> Any | None:
        pass

    @abstractmethod
    def handle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
    ):
        pass

    def _execute_tool_sync(self, tool_executor: Any, tool_name: str, arguments: dict[str, Any]) -> str:
        result: str = str(tool_executor._execute_tool(tool_name, arguments))
        return result

    def _execute_tools_parallel_sync(self, tool_executor: Any, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
        result: dict[str, str] = dict(tool_executor._execute_tools_parallel(tool_calls))
        return result

    async def _execute_tool_async(self, tool_executor: Any, tool_name: str, arguments: dict[str, Any]) -> str:
        result: str = str(await tool_executor._aexecute_tool(tool_name, arguments))
        return result

    async def _execute_tools_parallel_async(self, tool_executor: Any, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
        result: dict[str, str] = dict(await tool_executor._aexecute_tools_parallel(tool_calls))
        return result

    def _create_llm_response(
        self,
        content: str,
        usage: dict[str, Any],
        tool_calls: list[Any] | None = None,
        tool_outputs: dict[str, Any] | None = None,
    ) -> LLMResponse:
        cumulative_metrics = self._get_cumulative_metrics()

        model: str = str(usage.get("model", self.model) or "unknown")
        finish_reason: str = str(usage.get("finish_reason", "stop"))
        params: dict[str, Any] = dict(usage.get("params", {}))

        additional_metadata: dict[str, Any] = {}
        for key, value in usage.items():
            if key not in [
                "input_tokens",
                "output_tokens",
                "model",
                "finish_reason",
                "params",
            ]:
                additional_metadata[key] = value

        return LLMResponse(
            response_text=content,
            metrics=cumulative_metrics,
            model=model,
            finish_reason=finish_reason,
            params=params,
            tool_calls=tool_calls or [],
            tool_outputs=tool_outputs or {},
            additional_metadata=additional_metadata,
        )
