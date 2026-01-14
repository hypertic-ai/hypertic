from typing import Any, Literal

from pydantic import BaseModel


class ContentEvent(BaseModel):
    type: Literal["content"] = "content"
    content: str


class ToolCallsEvent(BaseModel):
    type: Literal["tool_calls"] = "tool_calls"
    tool_calls: list[dict[str, Any]]


class ToolOutputsEvent(BaseModel):
    type: Literal["tool_outputs"] = "tool_outputs"
    tool_outputs: dict[str, Any]


class MetadataEvent(BaseModel):
    type: Literal["metadata"] = "metadata"
    metadata: dict[str, Any]


class ReasoningEvent(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    reasoning: str


class FinishReasonEvent(BaseModel):
    type: Literal["finish_reason"] = "finish_reason"
    finish_reason: str


class ResponseCompletedEvent(BaseModel):
    type: Literal["response_completed"] = "response_completed"
    usage: dict[str, Any]


class ResponseCreatedEvent(BaseModel):
    type: Literal["response_created"] = "response_created"
    response_id: str


class StructuredOutputEvent(BaseModel):
    type: Literal["structured_output"] = "structured_output"
    structured_output: Any


StreamEvent = (
    ContentEvent
    | ToolCallsEvent
    | ToolOutputsEvent
    | MetadataEvent
    | ReasoningEvent
    | FinishReasonEvent
    | ResponseCompletedEvent
    | ResponseCreatedEvent
    | StructuredOutputEvent
)
