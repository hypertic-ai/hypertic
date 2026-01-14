import pytest

from hypertic.models.events import (
    ContentEvent,
    FinishReasonEvent,
    MetadataEvent,
    ReasoningEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    StructuredOutputEvent,
    ToolCallsEvent,
    ToolOutputsEvent,
)


class TestContentEvent:
    def test_content_event_creation(self):
        """Test ContentEvent creation."""
        event = ContentEvent(content="Test content")
        assert event.type == "content"
        assert event.content == "Test content"

    def test_content_event_type_is_fixed(self):
        """Test that ContentEvent type cannot be changed."""
        event = ContentEvent(content="Test")
        assert event.type == "content"
        # Type should be fixed by Literal
        with pytest.raises(ValueError):
            ContentEvent(type="wrong", content="Test")  # type: ignore[arg-type]


class TestToolCallsEvent:
    def test_tool_calls_event_creation(self):
        """Test ToolCallsEvent creation."""
        tool_calls = [{"id": "call_1", "function": {"name": "test_tool"}}]
        event = ToolCallsEvent(tool_calls=tool_calls)
        assert event.type == "tool_calls"
        assert event.tool_calls == tool_calls

    def test_tool_calls_event_empty_list(self):
        """Test ToolCallsEvent with empty list."""
        event = ToolCallsEvent(tool_calls=[])
        assert event.tool_calls == []


class TestToolOutputsEvent:
    def test_tool_outputs_event_creation(self):
        """Test ToolOutputsEvent creation."""
        tool_outputs = {"call_1": "result1", "call_2": "result2"}
        event = ToolOutputsEvent(tool_outputs=tool_outputs)
        assert event.type == "tool_outputs"
        assert event.tool_outputs == tool_outputs

    def test_tool_outputs_event_empty_dict(self):
        """Test ToolOutputsEvent with empty dict."""
        event = ToolOutputsEvent(tool_outputs={})
        assert event.tool_outputs == {}


class TestMetadataEvent:
    def test_metadata_event_creation(self):
        """Test MetadataEvent creation."""
        metadata = {"model": "test-model", "temperature": 0.7}
        event = MetadataEvent(metadata=metadata)
        assert event.type == "metadata"
        assert event.metadata == metadata

    def test_metadata_event_nested_dict(self):
        """Test MetadataEvent with nested metadata."""
        metadata = {"params": {"temperature": 0.7}, "usage": {"tokens": 100}}
        event = MetadataEvent(metadata=metadata)
        assert event.metadata["params"]["temperature"] == 0.7


class TestReasoningEvent:
    def test_reasoning_event_creation(self):
        """Test ReasoningEvent creation."""
        reasoning = "This is the reasoning step"
        event = ReasoningEvent(reasoning=reasoning)
        assert event.type == "reasoning"
        assert event.reasoning == reasoning

    def test_reasoning_event_empty_string(self):
        """Test ReasoningEvent with empty string."""
        event = ReasoningEvent(reasoning="")
        assert event.reasoning == ""


class TestFinishReasonEvent:
    def test_finish_reason_event_creation(self):
        """Test FinishReasonEvent creation."""
        event = FinishReasonEvent(finish_reason="stop")
        assert event.type == "finish_reason"
        assert event.finish_reason == "stop"

    @pytest.mark.parametrize(
        "finish_reason",
        ["stop", "length", "tool_calls", "content_filter", "function_call"],
    )
    def test_finish_reason_event_various_reasons(self, finish_reason):
        """Test FinishReasonEvent with various finish reasons."""
        event = FinishReasonEvent(finish_reason=finish_reason)
        assert event.finish_reason == finish_reason


class TestResponseCompletedEvent:
    def test_response_completed_event_creation(self):
        """Test ResponseCompletedEvent creation."""
        usage = {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        event = ResponseCompletedEvent(usage=usage)
        assert event.type == "response_completed"
        assert event.usage == usage

    def test_response_completed_event_with_metrics(self):
        """Test ResponseCompletedEvent with detailed metrics."""
        usage = {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "cost": 0.001,
            "model": "test-model",
        }
        event = ResponseCompletedEvent(usage=usage)
        assert event.usage["cost"] == 0.001
        assert event.usage["model"] == "test-model"


class TestResponseCreatedEvent:
    def test_response_created_event_creation(self):
        """Test ResponseCreatedEvent creation."""
        event = ResponseCreatedEvent(response_id="resp_123")
        assert event.type == "response_created"
        assert event.response_id == "resp_123"

    def test_response_created_event_various_ids(self):
        """Test ResponseCreatedEvent with various ID formats."""
        ids = ["resp_123", "response-456", "789", "id_with_underscores"]
        for response_id in ids:
            event = ResponseCreatedEvent(response_id=response_id)
            assert event.response_id == response_id


class TestStructuredOutputEvent:
    def test_structured_output_event_creation(self):
        """Test StructuredOutputEvent creation."""
        structured_output = {"key": "value", "number": 123}
        event = StructuredOutputEvent(structured_output=structured_output)
        assert event.type == "structured_output"
        assert event.structured_output == structured_output

    def test_structured_output_event_with_list(self):
        """Test StructuredOutputEvent with list output."""
        structured_output = [{"item": 1}, {"item": 2}]
        event = StructuredOutputEvent(structured_output=structured_output)
        assert event.structured_output == structured_output

    def test_structured_output_event_with_none(self):
        """Test StructuredOutputEvent with None output."""
        event = StructuredOutputEvent(structured_output=None)
        assert event.structured_output is None


class TestStreamEventUnion:
    def test_stream_event_can_be_any_event_type(self):
        """Test that StreamEvent can be any of the event types."""
        events = [
            ContentEvent(content="test"),
            ToolCallsEvent(tool_calls=[]),
            ToolOutputsEvent(tool_outputs={}),
            MetadataEvent(metadata={}),
            ReasoningEvent(reasoning="test"),
            FinishReasonEvent(finish_reason="stop"),
            ResponseCompletedEvent(usage={}),
            ResponseCreatedEvent(response_id="test"),
            StructuredOutputEvent(structured_output={}),
        ]

        for event in events:
            assert hasattr(event, "type")
            assert event.type in [
                "content",
                "tool_calls",
                "tool_outputs",
                "metadata",
                "reasoning",
                "finish_reason",
                "response_completed",
                "response_created",
                "structured_output",
            ]
