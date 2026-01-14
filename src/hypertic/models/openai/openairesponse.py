import json
from dataclasses import dataclass, field
from os import getenv
from typing import Any

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        AsyncOpenAI as AsyncOpenAIClient,
        OpenAI as OpenAIClient,
        RateLimitError,
    )
    from openai.types.responses.response import Response
    from openai.types.responses.response_stream_event import ResponseStreamEvent
except ImportError as e:
    raise ImportError("`openai` not installed. Please install using `pip install openai -U`") from e

from hypertic.models.base import Base
from hypertic.models.events import (
    ContentEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    StreamEvent,
    ToolCallsEvent,
    ToolOutputsEvent,
)


@dataclass
class OpenAIResponse(Base):
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None

    reasoning_effort: str | None = None

    async_client: Any = field(default=None, init=False)
    client: Any = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("OPENAI_API_KEY")

        if self.api_key is not None:
            self.async_client = AsyncOpenAIClient(api_key=self.api_key)
            self.client = OpenAIClient(api_key=self.api_key)
        else:
            self.async_client = AsyncOpenAIClient()
            self.client = OpenAIClient()

    def _clean_output_item(self, item_dict: dict[str, Any], item_type: str) -> dict[str, Any]:
        if item_type == "function_call":
            valid_fields = {"type", "id", "call_id", "name", "arguments"}
            return {k: v for k, v in item_dict.items() if k in valid_fields}
        elif item_type == "reasoning":
            invalid_fields = {"status", "model_dump_json", "model_fields", "model_config"}
            return {k: v for k, v in item_dict.items() if k not in invalid_fields}
        else:
            invalid_fields = {"status", "model_dump_json", "model_fields", "model_config"}
            return {k: v for k, v in item_dict.items() if k not in invalid_fields}

    async def ahandle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ) -> Any | None:
        processed_messages = []
        for msg in messages:
            formatted_msg = self._format_files_for_openai_response(msg.copy())
            processed_messages.append(formatted_msg)

        instructions = None
        filtered_messages = []
        for message in processed_messages:
            message_role = getattr(message, "role", None) or (message.get("role") if hasattr(message, "get") else None)
            if message_role == "system":
                instructions = getattr(message, "content", None) or (message.get("content") if hasattr(message, "get") else None)
            else:
                filtered_messages.append(message)

        if not instructions and hasattr(tool_executor, "instructions") and tool_executor.instructions:
            instructions = tool_executor.instructions

        input_data: str | list[dict[str, Any]]
        if len(filtered_messages) == 1:
            first_message = filtered_messages[0]
            first_role = getattr(first_message, "role", None) or (first_message.get("role") if hasattr(first_message, "get") else None)
            if first_role == "user":
                content = getattr(first_message, "content", None) or (first_message.get("content") if hasattr(first_message, "get") else "")
                if isinstance(content, list):
                    input_data = filtered_messages
                else:
                    input_data = str(content) if content is not None else ""
            else:
                input_data = filtered_messages
        else:
            input_data = filtered_messages

        request_params: dict[str, Any] = {
            "model": self.model,
            "input": input_data,
            "stream": False,
        }

        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        if self.top_p is not None:
            request_params["top_p"] = self.top_p
        if self.max_tokens is not None:
            request_params["max_output_tokens"] = self.max_tokens
        if instructions is not None:
            request_params["instructions"] = instructions

        if self.reasoning_effort is not None:
            request_params["reasoning"] = {"effort": self.reasoning_effort}

        if tools:
            responses_tools = []
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    responses_tool = {
                        "type": "function",
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "parameters": tool["function"]["parameters"],
                    }
                    responses_tools.append(responses_tool)
                else:
                    responses_tools.append(tool)

            request_params["tools"] = responses_tools
            request_params["tool_choice"] = "auto"
            if hasattr(tool_executor, "parallel_calls"):
                request_params["parallel_tool_calls"] = tool_executor.parallel_calls
            else:
                request_params["parallel_tool_calls"] = True

        try:
            api_response: Response
            if response_format is not None:
                api_response = await self.async_client.responses.parse(text_format=response_format, **request_params)
            else:
                api_response = await self.async_client.responses.create(**request_params)

            if hasattr(api_response, "error") and api_response.error:
                error_obj = api_response.error
                error_msg = getattr(error_obj, "message", "Unknown error") if error_obj else "Unknown error"
                raise Exception(f"OpenAI API error: {error_msg}")

            if hasattr(api_response, "incomplete_details") and api_response.incomplete_details:
                incomplete_obj = api_response.incomplete_details
                incomplete_msg = getattr(incomplete_obj, "reason", "Response incomplete") if incomplete_obj else "Response incomplete"
                raise Exception(f"OpenAI API incomplete: {incomplete_msg}")

            output = api_response.output if hasattr(api_response, "output") else []
            if not output:
                raise Exception("No output received from OpenAI Responses API")

            function_calls = []
            tool_outputs = {}
            has_tool_calls = False

            for output_item in output:
                item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                if item_type == "function_call":
                    has_tool_calls = True

                    item_id = getattr(output_item, "id", None) or (output_item.get("id") if hasattr(output_item, "get") else "")
                    item_name = getattr(output_item, "name", None) or (output_item.get("name") if hasattr(output_item, "get") else "")
                    item_arguments = getattr(output_item, "arguments", None) or (
                        output_item.get("arguments") if hasattr(output_item, "get") else "{}"
                    )
                    item_call_id = getattr(output_item, "call_id", None) or (output_item.get("call_id") if hasattr(output_item, "get") else "")

                    function_call = {
                        "id": item_id,
                        "call_id": item_call_id,
                        "type": "function",
                        "function": {"name": item_name, "arguments": item_arguments},
                    }
                    function_calls.append(function_call)

            if has_tool_calls:
                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls and len(function_calls) > 1:
                    first_function_call = function_calls[0]
                    function_dict = first_function_call.get("function", {})
                    function_name = function_dict.get("name", "") if isinstance(function_dict, dict) else ""

                    tool_outputs = await self._execute_tools_parallel_async(tool_executor, [first_function_call])
                    tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                    first_item_found = False
                    for output_item in output:
                        item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                        if item_type == "function_call" and not first_item_found:
                            if hasattr(output_item, "model_dump"):
                                item_dict = output_item.model_dump()
                            elif hasattr(output_item, "__dict__"):
                                item_dict = output_item.__dict__
                            else:
                                item_dict = output_item if isinstance(output_item, dict) else {}

                            cleaned_item = self._clean_output_item(item_dict, item_type)
                            messages.append(cleaned_item)

                            item_call_id = item_dict.get("call_id", "")
                            messages.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": item_call_id,
                                    "output": json.dumps({function_name: tool_result}),
                                }
                            )
                            first_item_found = True
                        elif item_type == "reasoning":
                            if hasattr(output_item, "model_dump"):
                                item_dict = output_item.model_dump()
                            elif hasattr(output_item, "__dict__"):
                                item_dict = output_item.__dict__
                            else:
                                item_dict = output_item if isinstance(output_item, dict) else {}
                            cleaned_item = self._clean_output_item(item_dict, item_type)
                            messages.append(cleaned_item)

                    if hasattr(tool_executor, "_tool_outputs"):
                        tool_executor._tool_outputs.update(tool_outputs)
                    if hasattr(tool_executor, "_tool_calls"):
                        first_function_call_clean = {k: v for k, v in first_function_call.items() if k != "call_id"}
                        tool_executor._tool_calls.append(first_function_call_clean)

                    usage_obj = api_response.usage if hasattr(api_response, "usage") else None
                    usage: dict[str, Any] = {}
                    if usage_obj is not None:
                        if hasattr(usage_obj, "__dict__"):
                            usage = usage_obj.__dict__
                        elif hasattr(usage_obj, "model_dump"):
                            usage = usage_obj.model_dump()
                        elif isinstance(usage_obj, dict):
                            usage = usage_obj
                    self._accumulate_metrics(usage)

                    return None

                tool_outputs = await self._execute_tools_parallel_async(tool_executor, function_calls)

                for output_item in output:
                    item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                    if item_type == "function_call":
                        if hasattr(output_item, "model_dump"):
                            item_dict = output_item.model_dump()
                        elif hasattr(output_item, "__dict__"):
                            item_dict = output_item.__dict__
                        else:
                            item_dict = output_item if isinstance(output_item, dict) else {}

                        item_id = item_dict.get("id", "")
                        item_name = item_dict.get("name", "")
                        item_arguments = item_dict.get("arguments", "{}")
                        item_call_id = item_dict.get("call_id", "")

                        function_name = item_name
                        tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                        cleaned_item = self._clean_output_item(item_dict, item_type)
                        messages.append(cleaned_item)
                        messages.append(
                            {
                                "type": "function_call_output",
                                "call_id": item_call_id,
                                "output": json.dumps({function_name: tool_result}),
                            }
                        )
                    elif item_type == "reasoning":
                        if hasattr(output_item, "model_dump"):
                            item_dict = output_item.model_dump()
                        elif hasattr(output_item, "__dict__"):
                            item_dict = output_item.__dict__
                        else:
                            item_dict = output_item if isinstance(output_item, dict) else {}
                        cleaned_item = self._clean_output_item(item_dict, item_type)
                        messages.append(cleaned_item)

                if hasattr(tool_executor, "_tool_outputs"):
                    tool_executor._tool_outputs.update(tool_outputs)
                if hasattr(tool_executor, "_tool_calls"):
                    function_calls_clean = [{k: v for k, v in fc.items() if k != "call_id"} for fc in function_calls]
                    tool_executor._tool_calls.extend(function_calls_clean)

                usage_obj = api_response.usage if hasattr(api_response, "usage") else None
                usage_dict: dict[str, Any] = {}
                if usage_obj is not None:
                    if hasattr(usage_obj, "__dict__"):
                        usage_dict = usage_obj.__dict__
                    elif hasattr(usage_obj, "model_dump"):
                        usage_dict = usage_obj.model_dump()
                    elif isinstance(usage_obj, dict):
                        usage_dict = usage_obj
                self._accumulate_metrics(usage_dict)
                return None

            response_text = ""
            parsed_content = None

            if response_format is not None and hasattr(api_response, "output_parsed"):
                parsed_content = api_response.output_parsed
                if hasattr(parsed_content, "model_dump"):
                    parsed_content = parsed_content.model_dump()

            for output_item in output:
                item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                if item_type == "message":
                    content = getattr(output_item, "content", None) or (output_item.get("content") if hasattr(output_item, "get") else [])
                    if isinstance(content, list):
                        for content_part in content:
                            part_type = getattr(content_part, "type", None) or (content_part.get("type") if hasattr(content_part, "get") else None)
                            if part_type == "output_text":
                                text = getattr(content_part, "text", None) or (content_part.get("text") if hasattr(content_part, "get") else "")
                                response_text += text
                    elif isinstance(content, str):
                        response_text += content

            if parsed_content is not None:
                final_content = json.dumps(parsed_content) if isinstance(parsed_content, dict) else str(parsed_content)
            else:
                final_content = response_text

            final_tool_calls = function_calls
            final_tool_outputs = tool_outputs

            if hasattr(tool_executor, "_tool_calls") and tool_executor._tool_calls:
                final_tool_calls = tool_executor._tool_calls.copy()
            if hasattr(tool_executor, "_tool_outputs") and tool_executor._tool_outputs:
                final_tool_outputs = tool_executor._tool_outputs.copy()

            usage_obj_final = api_response.usage if hasattr(api_response, "usage") else None
            usage_final: dict[str, Any] = {}
            if usage_obj_final is not None:
                if hasattr(usage_obj_final, "__dict__"):
                    usage_final = usage_obj_final.__dict__
                elif hasattr(usage_obj_final, "model_dump"):
                    usage_final = usage_obj_final.model_dump()
                elif isinstance(usage_obj_final, dict):
                    usage_final = usage_obj_final

            self._accumulate_metrics(usage_final)

            cumulative_metrics = self._get_cumulative_metrics()

            params: dict[str, Any] = {}
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
            if self.reasoning_effort is not None:
                params["reasoning_effort"] = str(self.reasoning_effort)

            usage_info = {
                "model": api_response.model if hasattr(api_response, "model") else self.model,
                "params": params,
                "finish_reason": "stop",
                "input_tokens": cumulative_metrics.input_tokens,
                "output_tokens": cumulative_metrics.output_tokens,
            }

            return self._create_llm_response(final_content, usage_info, final_tool_calls, final_tool_outputs)

        except RateLimitError as e:
            raise Exception(f"Rate limit error from OpenAI SDK: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from OpenAI SDK: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from OpenAI SDK: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from OpenAI SDK: {e}") from e

    async def ahandle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        """Streaming handler for Responses API."""
        processed_messages = []
        for msg in messages:
            formatted_msg = self._format_files_for_openai_response(msg.copy())
            processed_messages.append(formatted_msg)

        instructions = None
        filtered_messages = []
        for message in processed_messages:
            message_role = getattr(message, "role", None) or (message.get("role") if hasattr(message, "get") else None)
            if message_role == "system":
                instructions = getattr(message, "content", None) or (message.get("content") if hasattr(message, "get") else None)
            else:
                filtered_messages.append(message)

        if not instructions and hasattr(tool_executor, "instructions") and tool_executor.instructions:
            instructions = tool_executor.instructions

        input_data: str | list[dict[str, Any]]
        if len(filtered_messages) == 1:
            first_message = filtered_messages[0]
            first_role = getattr(first_message, "role", None) or (first_message.get("role") if hasattr(first_message, "get") else None)
            if first_role == "user":
                content = getattr(first_message, "content", None) or (first_message.get("content") if hasattr(first_message, "get") else "")
                if isinstance(content, list):
                    input_data = filtered_messages
                else:
                    input_data = str(content) if content is not None else ""
            else:
                input_data = filtered_messages
        else:
            input_data = filtered_messages

        request_params: dict[str, Any] = {
            "model": self.model,
            "input": input_data,
        }

        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        if self.top_p is not None:
            request_params["top_p"] = self.top_p
        if self.max_tokens is not None:
            request_params["max_output_tokens"] = self.max_tokens
        if instructions is not None:
            request_params["instructions"] = instructions

        if self.reasoning_effort is not None:
            request_params["reasoning"] = {"effort": self.reasoning_effort}

        if tools:
            responses_tools = []
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    responses_tool = {
                        "type": "function",
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "parameters": tool["function"]["parameters"],
                    }
                    responses_tools.append(responses_tool)
                else:
                    responses_tools.append(tool)

            request_params["tools"] = responses_tools
            request_params["tool_choice"] = "auto"
            if hasattr(tool_executor, "parallel_calls"):
                request_params["parallel_tool_calls"] = tool_executor.parallel_calls
            else:
                request_params["parallel_tool_calls"] = True

        try:
            if response_format is not None:
                request_params["text_format"] = response_format

            async with self.async_client.responses.stream(**request_params) as stream:
                tool_use: dict[str, Any] = {}

                async for event in stream:
                    events = await self._process_streaming_event_async(event, tool_executor, messages, tool_use)
                    for event_data in events:
                        if isinstance(event_data, bool):
                            yield event_data
                            continue

                        yield event_data

        except RateLimitError as e:
            raise Exception(f"Rate limit error from OpenAI SDK: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from OpenAI SDK: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from OpenAI SDK: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from OpenAI SDK: {e}") from e

    async def _process_streaming_event_async(
        self,
        event: ResponseStreamEvent,
        tool_executor: Any,
        messages: list[dict[str, Any]],
        tool_use: dict[str, Any],
    ):
        events: list[StreamEvent | bool] = []
        event_type = getattr(event, "type", "")

        if event_type == "response.created":
            response_obj = getattr(event, "response", None)
            response_id = getattr(response_obj, "id", "") if response_obj else ""
            events.append(ResponseCreatedEvent(response_id=response_id))

        elif event_type == "response.in_progress":
            pass

        elif event_type == "response.output_item.added":
            item = getattr(event, "item", {})
            item_type = getattr(item, "type", None) or (item.get("type") if hasattr(item, "get") else None)
            if item_type == "message":
                item_id = getattr(item, "id", None) or (item.get("id") if hasattr(item, "get") else "")
                pass
            elif item_type == "function_call":
                pass

        elif event_type == "response.content_part.added":
            pass

        elif event_type == "response.output_text.delta":
            delta_text = getattr(event, "delta", "")
            if delta_text:
                if not hasattr(tool_executor, "_streaming_content_buffer"):
                    tool_executor._streaming_content_buffer = ""
                tool_executor._streaming_content_buffer += delta_text

                events.append(ContentEvent(content=delta_text))

        elif event_type == "response.output_text.done" or event_type == "response.content_part.done":
            pass

        elif event_type == "response.output_item.done":
            item = getattr(event, "item", {})
            item_type = getattr(item, "type", None) or (item.get("type") if hasattr(item, "get") else None)
            if item_type == "function_call":
                function_name = getattr(item, "name", None) or (item.get("name") if hasattr(item, "get") else "")
                arguments = getattr(item, "arguments", None) or (item.get("arguments") if hasattr(item, "get") else "{}")
                item_id = getattr(item, "id", None) or (item.get("id") if hasattr(item, "get") else "")
                item_call_id = getattr(item, "call_id", None) or (item.get("call_id") if hasattr(item, "get") else "")

                if not hasattr(tool_executor, "_pending_function_calls"):
                    tool_executor._pending_function_calls = []

                function_call = {
                    "id": item_id,
                    "call_id": item_call_id,
                    "type": "function",
                    "function": {"name": function_name, "arguments": arguments},
                }
                tool_executor._pending_function_calls.append({"function_call": function_call, "item": item, "item_call_id": item_call_id})

                events.append(True)

        elif event_type == "response.completed":
            response_data = getattr(event, "response", None)
            usage_obj = getattr(response_data, "usage", None) if response_data and hasattr(response_data, "usage") else None
            usage: dict[str, Any] = {}
            if usage_obj:
                if hasattr(usage_obj, "__dict__"):
                    usage = usage_obj.__dict__
                elif hasattr(usage_obj, "model_dump"):
                    usage = usage_obj.model_dump()
                elif isinstance(usage_obj, dict):
                    usage = usage_obj

            if hasattr(tool_executor, "_Agent__usage"):
                tool_executor._Agent__usage = usage

            output = getattr(response_data, "output", []) if response_data and hasattr(response_data, "output") else []

            has_function_calls = False
            for item in output:
                item_type = getattr(item, "type", None) or (item.get("type") if hasattr(item, "get") else None)
                if item_type == "function_call":
                    has_function_calls = True
                    break

            if has_function_calls:
                if hasattr(tool_executor, "_pending_function_calls") and tool_executor._pending_function_calls:
                    function_calls = [call["function_call"] for call in tool_executor._pending_function_calls]

                    for i, output_item in enumerate(output):
                        item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                        if item_type == "reasoning":
                            if i + 1 < len(output):
                                next_item = output[i + 1]
                                next_item_type = getattr(next_item, "type", None) or (next_item.get("type") if hasattr(next_item, "get") else None)
                                if next_item_type == "function_call":
                                    next_item_id = getattr(next_item, "id", None) or (next_item.get("id") if hasattr(next_item, "get") else None)
                                    has_matching_call = False
                                    for call_data in tool_executor._pending_function_calls:
                                        call_item = call_data["item"]
                                        call_item_id = getattr(call_item, "id", None) or (call_item.get("id") if hasattr(call_item, "get") else None)
                                        if call_item_id == next_item_id:
                                            has_matching_call = True
                                            break

                                    if has_matching_call:
                                        if hasattr(output_item, "model_dump"):
                                            item_dict = output_item.model_dump()
                                        elif hasattr(output_item, "__dict__"):
                                            item_dict = output_item.__dict__
                                        else:
                                            item_dict = output_item if isinstance(output_item, dict) else {}
                                        cleaned_item = self._clean_output_item(item_dict, item_type)
                                        messages.append(cleaned_item)
                        elif item_type == "function_call":
                            item_id = getattr(output_item, "id", None) or (output_item.get("id") if hasattr(output_item, "get") else None)
                            if not any(
                                getattr(msg, "id", None) == item_id or (msg.get("id") if hasattr(msg, "get") else None) == item_id for msg in messages
                            ):
                                if hasattr(output_item, "model_dump"):
                                    item_dict = output_item.model_dump()
                                elif hasattr(output_item, "__dict__"):
                                    item_dict = output_item.__dict__
                                else:
                                    item_dict = output_item if isinstance(output_item, dict) else {}
                                cleaned_item = self._clean_output_item(item_dict, "function_call")
                                messages.append(cleaned_item)

                    function_calls_for_event = [{k: v for k, v in fc.items() if k != "call_id"} for fc in function_calls]

                    if hasattr(tool_executor, "_tool_calls"):
                        tool_executor._tool_calls.extend(function_calls_for_event)

                    events.append(ToolCallsEvent(tool_calls=function_calls_for_event))

                    if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls and len(function_calls) > 1:
                        first_function_call = function_calls[0]
                        function_name = first_function_call["function"]["name"]

                        tool_outputs = await self._execute_tools_parallel_async(tool_executor, [first_function_call])
                        tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                        first_item_found = False
                        for i, output_item in enumerate(output):
                            item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                            if item_type == "function_call" and not first_item_found:
                                if i > 0:
                                    prev_item = output[i - 1]
                                    prev_item_type = getattr(prev_item, "type", None) or (
                                        prev_item.get("type") if hasattr(prev_item, "get") else None
                                    )
                                    if prev_item_type == "reasoning":
                                        if hasattr(prev_item, "model_dump"):
                                            prev_item_dict = prev_item.model_dump()
                                        elif hasattr(prev_item, "__dict__"):
                                            prev_item_dict = prev_item.__dict__
                                        else:
                                            prev_item_dict = prev_item if isinstance(prev_item, dict) else {}
                                        cleaned_prev_item = self._clean_output_item(prev_item_dict, "reasoning")
                                        messages.append(cleaned_prev_item)

                                item_id = getattr(output_item, "id", None) or (output_item.get("id") if hasattr(output_item, "get") else None)
                                matching_call_data = None
                                for call_data in tool_executor._pending_function_calls:
                                    call_item = call_data["item"]
                                    call_item_id = getattr(call_item, "id", None) or (call_item.get("id") if hasattr(call_item, "get") else None)
                                    if call_item_id == item_id:
                                        matching_call_data = call_data
                                        break

                                if matching_call_data:
                                    item_call_id = matching_call_data["item_call_id"]

                                    if hasattr(output_item, "model_dump"):
                                        item_dict = output_item.model_dump()
                                    elif hasattr(output_item, "__dict__"):
                                        item_dict = output_item.__dict__
                                    else:
                                        item_dict = output_item if isinstance(output_item, dict) else {}

                                    cleaned_item = self._clean_output_item(item_dict, "function_call")
                                    messages.append(cleaned_item)

                                    messages.append(
                                        {
                                            "type": "function_call_output",
                                            "call_id": item_call_id,
                                            "output": json.dumps({function_name: tool_result}),
                                        }
                                    )

                                    if hasattr(tool_executor, "_tool_outputs"):
                                        tool_executor._tool_outputs[function_name] = tool_result
                                    if hasattr(tool_executor, "_tool_calls"):
                                        first_function_call_clean = {k: v for k, v in first_function_call.items() if k != "call_id"}
                                        tool_executor._tool_calls.append(first_function_call_clean)

                                    first_item_found = True
                                    break

                        events.append(ToolOutputsEvent(tool_outputs=tool_outputs))

                        tool_executor._pending_function_calls = []

                        events.append(True)
                        return events

                    tool_outputs = await self._execute_tools_parallel_async(tool_executor, function_calls)

                    for i, output_item in enumerate(output):
                        item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                        if item_type == "function_call":
                            item_id = getattr(output_item, "id", None) or (output_item.get("id") if hasattr(output_item, "get") else None)
                            matching_call_data = None
                            for call_data in tool_executor._pending_function_calls:
                                call_item = call_data["item"]
                                call_item_id = getattr(call_item, "id", None) or (call_item.get("id") if hasattr(call_item, "get") else None)
                                if call_item_id == item_id:
                                    matching_call_data = call_data
                                    break

                            if matching_call_data:
                                if i > 0:
                                    prev_item = output[i - 1]
                                    prev_item_type = getattr(prev_item, "type", None) or (
                                        prev_item.get("type") if hasattr(prev_item, "get") else None
                                    )
                                    if prev_item_type == "reasoning":
                                        prev_item_id = getattr(prev_item, "id", None) or (prev_item.get("id") if hasattr(prev_item, "get") else None)
                                        if not any(
                                            getattr(msg, "id", None) == prev_item_id
                                            or (msg.get("id") if hasattr(msg, "get") else None) == prev_item_id
                                            for msg in messages
                                        ):
                                            if hasattr(prev_item, "model_dump"):
                                                prev_item_dict = prev_item.model_dump()
                                            elif hasattr(prev_item, "__dict__"):
                                                prev_item_dict = prev_item.__dict__
                                            else:
                                                prev_item_dict = prev_item if isinstance(prev_item, dict) else {}
                                            cleaned_prev_item = self._clean_output_item(prev_item_dict, "reasoning")
                                            messages.append(cleaned_prev_item)

                                function_call = matching_call_data["function_call"]
                                item = matching_call_data["item"]
                                item_call_id = matching_call_data["item_call_id"]

                                function_name = function_call["function"]["name"]
                                tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                                if hasattr(tool_executor, "_tool_outputs"):
                                    tool_executor._tool_outputs[function_name] = tool_result

                                if not any(
                                    getattr(msg, "id", None) == item_id or (msg.get("id") if hasattr(msg, "get") else None) == item_id
                                    for msg in messages
                                ):
                                    if hasattr(output_item, "model_dump"):
                                        item_dict = output_item.model_dump()
                                    elif hasattr(output_item, "__dict__"):
                                        item_dict = output_item.__dict__
                                    else:
                                        item_dict = output_item if isinstance(output_item, dict) else {}
                                    cleaned_item = self._clean_output_item(item_dict, "function_call")
                                    messages.append(cleaned_item)

                                messages.append(
                                    {
                                        "type": "function_call_output",
                                        "call_id": item_call_id,
                                        "output": json.dumps({function_name: tool_result}),
                                    }
                                )

                                if hasattr(tool_executor, "_tool_calls"):
                                    function_call_clean = {k: v for k, v in function_call.items() if k != "call_id"}
                                    tool_executor._tool_calls.append(function_call_clean)

                    events.append(ToolOutputsEvent(tool_outputs=tool_outputs))

                    tool_executor._pending_function_calls = []

                if hasattr(tool_executor, "_streaming_content_buffer"):
                    tool_executor._streaming_content_buffer = ""

                if usage:
                    if hasattr(usage, "__dict__"):
                        usage_dict = usage.__dict__
                    elif hasattr(usage, "model_dump"):
                        usage_dict = usage.model_dump()
                    else:
                        usage_dict = usage if isinstance(usage, dict) else {}

                    self._accumulate_metrics(usage_dict)

                events.append(ResponseCompletedEvent(usage=usage_dict))
                events.append(True)
            else:
                if hasattr(tool_executor, "_streaming_content_buffer"):
                    tool_executor._streaming_content_buffer = ""

                if usage:
                    if hasattr(usage, "__dict__"):
                        usage_dict = usage.__dict__
                    elif hasattr(usage, "model_dump"):
                        usage_dict = usage.model_dump()
                    else:
                        usage_dict = usage if isinstance(usage, dict) else {}

                    self._accumulate_metrics(usage_dict)

                events.append(ResponseCompletedEvent(usage=usage_dict))
                events.append(False)

        elif event_type == "error":
            error = getattr(event, "error", {})
            error_msg = error.get("message", "Unknown error") if isinstance(error, dict) else str(error)
            raise Exception(f"Error from OpenAI Responses API: {error_msg}")

        return events

    def handle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ) -> Any | None:
        processed_messages = []
        for msg in messages:
            formatted_msg = self._format_files_for_openai_response(msg.copy())
            processed_messages.append(formatted_msg)

        instructions = None
        filtered_messages = []
        for message in processed_messages:
            message_role = getattr(message, "role", None) or (message.get("role") if hasattr(message, "get") else None)
            if message_role == "system":
                instructions = getattr(message, "content", None) or (message.get("content") if hasattr(message, "get") else None)
            else:
                filtered_messages.append(message)

        if not instructions and hasattr(tool_executor, "instructions") and tool_executor.instructions:
            instructions = tool_executor.instructions

        input_data: str | list[dict[str, Any]]
        if len(filtered_messages) == 1:
            first_message = filtered_messages[0]
            first_role = getattr(first_message, "role", None) or (first_message.get("role") if hasattr(first_message, "get") else None)
            if first_role == "user":
                content = getattr(first_message, "content", None) or (first_message.get("content") if hasattr(first_message, "get") else "")
                if isinstance(content, list):
                    input_data = filtered_messages
                else:
                    input_data = str(content) if content is not None else ""
            else:
                input_data = filtered_messages
        else:
            input_data = filtered_messages

        request_params: dict[str, Any] = {
            "model": self.model,
            "input": input_data,
            "stream": False,
        }

        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        if self.top_p is not None:
            request_params["top_p"] = self.top_p
        if self.max_tokens is not None:
            request_params["max_output_tokens"] = self.max_tokens
        if instructions is not None:
            request_params["instructions"] = instructions

        if self.reasoning_effort is not None:
            request_params["reasoning"] = {"effort": self.reasoning_effort}

        if tools:
            responses_tools = []
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    responses_tool = {
                        "type": "function",
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "parameters": tool["function"]["parameters"],
                    }
                    responses_tools.append(responses_tool)
                else:
                    responses_tools.append(tool)

            request_params["tools"] = responses_tools
            request_params["tool_choice"] = "auto"
            if hasattr(tool_executor, "parallel_calls"):
                request_params["parallel_tool_calls"] = tool_executor.parallel_calls
            else:
                request_params["parallel_tool_calls"] = True

        try:
            api_response: Response
            if response_format is not None:
                api_response = self.client.responses.parse(text_format=response_format, **request_params)
            else:
                api_response = self.client.responses.create(**request_params)

            if hasattr(api_response, "error") and api_response.error:
                error_obj = api_response.error
                error_msg = getattr(error_obj, "message", "Unknown error") if error_obj else "Unknown error"
                raise Exception(f"OpenAI API error: {error_msg}")

            if hasattr(api_response, "incomplete_details") and api_response.incomplete_details:
                incomplete_obj = api_response.incomplete_details
                incomplete_msg = getattr(incomplete_obj, "reason", "Response incomplete") if incomplete_obj else "Response incomplete"
                raise Exception(f"OpenAI API incomplete: {incomplete_msg}")

            output = api_response.output if hasattr(api_response, "output") else []
            if not output:
                raise Exception("No output received from OpenAI Responses API")

            function_calls = []
            tool_outputs = {}
            has_tool_calls = False

            for output_item in output:
                item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                if item_type == "function_call":
                    has_tool_calls = True

                    item_id = getattr(output_item, "id", None) or (output_item.get("id") if hasattr(output_item, "get") else "")
                    item_name = getattr(output_item, "name", None) or (output_item.get("name") if hasattr(output_item, "get") else "")
                    item_arguments = getattr(output_item, "arguments", None) or (
                        output_item.get("arguments") if hasattr(output_item, "get") else "{}"
                    )
                    item_call_id = getattr(output_item, "call_id", None) or (output_item.get("call_id") if hasattr(output_item, "get") else "")

                    function_call = {
                        "id": item_id,
                        "type": "function",
                        "function": {"name": item_name, "arguments": item_arguments},
                    }
                    function_calls.append(function_call)

            if has_tool_calls:
                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls and len(function_calls) > 1:
                    first_function_call = function_calls[0]
                    function_dict = first_function_call.get("function", {})
                    function_name = function_dict.get("name", "") if isinstance(function_dict, dict) else ""

                    tool_outputs = self._execute_tools_parallel_sync(tool_executor, [first_function_call])
                    tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                    first_item_found = False
                    for output_item in output:
                        item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                        if item_type == "function_call" and not first_item_found:
                            if hasattr(output_item, "model_dump"):
                                item_dict = output_item.model_dump()
                            elif hasattr(output_item, "__dict__"):
                                item_dict = output_item.__dict__
                            else:
                                item_dict = output_item if isinstance(output_item, dict) else {}

                            cleaned_item = self._clean_output_item(item_dict, item_type)
                            messages.append(cleaned_item)

                            item_call_id = item_dict.get("call_id", "")
                            messages.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": item_call_id,
                                    "output": json.dumps({function_name: tool_result}),
                                }
                            )
                            first_item_found = True
                        elif item_type == "reasoning":
                            if hasattr(output_item, "model_dump"):
                                item_dict = output_item.model_dump()
                            elif hasattr(output_item, "__dict__"):
                                item_dict = output_item.__dict__
                            else:
                                item_dict = output_item if isinstance(output_item, dict) else {}
                            cleaned_item = self._clean_output_item(item_dict, item_type)
                            messages.append(cleaned_item)

                    if hasattr(tool_executor, "_tool_outputs"):
                        tool_executor._tool_outputs.update(tool_outputs)
                    if hasattr(tool_executor, "_tool_calls"):
                        tool_executor._tool_calls.append(first_function_call)

                    usage_obj = api_response.usage if hasattr(api_response, "usage") else None
                    usage: dict[str, Any] = {}
                    if usage_obj is not None:
                        if hasattr(usage_obj, "__dict__"):
                            usage = usage_obj.__dict__
                        elif hasattr(usage_obj, "model_dump"):
                            usage = usage_obj.model_dump()
                        elif isinstance(usage_obj, dict):
                            usage = usage_obj
                    self._accumulate_metrics(usage)

                    return None

                tool_outputs = self._execute_tools_parallel_sync(tool_executor, function_calls)

                for output_item in output:
                    item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                    if item_type == "function_call":
                        if hasattr(output_item, "model_dump"):
                            item_dict = output_item.model_dump()
                        elif hasattr(output_item, "__dict__"):
                            item_dict = output_item.__dict__
                        else:
                            item_dict = output_item if isinstance(output_item, dict) else {}

                        item_id = item_dict.get("id", "")
                        item_name = item_dict.get("name", "")
                        item_arguments = item_dict.get("arguments", "{}")
                        item_call_id = item_dict.get("call_id", "")

                        function_name = item_name
                        tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                        cleaned_item = self._clean_output_item(item_dict, item_type)
                        messages.append(cleaned_item)
                        messages.append(
                            {
                                "type": "function_call_output",
                                "call_id": item_call_id,
                                "output": json.dumps({function_name: tool_result}),
                            }
                        )
                    elif item_type == "reasoning":
                        if hasattr(output_item, "model_dump"):
                            item_dict = output_item.model_dump()
                        elif hasattr(output_item, "__dict__"):
                            item_dict = output_item.__dict__
                        else:
                            item_dict = output_item if isinstance(output_item, dict) else {}
                        cleaned_item = self._clean_output_item(item_dict, item_type)
                        messages.append(cleaned_item)

            if hasattr(tool_executor, "_tool_outputs"):
                tool_executor._tool_outputs.update(tool_outputs)
            if hasattr(tool_executor, "_tool_calls"):
                tool_executor._tool_calls.extend(function_calls)

            if has_tool_calls:
                usage_obj_tool = api_response.usage if hasattr(api_response, "usage") else None
                usage_tool: dict[str, Any] = {}
                if usage_obj_tool is not None:
                    if hasattr(usage_obj_tool, "__dict__"):
                        usage_tool = usage_obj_tool.__dict__
                    elif hasattr(usage_obj_tool, "model_dump"):
                        usage_tool = usage_obj_tool.model_dump()
                    elif isinstance(usage_obj_tool, dict):
                        usage_tool = usage_obj_tool
                self._accumulate_metrics(usage_tool)
                return None

            response_text = ""
            parsed_content = None

            if response_format is not None and hasattr(api_response, "output_parsed"):
                parsed_content = api_response.output_parsed
                if hasattr(parsed_content, "model_dump"):
                    parsed_content = parsed_content.model_dump()

            for output_item in output:
                item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                if item_type == "message":
                    content = getattr(output_item, "content", None) or (output_item.get("content") if hasattr(output_item, "get") else [])
                    if isinstance(content, list):
                        for content_part in content:
                            part_type = getattr(content_part, "type", None) or (content_part.get("type") if hasattr(content_part, "get") else None)
                            if part_type == "output_text":
                                text = getattr(content_part, "text", None) or (content_part.get("text") if hasattr(content_part, "get") else "")
                                response_text += text
                    elif isinstance(content, str):
                        response_text += content

            if parsed_content is not None:
                final_content = json.dumps(parsed_content) if isinstance(parsed_content, dict) else str(parsed_content)
            else:
                final_content = response_text

            final_tool_calls = function_calls
            final_tool_outputs = tool_outputs

            if hasattr(tool_executor, "_tool_calls") and tool_executor._tool_calls:
                final_tool_calls = tool_executor._tool_calls.copy()
            if hasattr(tool_executor, "_tool_outputs") and tool_executor._tool_outputs:
                final_tool_outputs = tool_executor._tool_outputs.copy()

            usage_obj_final = api_response.usage if hasattr(api_response, "usage") else None
            usage_final: dict[str, Any] = {}
            if usage_obj_final is not None:
                if hasattr(usage_obj_final, "__dict__"):
                    usage_final = usage_obj_final.__dict__
                elif hasattr(usage_obj_final, "model_dump"):
                    usage_final = usage_obj_final.model_dump()
                elif isinstance(usage_obj_final, dict):
                    usage_final = usage_obj_final

            self._accumulate_metrics(usage_final)

            cumulative_metrics = self._get_cumulative_metrics()

            params: dict[str, Any] = {}
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
            if self.reasoning_effort is not None:
                params["reasoning_effort"] = str(self.reasoning_effort)

            usage_info = {
                "model": api_response.model if hasattr(api_response, "model") else self.model,
                "params": params,
                "finish_reason": "stop",
                "input_tokens": cumulative_metrics.input_tokens,
                "output_tokens": cumulative_metrics.output_tokens,
            }

            return self._create_llm_response(final_content, usage_info, final_tool_calls, final_tool_outputs)

        except RateLimitError as e:
            raise Exception(f"Rate limit error from OpenAI SDK: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from OpenAI SDK: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from OpenAI SDK: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from OpenAI SDK: {e}") from e

    def handle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        processed_messages = []
        for msg in messages:
            formatted_msg = self._format_files_for_openai_response(msg.copy())
            processed_messages.append(formatted_msg)

        instructions = None
        filtered_messages = []
        for message in processed_messages:
            message_role = getattr(message, "role", None) or (message.get("role") if hasattr(message, "get") else None)
            if message_role == "system":
                instructions = getattr(message, "content", None) or (message.get("content") if hasattr(message, "get") else None)
            else:
                filtered_messages.append(message)

        if not instructions and hasattr(tool_executor, "instructions") and tool_executor.instructions:
            instructions = tool_executor.instructions

        input_data: str | list[dict[str, Any]]
        if len(filtered_messages) == 1:
            first_message = filtered_messages[0]
            first_role = getattr(first_message, "role", None) or (first_message.get("role") if hasattr(first_message, "get") else None)
            if first_role == "user":
                content = getattr(first_message, "content", None) or (first_message.get("content") if hasattr(first_message, "get") else "")
                if isinstance(content, list):
                    input_data = filtered_messages
                else:
                    input_data = str(content) if content is not None else ""
            else:
                input_data = filtered_messages
        else:
            input_data = filtered_messages

        request_params: dict[str, Any] = {
            "model": self.model,
            "input": input_data,
        }

        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        if self.top_p is not None:
            request_params["top_p"] = self.top_p
        if self.max_tokens is not None:
            request_params["max_output_tokens"] = self.max_tokens
        if instructions is not None:
            request_params["instructions"] = instructions

        if self.reasoning_effort is not None:
            request_params["reasoning"] = {"effort": self.reasoning_effort}

        if tools:
            responses_tools = []
            for tool in tools:
                if tool.get("type") == "function" and "function" in tool:
                    responses_tool = {
                        "type": "function",
                        "name": tool["function"]["name"],
                        "description": tool["function"]["description"],
                        "parameters": tool["function"]["parameters"],
                    }
                    responses_tools.append(responses_tool)
                else:
                    responses_tools.append(tool)

            request_params["tools"] = responses_tools
            request_params["tool_choice"] = "auto"
            if hasattr(tool_executor, "parallel_calls"):
                request_params["parallel_tool_calls"] = tool_executor.parallel_calls
            else:
                request_params["parallel_tool_calls"] = True

        try:
            if response_format is not None:
                request_params["text_format"] = response_format

            with self.client.responses.stream(**request_params) as stream:
                tool_use: dict[str, Any] = {}

                for event in stream:
                    events = self._process_streaming_event_sync(event, tool_executor, messages, tool_use)
                    for event_data in events:
                        if isinstance(event_data, bool):
                            yield event_data
                            continue

                        yield event_data

        except RateLimitError as e:
            raise Exception(f"Rate limit error from OpenAI SDK: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from OpenAI SDK: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from OpenAI SDK: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from OpenAI SDK: {e}") from e

    def _process_streaming_event_sync(
        self,
        event: ResponseStreamEvent,
        tool_executor: Any,
        messages: list[dict[str, Any]],
        tool_use: dict[str, Any],
    ):
        events: list[StreamEvent | bool] = []
        event_type = getattr(event, "type", "")

        if event_type == "response.created":
            response_obj = getattr(event, "response", None)
            response_id = getattr(response_obj, "id", "") if response_obj else ""
            events.append(ResponseCreatedEvent(response_id=response_id))

        elif event_type == "response.in_progress":
            pass

        elif event_type == "response.output_item.added":
            item = getattr(event, "item", {})
            item_type = getattr(item, "type", None) or (item.get("type") if hasattr(item, "get") else None)
            if item_type == "message":
                item_id = getattr(item, "id", None) or (item.get("id") if hasattr(item, "get") else "")
                pass
            elif item_type == "function_call":
                pass

        elif event_type == "response.content_part.added":
            pass

        elif event_type == "response.output_text.delta":
            delta_text = getattr(event, "delta", "")
            if delta_text:
                if not hasattr(tool_executor, "_streaming_content_buffer"):
                    tool_executor._streaming_content_buffer = ""
                tool_executor._streaming_content_buffer += delta_text

                events.append(ContentEvent(content=delta_text))

        elif event_type == "response.output_text.done" or event_type == "response.content_part.done":
            pass

        elif event_type == "response.output_item.done":
            item = getattr(event, "item", {})
            item_type = getattr(item, "type", None) or (item.get("type") if hasattr(item, "get") else None)
            if item_type == "function_call":
                function_name = getattr(item, "name", None) or (item.get("name") if hasattr(item, "get") else "")
                arguments = getattr(item, "arguments", None) or (item.get("arguments") if hasattr(item, "get") else "{}")
                item_id = getattr(item, "id", None) or (item.get("id") if hasattr(item, "get") else "")
                item_call_id = getattr(item, "call_id", None) or (item.get("call_id") if hasattr(item, "get") else "")

                if not hasattr(tool_executor, "_pending_function_calls"):
                    tool_executor._pending_function_calls = []

                function_call = {
                    "id": item_id,
                    "call_id": item_call_id,
                    "type": "function",
                    "function": {"name": function_name, "arguments": arguments},
                }
                tool_executor._pending_function_calls.append({"function_call": function_call, "item": item, "item_call_id": item_call_id})

                events.append(True)

        elif event_type == "response.completed":
            response_data = getattr(event, "response", None)
            usage_obj = getattr(response_data, "usage", None) if response_data and hasattr(response_data, "usage") else None
            usage: dict[str, Any] = {}
            if usage_obj:
                if hasattr(usage_obj, "__dict__"):
                    usage = usage_obj.__dict__
                elif hasattr(usage_obj, "model_dump"):
                    usage = usage_obj.model_dump()
                elif isinstance(usage_obj, dict):
                    usage = usage_obj

            if hasattr(tool_executor, "_Agent__usage"):
                tool_executor._Agent__usage = usage

            output = getattr(response_data, "output", []) if response_data and hasattr(response_data, "output") else []

            has_function_calls = False
            for item in output:
                item_type = getattr(item, "type", None) or (item.get("type") if hasattr(item, "get") else None)
                if item_type == "function_call":
                    has_function_calls = True
                    break

            if has_function_calls:
                if hasattr(tool_executor, "_pending_function_calls") and tool_executor._pending_function_calls:
                    function_calls = [call["function_call"] for call in tool_executor._pending_function_calls]

                    for i, output_item in enumerate(output):
                        item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                        if item_type == "reasoning":
                            if i + 1 < len(output):
                                next_item = output[i + 1]
                                next_item_type = getattr(next_item, "type", None) or (next_item.get("type") if hasattr(next_item, "get") else None)
                                if next_item_type == "function_call":
                                    next_item_id = getattr(next_item, "id", None) or (next_item.get("id") if hasattr(next_item, "get") else None)
                                    has_matching_call = False
                                    for call_data in tool_executor._pending_function_calls:
                                        call_item = call_data["item"]
                                        call_item_id = getattr(call_item, "id", None) or (call_item.get("id") if hasattr(call_item, "get") else None)
                                        if call_item_id == next_item_id:
                                            has_matching_call = True
                                            break

                                    if has_matching_call:
                                        if hasattr(output_item, "model_dump"):
                                            item_dict = output_item.model_dump()
                                        elif hasattr(output_item, "__dict__"):
                                            item_dict = output_item.__dict__
                                        else:
                                            item_dict = output_item if isinstance(output_item, dict) else {}
                                        cleaned_item = self._clean_output_item(item_dict, item_type)
                                        messages.append(cleaned_item)
                        elif item_type == "function_call":
                            item_id = getattr(output_item, "id", None) or (output_item.get("id") if hasattr(output_item, "get") else None)
                            if not any(
                                getattr(msg, "id", None) == item_id or (msg.get("id") if hasattr(msg, "get") else None) == item_id for msg in messages
                            ):
                                if hasattr(output_item, "model_dump"):
                                    item_dict = output_item.model_dump()
                                elif hasattr(output_item, "__dict__"):
                                    item_dict = output_item.__dict__
                                else:
                                    item_dict = output_item if isinstance(output_item, dict) else {}
                                cleaned_item = self._clean_output_item(item_dict, "function_call")
                                messages.append(cleaned_item)

                    function_calls_for_event = [{k: v for k, v in fc.items() if k != "call_id"} for fc in function_calls]

                    if hasattr(tool_executor, "_tool_calls"):
                        tool_executor._tool_calls.extend(function_calls_for_event)

                    events.append(ToolCallsEvent(tool_calls=function_calls_for_event))

                    if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls and len(function_calls) > 1:
                        first_function_call = function_calls[0]
                        function_name = first_function_call["function"]["name"]

                        tool_outputs = self._execute_tools_parallel_sync(tool_executor, [first_function_call])
                        tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                        first_item_found = False
                        for i, output_item in enumerate(output):
                            item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                            if item_type == "function_call" and not first_item_found:
                                if i > 0:
                                    prev_item = output[i - 1]
                                    prev_item_type = getattr(prev_item, "type", None) or (
                                        prev_item.get("type") if hasattr(prev_item, "get") else None
                                    )
                                    if prev_item_type == "reasoning":
                                        if hasattr(prev_item, "model_dump"):
                                            prev_item_dict = prev_item.model_dump()
                                        elif hasattr(prev_item, "__dict__"):
                                            prev_item_dict = prev_item.__dict__
                                        else:
                                            prev_item_dict = prev_item if isinstance(prev_item, dict) else {}
                                        cleaned_prev_item = self._clean_output_item(prev_item_dict, "reasoning")
                                        messages.append(cleaned_prev_item)

                                item_id = getattr(output_item, "id", None) or (output_item.get("id") if hasattr(output_item, "get") else None)
                                matching_call_data = None
                                for call_data in tool_executor._pending_function_calls:
                                    call_item = call_data["item"]
                                    call_item_id = getattr(call_item, "id", None) or (call_item.get("id") if hasattr(call_item, "get") else None)
                                    if call_item_id == item_id:
                                        matching_call_data = call_data
                                        break

                                if matching_call_data:
                                    item_call_id = matching_call_data["item_call_id"]

                                    if hasattr(output_item, "model_dump"):
                                        item_dict = output_item.model_dump()
                                    elif hasattr(output_item, "__dict__"):
                                        item_dict = output_item.__dict__
                                    else:
                                        item_dict = output_item if isinstance(output_item, dict) else {}

                                    cleaned_item = self._clean_output_item(item_dict, "function_call")
                                    messages.append(cleaned_item)

                                    messages.append(
                                        {
                                            "type": "function_call_output",
                                            "call_id": item_call_id,
                                            "output": json.dumps({function_name: tool_result}),
                                        }
                                    )

                                    if hasattr(tool_executor, "_tool_outputs"):
                                        tool_executor._tool_outputs[function_name] = tool_result
                                    if hasattr(tool_executor, "_tool_calls"):
                                        first_function_call_clean = {k: v for k, v in first_function_call.items() if k != "call_id"}
                                        tool_executor._tool_calls.append(first_function_call_clean)

                                    first_item_found = True
                                    break

                        events.append(ToolOutputsEvent(tool_outputs=tool_outputs))

                        tool_executor._pending_function_calls = []

                        events.append(True)
                        return events

                    tool_outputs = self._execute_tools_parallel_sync(tool_executor, function_calls)

                    for i, output_item in enumerate(output):
                        item_type = getattr(output_item, "type", None) or (output_item.get("type") if hasattr(output_item, "get") else None)
                        if item_type == "function_call":
                            item_id = getattr(output_item, "id", None) or (output_item.get("id") if hasattr(output_item, "get") else None)
                            matching_call_data = None
                            for call_data in tool_executor._pending_function_calls:
                                call_item = call_data["item"]
                                call_item_id = getattr(call_item, "id", None) or (call_item.get("id") if hasattr(call_item, "get") else None)
                                if call_item_id == item_id:
                                    matching_call_data = call_data
                                    break

                            if matching_call_data:
                                if i > 0:
                                    prev_item = output[i - 1]
                                    prev_item_type = getattr(prev_item, "type", None) or (
                                        prev_item.get("type") if hasattr(prev_item, "get") else None
                                    )
                                    if prev_item_type == "reasoning":
                                        prev_item_id = getattr(prev_item, "id", None) or (prev_item.get("id") if hasattr(prev_item, "get") else None)
                                        if not any(
                                            getattr(msg, "id", None) == prev_item_id
                                            or (msg.get("id") if hasattr(msg, "get") else None) == prev_item_id
                                            for msg in messages
                                        ):
                                            if hasattr(prev_item, "model_dump"):
                                                prev_item_dict = prev_item.model_dump()
                                            elif hasattr(prev_item, "__dict__"):
                                                prev_item_dict = prev_item.__dict__
                                            else:
                                                prev_item_dict = prev_item if isinstance(prev_item, dict) else {}
                                            cleaned_prev_item = self._clean_output_item(prev_item_dict, "reasoning")
                                            messages.append(cleaned_prev_item)

                                function_call = matching_call_data["function_call"]
                                item = matching_call_data["item"]
                                item_call_id = matching_call_data["item_call_id"]

                                function_name = function_call["function"]["name"]
                                tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                                if hasattr(tool_executor, "_tool_outputs"):
                                    tool_executor._tool_outputs[function_name] = tool_result

                                if not any(
                                    getattr(msg, "id", None) == item_id or (msg.get("id") if hasattr(msg, "get") else None) == item_id
                                    for msg in messages
                                ):
                                    if hasattr(output_item, "model_dump"):
                                        item_dict = output_item.model_dump()
                                    elif hasattr(output_item, "__dict__"):
                                        item_dict = output_item.__dict__
                                    else:
                                        item_dict = output_item if isinstance(output_item, dict) else {}
                                    cleaned_item = self._clean_output_item(item_dict, "function_call")
                                    messages.append(cleaned_item)

                                messages.append(
                                    {
                                        "type": "function_call_output",
                                        "call_id": item_call_id,
                                        "output": json.dumps({function_name: tool_result}),
                                    }
                                )

                                if hasattr(tool_executor, "_tool_calls"):
                                    function_call_clean = {k: v for k, v in function_call.items() if k != "call_id"}
                                    tool_executor._tool_calls.append(function_call_clean)

                    events.append(ToolOutputsEvent(tool_outputs=tool_outputs))

                    tool_executor._pending_function_calls = []

                if hasattr(tool_executor, "_streaming_content_buffer"):
                    tool_executor._streaming_content_buffer = ""

                if usage:
                    if hasattr(usage, "__dict__"):
                        usage_dict = usage.__dict__
                    elif hasattr(usage, "model_dump"):
                        usage_dict = usage.model_dump()
                    else:
                        usage_dict = usage if isinstance(usage, dict) else {}

                    self._accumulate_metrics(usage_dict)

                events.append(ResponseCompletedEvent(usage=usage_dict))
                events.append(True)
            else:
                if hasattr(tool_executor, "_streaming_content_buffer"):
                    tool_executor._streaming_content_buffer = ""

                if usage:
                    if hasattr(usage, "__dict__"):
                        usage_dict = usage.__dict__
                    elif hasattr(usage, "model_dump"):
                        usage_dict = usage.model_dump()
                    else:
                        usage_dict = usage if isinstance(usage, dict) else {}

                    self._accumulate_metrics(usage_dict)

                events.append(ResponseCompletedEvent(usage=usage_dict))
                events.append(False)

        elif event_type == "error":
            error = getattr(event, "error", {})
            error_msg = error.get("message", "Unknown error") if isinstance(error, dict) else str(error)
            raise Exception(f"Error from OpenAI Responses API: {error_msg}")

        return events

    def _format_files_for_openai_response(self, message: dict[str, Any]) -> dict[str, Any]:
        from hypertic.utils.files import File, FileType

        if "_file_objects" not in message:
            return message

        file_objects: list[File] = message.get("_file_objects", [])
        if not file_objects:
            return message

        content: list[dict[str, Any]] = []
        message_content = message.get("content")
        if message_content:
            if isinstance(message_content, str):
                content.append({"type": "input_text", "text": message_content})
            elif isinstance(message_content, list):
                content = message_content

        for file_obj in file_objects:
            if file_obj.file_type == FileType.IMAGE:
                if file_obj.url:
                    content.append({"type": "input_image", "image_url": file_obj.url})
                else:
                    base64_data = file_obj.to_base64()
                    if base64_data:
                        mime_type = file_obj.mime_type or "image/jpeg"
                        content.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:{mime_type};base64,{base64_data}",
                            }
                        )
            elif file_obj.file_type == FileType.DOCUMENT:
                if file_obj.url:
                    content.append({"type": "input_file", "file_url": file_obj.url})
                else:
                    base64_data = file_obj.to_base64()
                    if base64_data:
                        mime_type = file_obj.mime_type or "application/pdf"
                        filename = file_obj.filename or str(file_obj.filepath).split("/")[-1]
                        data_url = f"data:{mime_type};base64,{base64_data}"
                        content.append({"type": "input_file", "filename": filename, "file_data": data_url})
            else:
                from hypertic.utils.log import get_logger

                logger = get_logger(__name__)
                logger.warning(
                    f"File type {file_obj.file_type} not supported in OpenAI Response API "
                    f"(only images and documents are supported): {file_obj.url or file_obj.filepath}"
                )

        message["content"] = content
        message.pop("_file_objects", None)
        message.pop("files", None)
        return message
