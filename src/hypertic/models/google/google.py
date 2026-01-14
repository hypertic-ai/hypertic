import base64
import json
from dataclasses import dataclass, field
from os import getenv
from typing import Any

from hypertic.models.base import Base
from hypertic.models.events import (
    ContentEvent,
    FinishReasonEvent,
    ReasoningEvent,
    ResponseCompletedEvent,
    ToolCallsEvent,
    ToolOutputsEvent,
)

try:
    from google.genai import Client as GeminiClient
    from google.genai.errors import ClientError, ServerError
    from google.genai.types import (
        Content,
        FunctionDeclaration,
        GenerateContentConfig,
        Part,
        Tool,
    )

except ImportError as err:
    raise ImportError("`google-genai` not installed. Please install it using `pip install google-genai`") from err


@dataclass
class GoogleAI(Base):
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None

    thinking_tokens: int | None = None
    thinking: bool | None = None

    client: Any = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("GEMINI_API_KEY")

        if self.api_key is not None:
            self.client = GeminiClient(api_key=self.api_key)
        else:
            self.client = GeminiClient()

    def _normalize_google_usage(self, usage: dict[str, Any]) -> dict[str, Any]:
        if not usage:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        normalized = usage.copy()

        input_tokens = None
        output_tokens = None
        total_tokens = None

        if "prompt_token_count" in usage:
            input_tokens = usage["prompt_token_count"]
        elif "promptTokenCount" in usage:
            input_tokens = usage["promptTokenCount"]

        if "candidates_token_count" in usage:
            output_tokens = usage["candidates_token_count"]
        elif "candidatesTokenCount" in usage:
            output_tokens = usage["candidatesTokenCount"]

        if "total_token_count" in usage:
            total_tokens = usage["total_token_count"]
        elif "totalTokenCount" in usage:
            total_tokens = usage["totalTokenCount"]

        normalized["input_tokens"] = input_tokens if input_tokens is not None else 0
        normalized["output_tokens"] = output_tokens if output_tokens is not None else 0
        normalized["total_tokens"] = total_tokens if total_tokens is not None else (normalized["input_tokens"] + normalized["output_tokens"])

        return normalized

    def _convert_tools_to_google(self, tools: list[dict[str, Any]]) -> list[Tool]:
        google_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                function = tool["function"]
                parameters = function["parameters"].copy() if function.get("parameters") else {}
                if "additionalProperties" in parameters:
                    del parameters["additionalProperties"]
                if "$schema" in parameters:
                    del parameters["$schema"]

                google_tool = Tool(
                    function_declarations=[
                        FunctionDeclaration(
                            name=function["name"],
                            description=function.get("description", ""),
                            parameters_json_schema=parameters if parameters else None,
                        )
                    ]
                )
                google_tools.append(google_tool)

        return google_tools

    def _convert_messages_to_google_format(self, messages: list[dict[str, Any]]) -> list[Content]:
        """Convert messages to Google Content format."""
        contents = []

        for message in messages:
            if message["role"] == "user":
                if isinstance(message["content"], list):
                    parts = []
                    for item in message["content"]:
                        if isinstance(item, str):
                            parts.append(Part(text=item))
                        elif isinstance(item, dict):
                            if "inline_data" in item:
                                parts.append(Part(inline_data=item["inline_data"]))
                            elif "file_data" in item:
                                parts.append(Part(file_data=item["file_data"]))
                            else:
                                parts.append(Part(text=str(item)))
                        else:
                            parts.append(Part(text=str(item)))
                    contents.append(Content(role="user", parts=parts))
                else:
                    parts = [Part(text=message["content"])]
                    contents.append(Content(role="user", parts=parts))
            elif message["role"] == "assistant":
                if isinstance(message["content"], str):
                    if message["content"].startswith("{"):
                        try:
                            parsed_content = json.loads(message["content"])
                            parts = []
                            for part_data in parsed_content.get("parts", []):
                                if "text" in part_data:
                                    parts.append(Part(text=part_data["text"]))
                                elif "functionCall" in part_data:
                                    part_kwargs = {"function_call": part_data["functionCall"]}
                                    if "thoughtSignature" in part_data:
                                        thought_sig_str = part_data["thoughtSignature"]
                                        if isinstance(thought_sig_str, str) and thought_sig_str not in [
                                            "skip_thought_signature_validator",
                                            "context_engineering_is_the_way_to_go",
                                        ]:
                                            try:
                                                part_kwargs["thought_signature"] = base64.b64decode(thought_sig_str)
                                            except Exception:
                                                part_kwargs["thought_signature"] = thought_sig_str
                                        else:
                                            part_kwargs["thought_signature"] = thought_sig_str
                                    parts.append(Part(**part_kwargs))
                            contents.append(Content(role="model", parts=parts))
                        except Exception:
                            parts = [Part(text=message["content"])]
                            contents.append(Content(role="model", parts=parts))
                    else:
                        parts = [Part(text=message["content"])]
                        contents.append(Content(role="model", parts=parts))
            elif message["role"] == "function":
                try:
                    parsed_content = json.loads(message["content"])
                    parts = []
                    for part_data in parsed_content.get("parts", []):
                        if "functionResponse" in part_data:
                            parts.append(Part(function_response=part_data["functionResponse"]))
                    contents.append(Content(role="user", parts=parts))
                except Exception:
                    parts = [Part(text=message["content"])]
                    contents.append(Content(role="user", parts=parts))

        return contents

    async def ahandle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ) -> Any | None:
        try:
            formatted_messages = [self._format_files_for_google(msg.copy()) for msg in messages]

            contents = self._convert_messages_to_google_format(formatted_messages)

            config_params: dict[str, Any] = {}

            if self.temperature is not None:
                config_params["temperature"] = self.temperature
            if self.top_p is not None:
                config_params["top_p"] = self.top_p
            if self.max_tokens is not None:
                config_params["max_output_tokens"] = self.max_tokens

            if self.thinking_tokens is not None or self.thinking is not None:
                from google.genai.types import ThinkingConfig

                thinking_config: dict[str, Any] = {}
                if self.thinking_tokens is not None:
                    thinking_config["thinking_budget"] = self.thinking_tokens
                if self.thinking is not None:
                    thinking_config["include_thoughts"] = bool(self.thinking)
                if thinking_config:
                    config_params["thinking_config"] = ThinkingConfig(**thinking_config)

            if tools:
                google_tools = self._convert_tools_to_google(tools)
                config_params["tools"] = google_tools
                if response_format is not None:
                    raise ValueError("Google AI does not support structured output (response_format) when tools are present.")

            if response_format is not None and not tools:
                config_params["response_mime_type"] = "application/json"
                config_params["response_schema"] = response_format

            config = GenerateContentConfig(**config_params)

            response = await self.client.aio.models.generate_content(model=self.model, contents=contents, config=config)

            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                parts = candidate.content.parts

                if parts is None:
                    return self._create_llm_response(
                        "No content generated (likely hit token limit)",
                        {"input_tokens": 0, "output_tokens": 0},
                        [],
                    )

                function_calls = [part for part in parts if part.function_call]

                if function_calls:
                    usage_raw = response.usage_metadata.model_dump() if response.usage_metadata else {}
                    usage = self._normalize_google_usage(usage_raw)
                    self._accumulate_metrics(usage)

                    tool_calls_for_storage: list[dict[str, Any]] = []
                    parts_list_for_message = []
                    for idx, part in enumerate(function_calls):
                        function_call = part.function_call
                        function_name = function_call.name
                        function_args = function_call.args or {}

                        thought_sig = None
                        if hasattr(part, "thought_signature") and part.thought_signature is not None:
                            thought_sig = base64.b64encode(part.thought_signature).decode("ascii")
                        elif hasattr(part, "thoughtSignature") and part.thoughtSignature is not None:
                            thought_sig = base64.b64encode(part.thoughtSignature).decode("ascii")

                        tool_call_dict = {
                            "id": f"call_{len(tool_calls_for_storage)}",
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": json.dumps(function_args),
                            },
                        }

                        if thought_sig:
                            tool_call_dict["_thought_signature"] = thought_sig

                        clean_tool_call = {k: v for k, v in tool_call_dict.items() if k != "_thought_signature"}
                        tool_calls_for_storage.append(clean_tool_call)

                        part_dict = {"functionCall": function_call.model_dump()}
                        if thought_sig:
                            part_dict["thoughtSignature"] = thought_sig
                        elif idx == 0:
                            part_dict["thoughtSignature"] = "skip_thought_signature_validator"
                        parts_list_for_message.append(part_dict)

                    messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps({"parts": parts_list_for_message}),
                        }
                    )

                    if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                        if function_calls:
                            first_part = function_calls[0]
                            first_function_call = first_part.function_call
                            function_name = first_function_call.name
                            function_args = first_function_call.args or {}

                            if parts_list_for_message:
                                messages[-1]["content"] = json.dumps({"parts": [parts_list_for_message[0]]})
                            else:
                                messages[-1]["content"] = json.dumps({"parts": [{"functionCall": first_function_call.model_dump()}]})

                            tool_call_dict = {
                                "id": tool_calls_for_storage[0]["id"] if tool_calls_for_storage else "call_0",
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": json.dumps(function_args),
                                },
                            }
                            results = await self._execute_tools_parallel_async(tool_executor, [tool_call_dict])
                            tool_result = results.get(function_name, f"Tool '{function_name}' execution failed")

                            tool_outputs = {function_name: tool_result}

                            first_tool_call_for_storage = {
                                "id": tool_calls_for_storage[0]["id"] if tool_calls_for_storage else "call_0",
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": json.dumps(function_args),
                                },
                            }

                            messages.append(
                                {
                                    "role": "function",
                                    "content": json.dumps(
                                        {
                                            "parts": [
                                                {
                                                    "functionResponse": {
                                                        "name": function_name,
                                                        "response": {"result": str(tool_result)},
                                                    }
                                                }
                                            ]
                                        }
                                    ),
                                }
                            )

                            if hasattr(tool_executor, "_tool_outputs"):
                                tool_executor._tool_outputs.update(tool_outputs)
                            if hasattr(tool_executor, "_tool_calls"):
                                tool_executor._tool_calls.append(first_tool_call_for_storage)

                            return None
                    else:
                        tool_outputs = await self._execute_tools_parallel_async(tool_executor, tool_calls_for_storage)

                        for part in function_calls:
                            function_call = part.function_call
                            function_name = function_call.name
                            tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                            messages.append(
                                {
                                    "role": "function",
                                    "content": json.dumps(
                                        {
                                            "parts": [
                                                {
                                                    "functionResponse": {
                                                        "name": function_name,
                                                        "response": {"result": str(tool_result)},
                                                    }
                                                }
                                            ]
                                        }
                                    ),
                                }
                            )

                    if hasattr(tool_executor, "_tool_outputs"):
                        tool_executor._tool_outputs.update(tool_outputs)
                    if hasattr(tool_executor, "_tool_calls"):
                        tool_executor._tool_calls.extend(tool_calls_for_storage)

                    return None
                else:
                    response_text = ""
                    reasoning_content = ""
                    tool_calls: list[dict[str, Any]] = []
                    for part in parts:
                        if part.text:
                            if hasattr(part, "thought") and part.thought:
                                reasoning_content += part.text
                            else:
                                response_text += part.text
                        elif part.function_call:
                            tool_calls.append(
                                {
                                    "id": f"call_{len(tool_calls)}",
                                    "type": "function",
                                    "function": {
                                        "name": part.function_call.name,
                                        "arguments": json.dumps(part.function_call.args or {}),
                                    },
                                }
                            )

                    usage_raw = response.usage_metadata.model_dump() if response.usage_metadata else {}
                    usage = self._normalize_google_usage(usage_raw)

                    self._accumulate_metrics(usage)

                    cumulative_metrics = self._get_cumulative_metrics()

                    finish_reason = "stop"
                    if candidate.finish_reason:
                        finish_reason = candidate.finish_reason

                    tool_outputs = {}
                    if hasattr(tool_executor, "_tool_outputs"):
                        tool_outputs = tool_executor._tool_outputs.copy()

                    stored_tool_calls = []
                    if hasattr(tool_executor, "_tool_calls"):
                        stored_tool_calls = tool_executor._tool_calls.copy()

                    params = {}
                    if self.temperature is not None:
                        params["temperature"] = self.temperature
                    if self.top_p is not None:
                        params["top_p"] = self.top_p
                    if self.max_tokens is not None:
                        params["max_tokens"] = self.max_tokens
                    if self.thinking_tokens is not None:
                        params["thinking_tokens"] = self.thinking_tokens
                    if self.thinking is not None:
                        params["thinking"] = self.thinking

                    usage_info = {
                        "model": self.model,
                        "params": params,
                        "finish_reason": finish_reason,
                        "input_tokens": cumulative_metrics.input_tokens,
                        "output_tokens": cumulative_metrics.output_tokens,
                    }

                    if reasoning_content:
                        usage_info["reasoning_content"] = reasoning_content

                    return self._create_llm_response(response_text, usage_info, stored_tool_calls, tool_outputs)
            else:
                return self._create_llm_response("No valid response from Google AI", {"input_tokens": 0, "output_tokens": 0}, [])

        except (ClientError, ServerError) as e:
            raise Exception(f"Google API Error: {e!s}") from e
        except Exception as e:
            raise Exception(f"Unexpected error in Google handler: {e!s}") from e

    async def ahandle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        function_calls = []
        function_call_parts = []
        current_text = ""

        try:
            formatted_messages = [self._format_files_for_google(msg.copy()) for msg in messages]

            contents = self._convert_messages_to_google_format(formatted_messages)

            config_params: dict[str, Any] = {}

            if self.temperature is not None:
                config_params["temperature"] = self.temperature
            if self.top_p is not None:
                config_params["top_p"] = self.top_p
            if self.max_tokens is not None:
                config_params["max_output_tokens"] = self.max_tokens

            if self.thinking_tokens is not None or self.thinking is not None:
                from google.genai.types import ThinkingConfig

                thinking_config: dict[str, Any] = {}
                if self.thinking_tokens is not None:
                    thinking_config["thinking_budget"] = self.thinking_tokens
                if self.thinking is not None:
                    thinking_config["include_thoughts"] = bool(self.thinking)
                if thinking_config:
                    config_params["thinking_config"] = ThinkingConfig(**thinking_config)

            if tools:
                google_tools = self._convert_tools_to_google(tools)
                config_params["tools"] = google_tools
                if response_format is not None:
                    raise ValueError("Google AI does not support structured output (response_format) when tools are present.")

            if response_format is not None and not tools:
                config_params["response_mime_type"] = "application/json"
                config_params["response_schema"] = response_format

            config = GenerateContentConfig(**config_params)

            async_stream = await self.client.aio.models.generate_content_stream(model=self.model, contents=contents, config=config)

            usage = None

            async for chunk in async_stream:
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata is not None:
                    usage_raw = chunk.usage_metadata.model_dump() if hasattr(chunk.usage_metadata, "model_dump") else chunk.usage_metadata
                    usage = self._normalize_google_usage(usage_raw if isinstance(usage_raw, dict) else {})

                if chunk.candidates and len(chunk.candidates) > 0:
                    candidate = chunk.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                if hasattr(part, "thought") and part.thought:
                                    reasoning_chunk = part.text
                                    if reasoning_chunk is not None and reasoning_chunk.strip():
                                        yield ReasoningEvent(reasoning=reasoning_chunk)
                                else:
                                    text_chunk = part.text
                                    if text_chunk is not None and text_chunk.strip():
                                        current_text += text_chunk
                                        yield ContentEvent(content=text_chunk)

                            elif part.function_call:
                                function_calls.append(part.function_call)
                                function_call_parts.append(part)

            if function_calls:
                tool_calls_formatted: list[dict[str, Any]] = []
                parts_list_for_message = []
                for _idx, (function_call, part) in enumerate(zip(function_calls, function_call_parts, strict=False)):
                    thought_sig = None
                    if hasattr(part, "thought_signature") and part.thought_signature is not None:
                        thought_sig = base64.b64encode(part.thought_signature).decode("utf-8")
                    elif hasattr(part, "thoughtSignature") and part.thoughtSignature is not None:
                        thought_sig = base64.b64encode(part.thoughtSignature).decode("utf-8")

                    tool_call_dict = {
                        "id": f"call_{len(tool_calls_formatted)}",
                        "type": "function",
                        "function": {
                            "name": function_call.name,
                            "arguments": json.dumps(function_call.args or {}),
                        },
                    }

                    if thought_sig:
                        tool_call_dict["_thought_signature"] = thought_sig

                    clean_tool_call = {k: v for k, v in tool_call_dict.items() if k != "_thought_signature"}
                    tool_calls_formatted.append(clean_tool_call)

                    part_dict = {"functionCall": function_call.model_dump()}
                    if thought_sig:
                        part_dict["thoughtSignature"] = thought_sig
                    parts_list_for_message.append(part_dict)

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if parts_list_for_message:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": json.dumps({"parts": [parts_list_for_message[0]]}),
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": json.dumps({"parts": parts_list_for_message}),
                            }
                        )
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps({"parts": parts_list_for_message}),
                        }
                    )

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    yield ToolCallsEvent(tool_calls=[tool_calls_formatted[0]] if tool_calls_formatted else [])
                else:
                    yield ToolCallsEvent(tool_calls=tool_calls_formatted)

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if function_calls and tool_calls_formatted:
                        first_function_call = function_calls[0]
                        first_tool_call_formatted = tool_calls_formatted[0]
                        function_name = first_function_call.name

                        tool_call_dict = first_tool_call_formatted
                        results = await self._execute_tools_parallel_async(tool_executor, [tool_call_dict])
                        tool_result = results.get(function_name, f"Tool '{function_name}' execution failed")

                        tool_outputs = {function_name: tool_result}

                        messages.append(
                            {
                                "role": "function",
                                "content": json.dumps(
                                    {
                                        "parts": [
                                            {
                                                "functionResponse": {
                                                    "name": function_name,
                                                    "response": {"result": str(tool_result)},
                                                }
                                            }
                                        ]
                                    }
                                ),
                            }
                        )

                        if hasattr(tool_executor, "_tool_outputs"):
                            tool_executor._tool_outputs.update(tool_outputs)
                        if hasattr(tool_executor, "_tool_calls"):
                            tool_executor._tool_calls.append(first_tool_call_formatted)

                        yield ToolOutputsEvent(tool_outputs=tool_outputs)

                        if usage:
                            if hasattr(usage, "__dict__"):
                                usage_dict = usage.__dict__
                            elif hasattr(usage, "model_dump"):
                                usage_dict = usage.model_dump()
                            else:
                                usage_dict = usage if isinstance(usage, dict) else {}
                            self._accumulate_metrics(usage_dict)
                            yield ResponseCompletedEvent(usage=usage)

                        yield True
                        return
                else:
                    tool_outputs = await self._execute_tools_parallel_async(tool_executor, tool_calls_formatted)

                    for function_call in function_calls:
                        function_name = function_call.name
                        tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                        messages.append(
                            {
                                "role": "function",
                                "content": json.dumps(
                                    {
                                        "parts": [
                                            {
                                                "functionResponse": {
                                                    "name": function_name,
                                                    "response": {"result": str(tool_result)},
                                                }
                                            }
                                        ]
                                    }
                                ),
                            }
                        )

                    yield ToolOutputsEvent(tool_outputs=tool_outputs)

                if usage:
                    if hasattr(usage, "__dict__"):
                        usage_dict = usage.__dict__
                    elif hasattr(usage, "model_dump"):
                        usage_dict = usage.model_dump()
                    else:
                        usage_dict = usage if isinstance(usage, dict) else {}
                    self._accumulate_metrics(usage_dict)
                    yield ResponseCompletedEvent(usage=usage)

                yield True
                return

            if usage:
                if hasattr(usage, "__dict__"):
                    usage_dict = usage.__dict__
                elif hasattr(usage, "model_dump"):
                    usage_dict = usage.model_dump()
                else:
                    usage_dict = usage if isinstance(usage, dict) else {}
                self._accumulate_metrics(usage_dict)
                yield ResponseCompletedEvent(usage=usage)

            yield FinishReasonEvent(finish_reason="stop")
            yield False

        except (ClientError, ServerError) as e:
            yield ContentEvent(content=f"Google API Error: {e!s}")
        except Exception as e:
            yield ContentEvent(content=f"Unexpected error in Google streaming: {e!s}")

    def handle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ) -> Any | None:
        try:
            formatted_messages = [self._format_files_for_google(msg.copy()) for msg in messages]

            contents = self._convert_messages_to_google_format(formatted_messages)

            config_params: dict[str, Any] = {}

            if self.temperature is not None:
                config_params["temperature"] = self.temperature
            if self.top_p is not None:
                config_params["top_p"] = self.top_p
            if self.max_tokens is not None:
                config_params["max_output_tokens"] = self.max_tokens

            if self.thinking_tokens is not None or self.thinking is not None:
                from google.genai.types import ThinkingConfig

                thinking_config: dict[str, Any] = {}
                if self.thinking_tokens is not None:
                    thinking_config["thinking_budget"] = self.thinking_tokens
                if self.thinking is not None:
                    thinking_config["include_thoughts"] = bool(self.thinking)
                if thinking_config:
                    config_params["thinking_config"] = ThinkingConfig(**thinking_config)

            if tools:
                google_tools = self._convert_tools_to_google(tools)
                config_params["tools"] = google_tools
                if response_format is not None:
                    raise ValueError("Google AI does not support structured output (response_format) when tools are present.")

            if response_format is not None and not tools:
                config_params["response_mime_type"] = "application/json"
                config_params["response_schema"] = response_format

            config = GenerateContentConfig(**config_params)

            response = self.client.models.generate_content(model=self.model, contents=contents, config=config)

            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                parts = candidate.content.parts

                if parts is None:
                    return self._create_llm_response(
                        "No content generated (likely hit token limit)",
                        {"input_tokens": 0, "output_tokens": 0},
                        [],
                    )

                function_calls = [part for part in parts if part.function_call]

                if function_calls:
                    usage_raw = response.usage_metadata.model_dump() if response.usage_metadata else {}
                    usage = self._normalize_google_usage(usage_raw)
                    self._accumulate_metrics(usage)

                    tool_calls_for_storage: list[dict[str, Any]] = []
                    parts_list_for_message = []
                    for idx, part in enumerate(function_calls):
                        function_call = part.function_call
                        function_name = function_call.name
                        function_args = function_call.args or {}

                        thought_sig = None
                        if hasattr(part, "thought_signature") and part.thought_signature is not None:
                            thought_sig = base64.b64encode(part.thought_signature).decode("ascii")
                        elif hasattr(part, "thoughtSignature") and part.thoughtSignature is not None:
                            thought_sig = base64.b64encode(part.thoughtSignature).decode("ascii")

                        tool_call_dict = {
                            "id": f"call_{len(tool_calls_for_storage)}",
                            "type": "function",
                            "function": {
                                "name": function_name,
                                "arguments": json.dumps(function_args),
                            },
                        }

                        if thought_sig:
                            tool_call_dict["_thought_signature"] = thought_sig

                        clean_tool_call = {k: v for k, v in tool_call_dict.items() if k != "_thought_signature"}
                        tool_calls_for_storage.append(clean_tool_call)

                        part_dict = {"functionCall": function_call.model_dump()}
                        if thought_sig:
                            part_dict["thoughtSignature"] = thought_sig
                        elif idx == 0:
                            part_dict["thoughtSignature"] = "skip_thought_signature_validator"
                        parts_list_for_message.append(part_dict)

                    messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps({"parts": parts_list_for_message}),
                        }
                    )

                    if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                        if function_calls:
                            first_part = function_calls[0]
                            first_function_call = first_part.function_call
                            function_name = first_function_call.name
                            function_args = first_function_call.args or {}

                            if parts_list_for_message:
                                messages[-1]["content"] = json.dumps({"parts": [parts_list_for_message[0]]})
                            else:
                                messages[-1]["content"] = json.dumps({"parts": [{"functionCall": first_function_call.model_dump()}]})

                            tool_call_dict = {
                                "id": tool_calls_for_storage[0]["id"] if tool_calls_for_storage else "call_0",
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": json.dumps(function_args),
                                },
                            }
                            results = self._execute_tools_parallel_sync(tool_executor, [tool_call_dict])
                            tool_result = results.get(function_name, f"Tool '{function_name}' execution failed")

                            tool_outputs = {function_name: tool_result}

                            first_tool_call_for_storage = {
                                "id": tool_calls_for_storage[0]["id"] if tool_calls_for_storage else "call_0",
                                "type": "function",
                                "function": {
                                    "name": function_name,
                                    "arguments": json.dumps(function_args),
                                },
                            }

                            messages.append(
                                {
                                    "role": "function",
                                    "content": json.dumps(
                                        {
                                            "parts": [
                                                {
                                                    "functionResponse": {
                                                        "name": function_name,
                                                        "response": {"result": str(tool_result)},
                                                    }
                                                }
                                            ]
                                        }
                                    ),
                                }
                            )

                            if hasattr(tool_executor, "_tool_outputs"):
                                tool_executor._tool_outputs.update(tool_outputs)
                            if hasattr(tool_executor, "_tool_calls"):
                                tool_executor._tool_calls.append(first_tool_call_for_storage)

                            return None
                    else:
                        tool_outputs = self._execute_tools_parallel_sync(tool_executor, tool_calls_for_storage)

                        for part in function_calls:
                            function_call = part.function_call
                            function_name = function_call.name
                            tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                            messages.append(
                                {
                                    "role": "function",
                                    "content": json.dumps(
                                        {
                                            "parts": [
                                                {
                                                    "functionResponse": {
                                                        "name": function_name,
                                                        "response": {"result": str(tool_result)},
                                                    }
                                                }
                                            ]
                                        }
                                    ),
                                }
                            )

                    if hasattr(tool_executor, "_tool_outputs"):
                        tool_executor._tool_outputs.update(tool_outputs)
                    if hasattr(tool_executor, "_tool_calls"):
                        tool_executor._tool_calls.extend(tool_calls_for_storage)

                    return None
                else:
                    response_text = ""
                    reasoning_content = ""
                    tool_calls: list[dict[str, Any]] = []
                    for part in parts:
                        if part.text:
                            if hasattr(part, "thought") and part.thought:
                                reasoning_content += part.text
                            else:
                                response_text += part.text
                        elif part.function_call:
                            tool_calls.append(
                                {
                                    "id": f"call_{len(tool_calls)}",
                                    "type": "function",
                                    "function": {
                                        "name": part.function_call.name,
                                        "arguments": json.dumps(part.function_call.args or {}),
                                    },
                                }
                            )

                    usage_raw = response.usage_metadata.model_dump() if response.usage_metadata else {}
                    usage = self._normalize_google_usage(usage_raw)

                    self._accumulate_metrics(usage)

                    cumulative_metrics = self._get_cumulative_metrics()

                    finish_reason = "stop"
                    if candidate.finish_reason:
                        finish_reason = candidate.finish_reason

                    tool_outputs = {}
                    if hasattr(tool_executor, "_tool_outputs"):
                        tool_outputs = tool_executor._tool_outputs.copy()

                    stored_tool_calls = []
                    if hasattr(tool_executor, "_tool_calls"):
                        stored_tool_calls = tool_executor._tool_calls.copy()

                    params = {}
                    if self.temperature is not None:
                        params["temperature"] = self.temperature
                    if self.top_p is not None:
                        params["top_p"] = self.top_p
                    if self.max_tokens is not None:
                        params["max_tokens"] = self.max_tokens
                    if self.thinking_tokens is not None:
                        params["thinking_tokens"] = self.thinking_tokens
                    if self.thinking is not None:
                        params["thinking"] = self.thinking

                    usage_info = {
                        "model": self.model,
                        "params": params,
                        "finish_reason": finish_reason,
                        "input_tokens": cumulative_metrics.input_tokens,
                        "output_tokens": cumulative_metrics.output_tokens,
                    }

                    if reasoning_content:
                        usage_info["reasoning_content"] = reasoning_content

                    return self._create_llm_response(response_text, usage_info, stored_tool_calls, tool_outputs)
            else:
                return self._create_llm_response("No valid response from Google AI", {"input_tokens": 0, "output_tokens": 0}, [])

        except (ClientError, ServerError) as e:
            raise Exception(f"Google API Error: {e!s}") from e
        except Exception as e:
            raise Exception(f"Unexpected error in Google handler: {e!s}") from e

    def handle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        function_calls = []
        function_call_parts = []
        current_text = ""

        try:
            formatted_messages = [self._format_files_for_google(msg.copy()) for msg in messages]

            contents = self._convert_messages_to_google_format(formatted_messages)

            config_params: dict[str, Any] = {}

            if self.temperature is not None:
                config_params["temperature"] = self.temperature
            if self.top_p is not None:
                config_params["top_p"] = self.top_p
            if self.max_tokens is not None:
                config_params["max_output_tokens"] = self.max_tokens

            if self.thinking_tokens is not None or self.thinking is not None:
                from google.genai.types import ThinkingConfig

                thinking_config: dict[str, Any] = {}
                if self.thinking_tokens is not None:
                    thinking_config["thinking_budget"] = self.thinking_tokens
                if self.thinking is not None:
                    thinking_config["include_thoughts"] = bool(self.thinking)
                if thinking_config:
                    config_params["thinking_config"] = ThinkingConfig(**thinking_config)

            if tools:
                google_tools = self._convert_tools_to_google(tools)
                config_params["tools"] = google_tools
                if response_format is not None:
                    raise ValueError("Google AI does not support structured output (response_format) when tools are present.")

            if response_format is not None and not tools:
                config_params["response_mime_type"] = "application/json"
                config_params["response_schema"] = response_format

            config = GenerateContentConfig(**config_params)

            async_stream = self.client.models.generate_content_stream(model=self.model, contents=contents, config=config)

            usage = None

            for chunk in async_stream:
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata is not None:
                    usage_raw = chunk.usage_metadata.model_dump() if hasattr(chunk.usage_metadata, "model_dump") else chunk.usage_metadata
                    usage = self._normalize_google_usage(usage_raw if isinstance(usage_raw, dict) else {})

                if chunk.candidates and len(chunk.candidates) > 0:
                    candidate = chunk.candidates[0]
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                if hasattr(part, "thought") and part.thought:
                                    reasoning_chunk = part.text
                                    if reasoning_chunk is not None and reasoning_chunk.strip():
                                        yield ReasoningEvent(reasoning=reasoning_chunk)
                                else:
                                    text_chunk = part.text
                                    if text_chunk is not None and text_chunk.strip():
                                        current_text += text_chunk
                                        yield ContentEvent(content=text_chunk)

                            elif part.function_call:
                                function_calls.append(part.function_call)
                                function_call_parts.append(part)

            if function_calls:
                tool_calls_formatted: list[dict[str, Any]] = []
                parts_list_for_message = []
                for _idx, (function_call, part) in enumerate(zip(function_calls, function_call_parts, strict=False)):
                    thought_sig = None
                    if hasattr(part, "thought_signature") and part.thought_signature is not None:
                        thought_sig = base64.b64encode(part.thought_signature).decode("utf-8")
                    elif hasattr(part, "thoughtSignature") and part.thoughtSignature is not None:
                        thought_sig = base64.b64encode(part.thoughtSignature).decode("utf-8")

                    tool_call_dict = {
                        "id": f"call_{len(tool_calls_formatted)}",
                        "type": "function",
                        "function": {
                            "name": function_call.name,
                            "arguments": json.dumps(function_call.args or {}),
                        },
                    }

                    if thought_sig:
                        tool_call_dict["_thought_signature"] = thought_sig

                    clean_tool_call = {k: v for k, v in tool_call_dict.items() if k != "_thought_signature"}
                    tool_calls_formatted.append(clean_tool_call)

                    part_dict = {"functionCall": function_call.model_dump()}
                    if thought_sig:
                        part_dict["thoughtSignature"] = thought_sig
                    parts_list_for_message.append(part_dict)

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if parts_list_for_message:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": json.dumps({"parts": [parts_list_for_message[0]]}),
                            }
                        )
                    else:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": json.dumps({"parts": parts_list_for_message}),
                            }
                        )
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps({"parts": parts_list_for_message}),
                        }
                    )

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    yield ToolCallsEvent(tool_calls=[tool_calls_formatted[0]] if tool_calls_formatted else [])
                else:
                    yield ToolCallsEvent(tool_calls=tool_calls_formatted)

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if function_calls and tool_calls_formatted:
                        first_function_call = function_calls[0]
                        first_tool_call_formatted = tool_calls_formatted[0]
                        function_name = first_function_call.name

                        tool_call_dict = first_tool_call_formatted
                        results = self._execute_tools_parallel_sync(tool_executor, [tool_call_dict])
                        tool_result = results.get(function_name, f"Tool '{function_name}' execution failed")

                        tool_outputs = {function_name: tool_result}

                        messages.append(
                            {
                                "role": "function",
                                "content": json.dumps(
                                    {
                                        "parts": [
                                            {
                                                "functionResponse": {
                                                    "name": function_name,
                                                    "response": {"result": str(tool_result)},
                                                }
                                            }
                                        ]
                                    }
                                ),
                            }
                        )

                        if hasattr(tool_executor, "_tool_outputs"):
                            tool_executor._tool_outputs.update(tool_outputs)
                        if hasattr(tool_executor, "_tool_calls"):
                            tool_executor._tool_calls.append(first_tool_call_formatted)

                        yield ToolOutputsEvent(tool_outputs=tool_outputs)

                        if usage:
                            if hasattr(usage, "__dict__"):
                                usage_dict = usage.__dict__
                            elif hasattr(usage, "model_dump"):
                                usage_dict = usage.model_dump()
                            else:
                                usage_dict = usage if isinstance(usage, dict) else {}
                            self._accumulate_metrics(usage_dict)
                            yield ResponseCompletedEvent(usage=usage)

                        yield True
                        return
                else:
                    tool_outputs = self._execute_tools_parallel_sync(tool_executor, tool_calls_formatted)

                    for function_call in function_calls:
                        function_name = function_call.name
                        tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                        messages.append(
                            {
                                "role": "function",
                                "content": json.dumps(
                                    {
                                        "parts": [
                                            {
                                                "functionResponse": {
                                                    "name": function_name,
                                                    "response": {"result": str(tool_result)},
                                                }
                                            }
                                        ]
                                    }
                                ),
                            }
                        )

                    yield ToolOutputsEvent(tool_outputs=tool_outputs)

                if usage:
                    if hasattr(usage, "__dict__"):
                        usage_dict = usage.__dict__
                    elif hasattr(usage, "model_dump"):
                        usage_dict = usage.model_dump()
                    else:
                        usage_dict = usage if isinstance(usage, dict) else {}
                    self._accumulate_metrics(usage_dict)
                    yield ResponseCompletedEvent(usage=usage)

                yield True
                return

            if usage:
                if hasattr(usage, "__dict__"):
                    usage_dict = usage.__dict__
                elif hasattr(usage, "model_dump"):
                    usage_dict = usage.model_dump()
                else:
                    usage_dict = usage if isinstance(usage, dict) else {}
                self._accumulate_metrics(usage_dict)
                yield ResponseCompletedEvent(usage=usage)

            yield FinishReasonEvent(finish_reason="stop")
            yield False

        except (ClientError, ServerError) as e:
            yield ContentEvent(content=f"Google API Error: {e!s}")
        except Exception as e:
            yield ContentEvent(content=f"Unexpected error in Google streaming: {e!s}")

    def _build_function_call_parts_with_signatures(self, all_tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        function_call_parts = []
        for idx, tool_call in enumerate(all_tool_calls):
            function_name = tool_call["function"]["name"]
            try:
                function_args = json.loads(tool_call["function"]["arguments"])
                if function_args is None:
                    function_args = {}
            except (json.JSONDecodeError, TypeError):
                function_args = {}

            part_dict: dict[str, Any] = {"functionCall": {"name": function_name, "args": function_args}}

            if "_thought_signature" in tool_call:
                thought_sig_str = tool_call["_thought_signature"]
                part_dict["thoughtSignature"] = thought_sig_str
            elif idx == 0:
                part_dict["thoughtSignature"] = "skip_thought_signature_validator"

            function_call_parts.append(part_dict)

        return function_call_parts

    def _format_files_for_google(self, message: dict[str, Any]) -> dict[str, Any]:
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
                content.append({"text": message_content})
            elif isinstance(message_content, list):
                content = message_content

        for file_obj in file_objects:
            if file_obj.file_type == FileType.IMAGE:
                if file_obj.url:
                    content.append({"file_data": {"file_uri": file_obj.url}})
                else:
                    base64_data = file_obj.to_base64()
                    if base64_data:
                        mime_type = file_obj.mime_type or "image/jpeg"
                        content.append(
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_data,
                                }
                            }
                        )
            elif file_obj.file_type == FileType.DOCUMENT:
                if file_obj.url:
                    content.append({"file_data": {"file_uri": file_obj.url}})
                else:
                    base64_data = file_obj.to_base64()
                    if base64_data:
                        mime_type = file_obj.mime_type or "application/pdf"
                        content.append(
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_data,
                                }
                            }
                        )
            elif file_obj.file_type in (FileType.AUDIO, FileType.VIDEO):
                if file_obj.url:
                    content.append({"file_data": {"file_uri": file_obj.url}})
                else:
                    base64_data = file_obj.to_base64()
                    if base64_data:
                        mime_type = file_obj.mime_type or ("audio/mpeg" if file_obj.file_type == FileType.AUDIO else "video/mp4")
                        content.append(
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": base64_data,
                                }
                            }
                        )
            else:
                from hypertic.utils.log import get_logger

                logger = get_logger(__name__)
                logger.warning(f"File type {file_obj.file_type} may not be supported in Google Gemini API: {file_obj.url or file_obj.filepath}")
                base64_data = file_obj.to_base64()
                if base64_data:
                    mime_type = file_obj.mime_type or "application/octet-stream"
                    content.append({"inline_data": {"mime_type": mime_type, "data": base64_data}})

        message["content"] = content
        message.pop("_file_objects", None)
        message.pop("files", None)
        return message
