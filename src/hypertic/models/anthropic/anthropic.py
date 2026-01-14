import json
from os import getenv
from typing import Any

from hypertic.models.base import Base

try:
    from anthropic import (
        Anthropic as AnthropicClient,
        APIConnectionError,
        APIStatusError,
        AsyncAnthropic as AsyncAnthropicClient,
        RateLimitError,
        transform_schema,
    )
    from anthropic.lib.streaming._beta_types import (
        BetaInputJsonEvent,
        BetaRawContentBlockStartEvent,
        ParsedBetaContentBlockStopEvent,
        ParsedBetaMessageStopEvent,
        ParsedBetaTextEvent,
    )
    from anthropic.types import (
        ContentBlockDeltaEvent,
        ContentBlockStartEvent,
        ContentBlockStopEvent,
        Message as AnthropicMessage,
        MessageStopEvent,
    )
except (ImportError, ModuleNotFoundError) as err:
    raise ImportError("`anthropic` not installed. Please install using `pip install anthropic`") from err

from dataclasses import dataclass, field

from hypertic.models.events import (
    ContentEvent,
    ReasoningEvent,
    ResponseCompletedEvent,
    ToolCallsEvent,
    ToolOutputsEvent,
)


@dataclass
class Anthropic(Base):
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None

    thinking_tokens: int | None = None

    async_client: Any = field(default=None, init=False)
    client: Any = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("ANTHROPIC_API_KEY")

        if self.api_key is not None:
            self.async_client = AsyncAnthropicClient(api_key=self.api_key)
            self.client = AnthropicClient(api_key=self.api_key)
        else:
            self.async_client = AsyncAnthropicClient()
            self.client = AnthropicClient()

    def _convert_tools_to_anthropic(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic tool format."""
        anthropic_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                function = tool["function"]
                parameters = function.get("parameters", {})
                if not isinstance(parameters, dict):
                    parameters = {}
                input_schema = parameters.copy()
                if "type" not in input_schema:
                    input_schema = {"type": "object", **input_schema}

                anthropic_tool = {
                    "name": function["name"],
                    "description": function["description"],
                    "input_schema": input_schema,
                }
                anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    def _prepare_structured_output(self, response_format: Any, tools: list[dict[str, Any]] | None) -> tuple[Any, bool]:
        if response_format is None:
            return None, False

        from pydantic import BaseModel

        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            if tools:
                try:
                    json_schema = response_format.model_json_schema()
                    transformed_schema = transform_schema(json_schema)
                    return {"type": "json_schema", "schema": transformed_schema}, True
                except (ImportError, AttributeError):
                    schema = response_format.model_json_schema()
                    return {"type": "json_schema", "schema": schema}, True
            else:
                return response_format, True

        return response_format, False

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
            formatted_msg = self._format_files_for_anthropic(msg.copy())
            processed_messages.append(formatted_msg)

        max_tokens = self.max_tokens if self.max_tokens is not None else 4096
        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": processed_messages,
            "max_tokens": max_tokens,
        }
        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        if self.top_p is not None:
            request_params["top_p"] = self.top_p

        if self.thinking_tokens is not None:
            request_params["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_tokens}

        output_format, use_beta_api = self._prepare_structured_output(response_format, tools)

        if tools:
            request_params["tools"] = self._convert_tools_to_anthropic(tools)

            if hasattr(tool_executor, "parallel_calls"):
                if "tool_choice" not in request_params:
                    request_params["tool_choice"] = {"type": "auto"}

                request_params["tool_choice"]["disable_parallel_tool_use"] = not tool_executor.parallel_calls

        try:
            if use_beta_api and not tools:
                api_response = await self.async_client.beta.messages.parse(
                    betas=["structured-outputs-2025-11-13"], output_format=output_format, **request_params
                )
                parsed_output = api_response.parsed_output
                response_text = json.dumps(parsed_output.model_dump()) if hasattr(parsed_output, "model_dump") else json.dumps(parsed_output)

                message = api_response.message if hasattr(api_response, "message") else None
                if message:
                    usage = {
                        "input_tokens": message.usage.input_tokens if hasattr(message, "usage") else 0,
                        "output_tokens": message.usage.output_tokens if hasattr(message, "usage") else 0,
                        "total_tokens": (message.usage.input_tokens + message.usage.output_tokens) if hasattr(message, "usage") else 0,
                    }
                    self._accumulate_metrics(usage)

                    finish_reason = message.stop_reason if hasattr(message, "stop_reason") else "stop"
                    params = {}
                    if self.temperature is not None:
                        params["temperature"] = self.temperature
                    if self.top_p is not None:
                        params["top_p"] = self.top_p
                    if self.max_tokens is not None:
                        params["max_tokens"] = self.max_tokens
                    if self.thinking_tokens is not None:
                        params["thinking_tokens"] = self.thinking_tokens

                    cumulative_metrics = self._get_cumulative_metrics()
                    usage_info = {
                        "model": self.model,
                        "params": params,
                        "finish_reason": finish_reason,
                        "input_tokens": cumulative_metrics.input_tokens,
                        "output_tokens": cumulative_metrics.output_tokens,
                    }

                    return self._create_llm_response(response_text, usage_info, [], {})
                else:
                    return self._create_llm_response(response_text, {"model": self.model}, [], {})

            elif use_beta_api and tools:
                request_params["betas"] = ["structured-outputs-2025-11-13"]
                request_params["output_format"] = output_format
                api_response_beta: AnthropicMessage = await self.async_client.beta.messages.create(**request_params)
                api_response = api_response_beta
            else:
                api_response_regular: AnthropicMessage = await self.async_client.messages.create(**request_params)
                api_response = api_response_regular

            content_blocks = api_response.content

            tool_use_blocks = [block for block in content_blocks if block.type == "tool_use"]

            if tool_use_blocks:
                usage = {
                    "input_tokens": api_response.usage.input_tokens,
                    "output_tokens": api_response.usage.output_tokens,
                    "total_tokens": api_response.usage.input_tokens + api_response.usage.output_tokens,
                }
                self._accumulate_metrics(usage)

                filtered_content_blocks = []
                tool_calls = []
                tool_outputs = {}

                for block in content_blocks:
                    if block.type == "text" and not block.text.strip():
                        continue

                    if block.type == "tool_use":
                        clean_block = {
                            "type": block.type,
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                        filtered_content_blocks.append(clean_block)

                        tool_calls.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                },
                            }
                        )
                    else:
                        filtered_content_blocks.append(block.model_dump())

                messages.append({"role": "assistant", "content": filtered_content_blocks})

                if hasattr(tool_executor, "parallel_calls") and tool_executor.parallel_calls:
                    tool_outputs = await self._execute_tools_parallel_async(tool_executor, tool_calls)

                    messages[-1] = {"role": "assistant", "content": []}

                    tool_results = []
                    for tc_dict in tool_calls:
                        tc_item: dict[str, Any] = tc_dict if isinstance(tc_dict, dict) else {}
                        messages[-1]["content"].append(
                            {
                                "type": "tool_use",
                                "id": tc_item.get("id", ""),
                                "name": tc_item.get("function", {}).get("name", ""),
                                "input": json.loads(tc_item.get("function", {}).get("arguments", "{}"))
                                if isinstance(tc_item.get("function", {}).get("arguments"), str)
                                else tc_item.get("function", {}).get("arguments", {}),
                            }
                        )

                        function_name = tc_item.get("function", {}).get("name", "")
                        tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tc_item["id"],
                                "content": str(tool_result),
                            }
                        )

                    messages.append({"role": "user", "content": tool_results})
                else:
                    tool_outputs = {}
                    tool_results = []

                    for tool_block in tool_use_blocks:
                        function_name = tool_block.name
                        function_args = tool_block.input

                        tool_result = await self._execute_tool_async(tool_executor, function_name, function_args)
                        tool_outputs[function_name] = tool_result

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_block.id,
                                "content": tool_result,
                            }
                        )

                    messages.append({"role": "user", "content": tool_results})

                if hasattr(tool_executor, "_tool_outputs"):
                    tool_executor._tool_outputs.update(tool_outputs)
                if hasattr(tool_executor, "_tool_calls"):
                    tool_executor._tool_calls.extend(tool_calls)

                return None
            else:
                usage = {
                    "input_tokens": api_response.usage.input_tokens,
                    "output_tokens": api_response.usage.output_tokens,
                    "total_tokens": api_response.usage.input_tokens + api_response.usage.output_tokens,
                }

                self._accumulate_metrics(usage)

                response_text = ""
                thinking_content = ""
                for block in content_blocks:
                    if block.type == "text":
                        response_text += block.text
                    elif block.type == "thinking":
                        thinking_content += block.thinking

                tool_calls = []
                if hasattr(tool_executor, "_tool_calls"):
                    tool_calls = tool_executor._tool_calls.copy()

                tool_outputs = {}
                if hasattr(tool_executor, "_tool_outputs"):
                    tool_outputs = tool_executor._tool_outputs.copy()

                cumulative_metrics = self._get_cumulative_metrics()

                finish_reason = "stop"
                if api_response.stop_reason:
                    finish_reason = api_response.stop_reason

                params = {}
                if self.temperature is not None:
                    params["temperature"] = self.temperature
                if self.top_p is not None:
                    params["top_p"] = self.top_p
                if self.max_tokens is not None:
                    params["max_tokens"] = self.max_tokens
                if self.thinking_tokens is not None:
                    params["thinking_tokens"] = self.thinking_tokens

                usage_info = {
                    "model": self.model,
                    "params": params,
                    "finish_reason": finish_reason,
                    "input_tokens": cumulative_metrics.input_tokens,
                    "output_tokens": cumulative_metrics.output_tokens,
                }

                if thinking_content:
                    usage_info["reasoning_content"] = thinking_content

                return self._create_llm_response(response_text, usage_info, tool_calls, tool_outputs)

        except RateLimitError as e:
            raise Exception(f"Rate limit error from Anthropic SDK: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from Anthropic SDK: {e}") from e
        except APIStatusError as e:
            try:
                error_message = e.response.json().get("error", {})
                error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            except Exception:
                error_message = str(e)
            raise Exception(f"Status error from Anthropic SDK: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from Anthropic SDK: {e}") from e

    async def ahandle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        content_blocks: list[dict[str, Any]] = []
        current_tool_use = None

        processed_messages = []
        for msg in messages:
            formatted_msg = self._format_files_for_anthropic(msg.copy())
            processed_messages.append(formatted_msg)

        max_tokens = self.max_tokens if self.max_tokens is not None else 4096
        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": processed_messages,
            "max_tokens": max_tokens,
        }
        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        if self.top_p is not None:
            request_params["top_p"] = self.top_p
        if self.thinking_tokens is not None:
            request_params["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_tokens}

        from pydantic import BaseModel

        use_beta_api = False
        if response_format is not None:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                use_beta_api = True

        if tools:
            request_params["tools"] = self._convert_tools_to_anthropic(tools)
            if hasattr(tool_executor, "parallel_calls"):
                if "tool_choice" not in request_params:
                    request_params["tool_choice"] = {"type": "auto"}
                request_params["tool_choice"]["disable_parallel_tool_use"] = not tool_executor.parallel_calls

        try:
            if use_beta_api:
                request_params["betas"] = ["structured-outputs-2025-11-13"]
                request_params["output_format"] = response_format
                stream_method = self.async_client.beta.messages.stream
            else:
                stream_method = self.async_client.messages.stream

            async with stream_method(**request_params) as stream:
                usage = None
                async for event in stream:
                    text_content = None
                    event_type_name = type(event).__name__

                    if hasattr(event, "delta"):
                        delta = event.delta
                        if hasattr(delta, "type") and delta.type == "text_delta":
                            text_content = getattr(delta, "text", None)
                        elif hasattr(delta, "text"):
                            text_content = delta.text

                    if text_content is None and use_beta_api:
                        if event_type_name == "ParsedBetaTextEvent" or isinstance(event, ParsedBetaTextEvent):
                            continue

                    if text_content:
                        yield ContentEvent(content=text_content)
                        continue

                    if use_beta_api and isinstance(event, ParsedBetaMessageStopEvent):
                        if hasattr(event, "message") and hasattr(event.message, "usage"):
                            usage = {
                                "input_tokens": event.message.usage.input_tokens,
                                "output_tokens": event.message.usage.output_tokens,
                                "total_tokens": event.message.usage.input_tokens + event.message.usage.output_tokens,
                            }

                    if use_beta_api and isinstance(event, ParsedBetaContentBlockStopEvent):
                        pass

                    if use_beta_api and isinstance(event, BetaRawContentBlockStartEvent):
                        beta_content_block = event.content_block
                        if beta_content_block.type == "tool_use":
                            current_tool_use = {
                                "type": "tool_use",
                                "id": beta_content_block.id,
                                "name": beta_content_block.name,
                                "input": {},
                            }
                            content_blocks.append(current_tool_use)
                        else:
                            block_dict = (
                                beta_content_block.model_dump() if hasattr(beta_content_block, "model_dump") else {"type": beta_content_block.type}
                            )
                            content_blocks.append(block_dict)
                            current_tool_use = None
                        continue

                    if use_beta_api and isinstance(event, BetaInputJsonEvent):
                        if current_tool_use:
                            if "partial_json" not in current_tool_use:
                                current_tool_use["partial_json"] = ""
                            partial_json = event.partial_json if hasattr(event, "partial_json") else ""
                            json_str: str = ""
                            if isinstance(partial_json, str):
                                json_str = partial_json
                            elif hasattr(partial_json, "__iter__") and not isinstance(partial_json, str):
                                json_str = "".join(str(item) for item in partial_json)
                            else:
                                json_str = str(partial_json)
                            existing_json: str = str(current_tool_use.get("partial_json", ""))
                            current_tool_use["partial_json"] = existing_json + json_str
                        continue

                    if isinstance(event, ContentBlockStartEvent):
                        content_block: Any = event.content_block

                        if content_block.type == "tool_use":
                            current_tool_use = {
                                "type": "tool_use",
                                "id": content_block.id,
                                "name": content_block.name,
                                "input": {},
                            }
                            content_blocks.append(current_tool_use)
                        else:
                            content_blocks.append(content_block.model_dump())
                            current_tool_use = None

                    elif isinstance(event, ContentBlockDeltaEvent):
                        delta = event.delta

                        if delta.type == "text_delta":
                            if not use_beta_api:
                                yield ContentEvent(content=delta.text)

                        elif delta.type == "thinking_delta":
                            yield ReasoningEvent(reasoning=delta.thinking)

                        elif delta.type == "input_json_delta" and current_tool_use:
                            if "partial_json" not in current_tool_use:
                                current_tool_use["partial_json"] = ""
                            partial_json_str: str = (
                                delta.partial_json
                                if isinstance(delta.partial_json, str)
                                else "".join(delta.partial_json)
                                if hasattr(delta.partial_json, "__iter__") and not isinstance(delta.partial_json, str)
                                else str(delta.partial_json)
                            )
                            existing_partial = current_tool_use.get("partial_json", "")
                            existing_str: str = (
                                existing_partial
                                if isinstance(existing_partial, str)
                                else "".join(existing_partial)
                                if hasattr(existing_partial, "__iter__") and not isinstance(existing_partial, str)
                                else str(existing_partial)
                            )
                            current_tool_use["partial_json"] = existing_str + partial_json_str

                    elif isinstance(event, (ContentBlockStopEvent, ParsedBetaContentBlockStopEvent)):
                        if current_tool_use and "partial_json" in current_tool_use:
                            try:
                                partial_json_value = current_tool_use["partial_json"]
                                partial_json_converted: str = (
                                    partial_json_value
                                    if isinstance(partial_json_value, str)
                                    else "".join(partial_json_value)
                                    if hasattr(partial_json_value, "__iter__") and not isinstance(partial_json_value, str)
                                    else str(partial_json_value)
                                )
                                current_tool_use["input"] = json.loads(partial_json_converted)
                                del current_tool_use["partial_json"]
                            except json.JSONDecodeError:
                                current_tool_use["input"] = {}
                        if isinstance(event, ParsedBetaContentBlockStopEvent) and hasattr(event, "content_block"):
                            cb = event.content_block
                            if cb.type == "tool_use" and hasattr(cb, "input") and cb.input:
                                if current_tool_use:
                                    current_tool_use["input"] = cb.input
                        current_tool_use = None

                    elif isinstance(event, (MessageStopEvent, ParsedBetaMessageStopEvent)):
                        final_message = None
                        try:
                            if isinstance(event, ParsedBetaMessageStopEvent) and hasattr(event, "message"):
                                final_message = event.message
                                if hasattr(final_message, "usage"):
                                    usage = {
                                        "input_tokens": final_message.usage.input_tokens,
                                        "output_tokens": final_message.usage.output_tokens,
                                        "total_tokens": final_message.usage.input_tokens + final_message.usage.output_tokens,
                                    }
                            else:
                                final_message = await stream.get_final_message()
                                if hasattr(final_message, "usage") and final_message.usage:
                                    usage = {
                                        "input_tokens": final_message.usage.input_tokens,
                                        "output_tokens": final_message.usage.output_tokens,
                                        "total_tokens": final_message.usage.input_tokens + final_message.usage.output_tokens,
                                    }
                        except (AttributeError, Exception):
                            pass

                        tool_use_blocks = [block for block in content_blocks if block.get("type") == "tool_use"]

                        if tool_use_blocks:
                            filtered_content_blocks = []
                            for block in content_blocks:
                                if block.get("type") == "text" and not block.get("text", "").strip():
                                    continue

                                if block.get("type") == "tool_use":
                                    clean_block = {
                                        "type": block["type"],
                                        "id": block["id"],
                                        "name": block["name"],
                                        "input": block["input"],
                                    }
                                    filtered_content_blocks.append(clean_block)
                                else:
                                    filtered_content_blocks.append(block)

                            messages.append({"role": "assistant", "content": filtered_content_blocks})

                            tool_calls = []
                            for tool_block in tool_use_blocks:
                                tool_calls.append(
                                    {
                                        "id": tool_block["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tool_block["name"],
                                            "arguments": json.dumps(tool_block["input"]),
                                        },
                                    }
                                )

                            if hasattr(tool_executor, "parallel_calls") and tool_executor.parallel_calls:
                                tool_outputs = await self._execute_tools_parallel_async(tool_executor, tool_calls)

                                messages[-1] = {"role": "assistant", "content": []}

                                tool_results = []
                                for tc in tool_calls:
                                    messages[-1]["content"].append(
                                        {
                                            "type": "tool_use",
                                            "id": tc["id"],
                                            "name": tc["function"]["name"],
                                            "input": json.loads(tc["function"]["arguments"])
                                            if isinstance(tc["function"]["arguments"], str)
                                            else tc["function"]["arguments"],
                                        }
                                    )

                                    function_name = tc["function"]["name"]
                                    tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")
                                    tool_results.append(
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tc["id"],
                                            "content": str(tool_result),
                                        }
                                    )

                                messages.append({"role": "user", "content": tool_results})

                                if usage:
                                    self._accumulate_metrics(usage)

                                yield ToolCallsEvent(tool_calls=tool_calls)

                                yield ToolOutputsEvent(tool_outputs=tool_outputs)

                                if usage:
                                    yield ResponseCompletedEvent(usage=usage)
                                yield True
                                return
                            else:
                                tool_outputs = {}
                                tool_results = []

                                for tool_block in tool_use_blocks:
                                    function_name = tool_block["name"]
                                    function_args = tool_block["input"]

                                    tool_result = await self._execute_tool_async(tool_executor, function_name, function_args)
                                    tool_outputs[function_name] = tool_result

                                    tool_results.append(
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_block["id"],
                                            "content": tool_result,
                                        }
                                    )

                                messages.append({"role": "user", "content": tool_results})

                                if usage:
                                    self._accumulate_metrics(usage)

                                yield ToolCallsEvent(tool_calls=tool_calls)

                                yield ToolOutputsEvent(tool_outputs=tool_outputs)

                                if usage:
                                    yield ResponseCompletedEvent(usage=usage)
                                yield True
                                return

                        if usage:
                            self._accumulate_metrics(usage)

                        if usage:
                            yield ResponseCompletedEvent(usage=usage)
                        yield False

        except RateLimitError as e:
            raise Exception(f"Rate limit error from Anthropic SDK: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from Anthropic SDK: {e}") from e
        except APIStatusError as e:
            try:
                error_message = e.response.json().get("error", {})
                error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            except Exception:
                error_message = str(e)
            raise Exception(f"Status error from Anthropic SDK: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from Anthropic SDK: {e}") from e

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
            formatted_msg = self._format_files_for_anthropic(msg.copy())
            processed_messages.append(formatted_msg)

        max_tokens = self.max_tokens if self.max_tokens is not None else 4096
        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": processed_messages,
            "max_tokens": max_tokens,
        }
        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        if self.top_p is not None:
            request_params["top_p"] = self.top_p

        if self.thinking_tokens is not None:
            request_params["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_tokens}

        output_format, use_beta_api = self._prepare_structured_output(response_format, tools)

        if tools:
            request_params["tools"] = self._convert_tools_to_anthropic(tools)

            if hasattr(tool_executor, "parallel_calls"):
                if "tool_choice" not in request_params:
                    request_params["tool_choice"] = {"type": "auto"}

                request_params["tool_choice"]["disable_parallel_tool_use"] = not tool_executor.parallel_calls

        try:
            if use_beta_api and not tools:
                api_response = self.client.beta.messages.parse(betas=["structured-outputs-2025-11-13"], output_format=output_format, **request_params)
                parsed_output = api_response.parsed_output
                response_text = json.dumps(parsed_output.model_dump()) if hasattr(parsed_output, "model_dump") else json.dumps(parsed_output)

                message = api_response.message if hasattr(api_response, "message") else None
                if message:
                    usage = {
                        "input_tokens": message.usage.input_tokens if hasattr(message, "usage") else 0,
                        "output_tokens": message.usage.output_tokens if hasattr(message, "usage") else 0,
                        "total_tokens": (message.usage.input_tokens + message.usage.output_tokens) if hasattr(message, "usage") else 0,
                    }
                    self._accumulate_metrics(usage)

                    finish_reason = message.stop_reason if hasattr(message, "stop_reason") else "stop"
                    params = {}
                    if self.temperature is not None:
                        params["temperature"] = self.temperature
                    if self.top_p is not None:
                        params["top_p"] = self.top_p
                    if self.max_tokens is not None:
                        params["max_tokens"] = self.max_tokens
                    if self.thinking_tokens is not None:
                        params["thinking_tokens"] = self.thinking_tokens

                    cumulative_metrics = self._get_cumulative_metrics()
                    usage_info = {
                        "model": self.model,
                        "params": params,
                        "finish_reason": finish_reason,
                        "input_tokens": cumulative_metrics.input_tokens,
                        "output_tokens": cumulative_metrics.output_tokens,
                    }

                    return self._create_llm_response(response_text, usage_info, [], {})
                else:
                    return self._create_llm_response(response_text, {"model": self.model}, [], {})

            elif use_beta_api and tools:
                request_params["betas"] = ["structured-outputs-2025-11-13"]
                request_params["output_format"] = output_format
                api_response_beta: AnthropicMessage = self.client.beta.messages.create(**request_params)
                api_response = api_response_beta
            else:
                api_response_regular: AnthropicMessage = self.client.messages.create(**request_params)
                api_response = api_response_regular

            content_blocks = api_response.content

            tool_use_blocks = [block for block in content_blocks if block.type == "tool_use"]

            if tool_use_blocks:
                usage = {
                    "input_tokens": api_response.usage.input_tokens,
                    "output_tokens": api_response.usage.output_tokens,
                    "total_tokens": api_response.usage.input_tokens + api_response.usage.output_tokens,
                }
                self._accumulate_metrics(usage)

                filtered_content_blocks = []
                tool_calls = []
                tool_outputs: dict[str, Any] = {}

                for block in content_blocks:
                    if block.type == "text" and not block.text.strip():
                        continue

                    if block.type == "tool_use":
                        clean_block = {
                            "type": block.type,
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                        filtered_content_blocks.append(clean_block)

                        tool_calls.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input),
                                },
                            }
                        )
                    else:
                        filtered_content_blocks.append(block.model_dump())

                messages.append({"role": "assistant", "content": filtered_content_blocks})

                if hasattr(tool_executor, "parallel_calls") and tool_executor.parallel_calls:
                    tool_outputs = {}
                    tool_results = []
                    for tc_dict_exec in tool_calls:
                        tc_exec: dict[str, Any] = tc_dict_exec if isinstance(tc_dict_exec, dict) else {}
                        function_name = tc_exec.get("function", {}).get("name", "")
                        try:
                            function_args = json.loads(tc_exec.get("function", {}).get("arguments", "{}"))
                            if function_args is None:
                                function_args = {}
                        except (json.JSONDecodeError, TypeError):
                            function_args = {}
                        tool_result = self._execute_tool_sync(tool_executor, function_name, function_args)
                        tool_outputs[function_name] = tool_result
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tc_exec.get("id", ""),
                                "content": tool_result,
                            }
                        )

                    messages[-1] = {"role": "assistant", "content": []}

                    for tc_dict_msg in tool_calls:
                        tc_msg: dict[str, Any] = tc_dict_msg if isinstance(tc_dict_msg, dict) else {}
                        messages[-1]["content"].append(
                            {
                                "type": "tool_use",
                                "id": tc_msg.get("id", ""),
                                "name": tc_msg.get("function", {}).get("name", ""),
                                "input": json.loads(tc_msg.get("function", {}).get("arguments", "{}"))
                                if isinstance(tc_msg.get("function", {}).get("arguments"), str)
                                else tc_msg.get("function", {}).get("arguments", {}),
                            }
                        )

                    messages.append({"role": "user", "content": tool_results})

                    if hasattr(tool_executor, "_tool_outputs"):
                        tool_executor._tool_outputs.update(tool_outputs)
                    if hasattr(tool_executor, "_tool_calls"):
                        tool_executor._tool_calls.extend(tool_calls)

                    return None
                else:
                    tool_outputs = {}
                    tool_results = []

                    for tool_block in tool_use_blocks:
                        function_name = tool_block.name
                        function_args = tool_block.input

                        tool_result = self._execute_tool_sync(tool_executor, function_name, function_args)
                        tool_outputs[function_name] = tool_result

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_block.id,
                                "content": tool_result,
                            }
                        )

                    messages.append({"role": "user", "content": tool_results})

                    if hasattr(tool_executor, "_tool_outputs"):
                        tool_executor._tool_outputs.update(tool_outputs)
                    if hasattr(tool_executor, "_tool_calls"):
                        tool_executor._tool_calls.extend(tool_calls)

                return None
            else:
                usage = {
                    "input_tokens": api_response.usage.input_tokens,
                    "output_tokens": api_response.usage.output_tokens,
                    "total_tokens": api_response.usage.input_tokens + api_response.usage.output_tokens,
                }

                self._accumulate_metrics(usage)

                response_text = ""
                thinking_content = ""
                for block in content_blocks:
                    if block.type == "text":
                        response_text += block.text
                    elif block.type == "thinking":
                        thinking_content += block.thinking

                tool_calls = []
                if hasattr(tool_executor, "_tool_calls"):
                    tool_calls = tool_executor._tool_calls.copy()

                tool_outputs = {}
                if hasattr(tool_executor, "_tool_outputs"):
                    tool_outputs = tool_executor._tool_outputs.copy()

                cumulative_metrics = self._get_cumulative_metrics()

                finish_reason = "stop"
                if api_response.stop_reason:
                    finish_reason = api_response.stop_reason

                params = {}
                if self.temperature is not None:
                    params["temperature"] = self.temperature
                if self.top_p is not None:
                    params["top_p"] = self.top_p
                if self.max_tokens is not None:
                    params["max_tokens"] = self.max_tokens
                if self.thinking_tokens is not None:
                    params["thinking_tokens"] = self.thinking_tokens

                usage_info = {
                    "model": self.model,
                    "params": params,
                    "finish_reason": finish_reason,
                    "input_tokens": cumulative_metrics.input_tokens,
                    "output_tokens": cumulative_metrics.output_tokens,
                }

                if thinking_content:
                    usage_info["reasoning_content"] = thinking_content

                return self._create_llm_response(response_text, usage_info, tool_calls, tool_outputs)

        except RateLimitError as e:
            raise Exception(f"Rate limit error from Anthropic SDK: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from Anthropic SDK: {e}") from e
        except APIStatusError as e:
            try:
                error_message = e.response.json().get("error", {})
                error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            except Exception:
                error_message = str(e)
            raise Exception(f"Status error from Anthropic SDK: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from Anthropic SDK: {e}") from e

    def handle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        content_blocks: list[dict[str, Any]] = []
        current_tool_use = None

        processed_messages = []
        for msg in messages:
            formatted_msg = self._format_files_for_anthropic(msg.copy())
            processed_messages.append(formatted_msg)

        max_tokens = self.max_tokens if self.max_tokens is not None else 4096
        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": processed_messages,
            "max_tokens": max_tokens,
        }
        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        if self.top_p is not None:
            request_params["top_p"] = self.top_p

        if self.thinking_tokens is not None:
            request_params["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_tokens}

        from pydantic import BaseModel

        use_beta_api = False
        if response_format is not None:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                use_beta_api = True

        if tools:
            request_params["tools"] = self._convert_tools_to_anthropic(tools)

            if hasattr(tool_executor, "parallel_calls"):
                if "tool_choice" not in request_params:
                    request_params["tool_choice"] = {"type": "auto"}

                request_params["tool_choice"]["disable_parallel_tool_use"] = not tool_executor.parallel_calls

        try:
            if use_beta_api:
                request_params["betas"] = ["structured-outputs-2025-11-13"]
                request_params["output_format"] = response_format
                stream_method = self.client.beta.messages.stream
            else:
                stream_method = self.client.messages.stream

            with stream_method(**request_params) as stream:
                usage = None
                for event in stream:
                    text_content = None
                    event_type_name = type(event).__name__

                    if hasattr(event, "delta"):
                        delta = event.delta
                        if hasattr(delta, "type") and delta.type == "text_delta":
                            text_content = getattr(delta, "text", None)
                        elif hasattr(delta, "text"):
                            text_content = delta.text

                    if text_content is None and use_beta_api:
                        if event_type_name == "ParsedBetaTextEvent" or isinstance(event, ParsedBetaTextEvent):
                            continue

                    if text_content:
                        yield ContentEvent(content=text_content)
                        continue

                    if use_beta_api and isinstance(event, ParsedBetaMessageStopEvent):
                        if hasattr(event, "message") and hasattr(event.message, "usage"):
                            usage = {
                                "input_tokens": event.message.usage.input_tokens,
                                "output_tokens": event.message.usage.output_tokens,
                                "total_tokens": event.message.usage.input_tokens + event.message.usage.output_tokens,
                            }

                    if use_beta_api and isinstance(event, ParsedBetaContentBlockStopEvent):
                        pass

                    if use_beta_api and isinstance(event, BetaRawContentBlockStartEvent):
                        beta_content_block = event.content_block
                        if beta_content_block.type == "tool_use":
                            current_tool_use = {
                                "type": "tool_use",
                                "id": beta_content_block.id,
                                "name": beta_content_block.name,
                                "input": {},
                            }
                            content_blocks.append(current_tool_use)
                        else:
                            block_dict = (
                                beta_content_block.model_dump() if hasattr(beta_content_block, "model_dump") else {"type": beta_content_block.type}
                            )
                            content_blocks.append(block_dict)
                            current_tool_use = None
                        continue

                    if use_beta_api and isinstance(event, BetaInputJsonEvent):
                        if current_tool_use:
                            if "partial_json" not in current_tool_use:
                                current_tool_use["partial_json"] = ""
                            partial_json = event.partial_json if hasattr(event, "partial_json") else ""
                            json_str: str = ""
                            if isinstance(partial_json, str):
                                json_str = partial_json
                            elif hasattr(partial_json, "__iter__") and not isinstance(partial_json, str):
                                json_str = "".join(str(item) for item in partial_json)
                            else:
                                json_str = str(partial_json)
                            existing_json: str = str(current_tool_use.get("partial_json", ""))
                            current_tool_use["partial_json"] = existing_json + json_str
                        continue

                    if isinstance(event, ContentBlockStartEvent):
                        content_block: Any = event.content_block

                        if content_block.type == "tool_use":
                            current_tool_use = {
                                "type": "tool_use",
                                "id": content_block.id,
                                "name": content_block.name,
                                "input": {},
                            }
                            content_blocks.append(current_tool_use)
                        else:
                            content_blocks.append(content_block.model_dump())
                            current_tool_use = None

                    elif isinstance(event, ContentBlockDeltaEvent):
                        delta = event.delta

                        if delta.type == "text_delta":
                            if not use_beta_api:
                                yield ContentEvent(content=delta.text)

                        elif delta.type == "thinking_delta":
                            yield ReasoningEvent(reasoning=delta.thinking)

                        elif delta.type == "input_json_delta" and current_tool_use:
                            if "partial_json" not in current_tool_use:
                                current_tool_use["partial_json"] = ""
                            partial_json_str: str = (
                                delta.partial_json
                                if isinstance(delta.partial_json, str)
                                else "".join(delta.partial_json)
                                if hasattr(delta.partial_json, "__iter__") and not isinstance(delta.partial_json, str)
                                else str(delta.partial_json)
                            )
                            existing_partial = current_tool_use.get("partial_json", "")
                            existing_str: str = (
                                existing_partial
                                if isinstance(existing_partial, str)
                                else "".join(existing_partial)
                                if hasattr(existing_partial, "__iter__") and not isinstance(existing_partial, str)
                                else str(existing_partial)
                            )
                            current_tool_use["partial_json"] = existing_str + partial_json_str

                    elif isinstance(event, (ContentBlockStopEvent, ParsedBetaContentBlockStopEvent)):
                        if current_tool_use and "partial_json" in current_tool_use:
                            try:
                                partial_json_value = current_tool_use["partial_json"]
                                partial_json_converted: str = (
                                    partial_json_value
                                    if isinstance(partial_json_value, str)
                                    else "".join(partial_json_value)
                                    if hasattr(partial_json_value, "__iter__") and not isinstance(partial_json_value, str)
                                    else str(partial_json_value)
                                )
                                current_tool_use["input"] = json.loads(partial_json_converted)
                                del current_tool_use["partial_json"]
                            except json.JSONDecodeError:
                                current_tool_use["input"] = {}
                        if isinstance(event, ParsedBetaContentBlockStopEvent) and hasattr(event, "content_block"):
                            cb = event.content_block
                            if cb.type == "tool_use" and hasattr(cb, "input") and cb.input:
                                if current_tool_use:
                                    current_tool_use["input"] = cb.input
                        current_tool_use = None

                    elif isinstance(event, (MessageStopEvent, ParsedBetaMessageStopEvent)):
                        try:
                            if isinstance(event, ParsedBetaMessageStopEvent) and hasattr(event, "message"):
                                final_message = event.message
                                if hasattr(final_message, "usage"):
                                    usage = {
                                        "input_tokens": final_message.usage.input_tokens,
                                        "output_tokens": final_message.usage.output_tokens,
                                        "total_tokens": final_message.usage.input_tokens + final_message.usage.output_tokens,
                                    }
                            else:
                                final_message = stream.get_final_message()
                                if hasattr(final_message, "usage") and final_message.usage:
                                    usage = {
                                        "input_tokens": final_message.usage.input_tokens,
                                        "output_tokens": final_message.usage.output_tokens,
                                        "total_tokens": final_message.usage.input_tokens + final_message.usage.output_tokens,
                                    }
                        except (AttributeError, Exception):
                            pass

                        tool_use_blocks = [block for block in content_blocks if block.get("type") == "tool_use"]

                        if tool_use_blocks:
                            filtered_content_blocks = []
                            for block in content_blocks:
                                if block.get("type") == "text" and not block.get("text", "").strip():
                                    continue

                                if block.get("type") == "tool_use":
                                    clean_block = {
                                        "type": block["type"],
                                        "id": block["id"],
                                        "name": block["name"],
                                        "input": block["input"],
                                    }
                                    filtered_content_blocks.append(clean_block)
                                else:
                                    filtered_content_blocks.append(block)

                            messages.append({"role": "assistant", "content": filtered_content_blocks})

                            tool_calls = []
                            for tool_block in tool_use_blocks:
                                tool_calls.append(
                                    {
                                        "id": tool_block["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tool_block["name"],
                                            "arguments": json.dumps(tool_block["input"]),
                                        },
                                    }
                                )

                            if hasattr(tool_executor, "parallel_calls") and tool_executor.parallel_calls:
                                tool_outputs = {}
                                tool_results = []
                                for tc in tool_calls:
                                    function_name = tc["function"]["name"]
                                    try:
                                        function_args = json.loads(tc["function"]["arguments"])
                                        if function_args is None:
                                            function_args = {}
                                    except (json.JSONDecodeError, TypeError):
                                        function_args = {}
                                    tool_result = self._execute_tool_sync(tool_executor, function_name, function_args)
                                    tool_outputs[function_name] = tool_result
                                    tool_results.append(
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tc["id"],
                                            "content": tool_result,
                                        }
                                    )

                                messages[-1] = {"role": "assistant", "content": []}

                                for tc in tool_calls:
                                    messages[-1]["content"].append(
                                        {
                                            "type": "tool_use",
                                            "id": tc["id"],
                                            "name": tc["function"]["name"],
                                            "input": json.loads(tc["function"]["arguments"])
                                            if isinstance(tc["function"]["arguments"], str)
                                            else tc["function"]["arguments"],
                                        }
                                    )

                                messages.append({"role": "user", "content": tool_results})

                                if usage:
                                    self._accumulate_metrics(usage)

                                yield ToolCallsEvent(tool_calls=tool_calls)

                                yield ToolOutputsEvent(tool_outputs=tool_outputs)

                                if usage:
                                    yield ResponseCompletedEvent(usage=usage)
                                yield True
                                return
                            else:
                                tool_outputs = {}
                                tool_results = []

                                for tool_block in tool_use_blocks:
                                    function_name = tool_block["name"]
                                    function_args = tool_block["input"]

                                    tool_result = self._execute_tool_sync(tool_executor, function_name, function_args)
                                    tool_outputs[function_name] = tool_result

                                    tool_results.append(
                                        {
                                            "type": "tool_result",
                                            "tool_use_id": tool_block["id"],
                                            "content": tool_result,
                                        }
                                    )

                                messages.append({"role": "user", "content": tool_results})

                                if usage:
                                    self._accumulate_metrics(usage)

                                yield ToolCallsEvent(tool_calls=tool_calls)

                                yield ToolOutputsEvent(tool_outputs=tool_outputs)

                                if usage:
                                    yield ResponseCompletedEvent(usage=usage)
                                yield True
                                return

                        if usage:
                            self._accumulate_metrics(usage)

                        if usage:
                            yield ResponseCompletedEvent(usage=usage)
                        yield False

        except RateLimitError as e:
            raise Exception(f"Rate limit error from Anthropic SDK: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from Anthropic SDK: {e}") from e
        except APIStatusError as e:
            try:
                error_message = e.response.json().get("error", {})
                error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            except Exception:
                error_message = str(e)
            raise Exception(f"Status error from Anthropic SDK: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from Anthropic SDK: {e}") from e

    def _format_files_for_anthropic(self, message: dict[str, Any]) -> dict[str, Any]:
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
                content.append({"type": "text", "text": message_content})
            elif isinstance(message_content, list):
                content = message_content

        for file_obj in file_objects:
            if file_obj.file_type == FileType.IMAGE:
                if file_obj.url:
                    content.append({"type": "image", "source": {"type": "url", "url": file_obj.url}})
                else:
                    base64_data = file_obj.to_base64()
                    if base64_data:
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": file_obj.mime_type or "image/jpeg",
                                    "data": base64_data,
                                },
                            }
                        )
            elif file_obj.file_type == FileType.DOCUMENT:
                if file_obj.url:
                    content.append({"type": "document", "source": {"type": "url", "url": file_obj.url}})
                else:
                    base64_data = file_obj.to_base64()
                    if base64_data:
                        content.append(
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": file_obj.mime_type or "application/pdf",
                                    "data": base64_data,
                                },
                            }
                        )
            else:
                from hypertic.utils.log import get_logger

                logger = get_logger(__name__)
                logger.warning(
                    f"File type {file_obj.file_type} not supported in Anthropic API "
                    f"(only images and documents are supported): {file_obj.url or file_obj.filepath}"
                )

        message["content"] = content
        message.pop("_file_objects", None)
        message.pop("files", None)
        return message
