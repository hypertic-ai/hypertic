import json
from dataclasses import dataclass, field
from os import getenv
from typing import Any, TypeAlias

from hypertic.models.base import Base
from hypertic.models.events import (
    ContentEvent,
    FinishReasonEvent,
    ResponseCompletedEvent,
    ToolCallsEvent,
    ToolOutputsEvent,
)
from hypertic.utils.log import get_logger

logger = get_logger(__name__)

try:
    from mistralai import Mistral as MistralClient
    from mistralai.models import (
        AssistantMessage,
        ChatCompletionResponse,
        DocumentURLChunk,
        HTTPValidationError,
        ImageURLChunk,
        SDKError,
        SystemMessage,
        TextChunk,
        ToolMessage,
        UserMessage,
    )

    MistralMessage: TypeAlias = UserMessage | AssistantMessage | SystemMessage | ToolMessage

except ImportError as err:
    raise ImportError("`mistralai` not installed. Please install using `pip install mistralai`") from err


@dataclass
class MistralAI(Base):
    api_key: str | None = None
    model: str | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None

    client: Any = field(default=None, init=False)
    async_client: Any = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("MISTRAL_API_KEY")

        if self.api_key is not None:
            self.async_client = MistralClient(api_key=self.api_key)
            self.client = MistralClient(api_key=self.api_key)
        else:
            self.async_client = MistralClient()
            self.client = MistralClient()

    def _format_message(self, message: dict[str, Any]) -> MistralMessage:
        role = message["role"]
        content = message.get("content")

        if role == "system":
            if isinstance(content, list):
                text_content = ""
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content = block.get("text", "")
                        break
                return SystemMessage(content=text_content)
            return SystemMessage(content=content or "")
        elif role == "user":
            return UserMessage(content=content)
        elif role == "assistant":
            if message.get("tool_calls"):
                return AssistantMessage(
                    content=content,
                    tool_calls=message["tool_calls"],
                )
            else:
                return AssistantMessage(content=content or "")
        elif role == "tool":
            return ToolMessage(content=content or "", tool_call_id=message.get("tool_call_id", ""))
        else:
            return UserMessage(content=content)

    async def ahandle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ) -> Any | None:
        try:
            formatted_messages = [self._format_files_for_mistral(msg.copy()) for msg in messages]

            mistral_messages = [self._format_message(msg) for msg in formatted_messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": mistral_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            }

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
                if response_format is not None:
                    raise ValueError("Mistral does not support structured output (response_format) when tools are present.")

            if response_format is not None and not tools:
                from pydantic import BaseModel

                if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                    pydantic_model = response_format
                    if hasattr(pydantic_model, "model_json_schema"):
                        json_schema = pydantic_model.model_json_schema()
                        request_params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {"name": pydantic_model.__name__, "schema": json_schema},
                        }
                    else:
                        request_params["response_format"] = response_format
                else:
                    request_params["response_format"] = response_format

            api_response: ChatCompletionResponse = await self.async_client.chat.complete_async(**request_params)

            message: AssistantMessage = api_response.choices[0].message

            if message.tool_calls:
                tool_usage: dict[str, Any] = api_response.usage.__dict__ if api_response.usage else {}
                self._accumulate_metrics(tool_usage)

                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )

                messages.append({"role": message.role, "content": message.content, "tool_calls": tool_calls})

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if message.tool_calls:
                        first_tool_call = message.tool_calls[0]
                        function_name = first_tool_call.function.name
                        try:
                            arguments_raw = first_tool_call.function.arguments
                            if arguments_raw:
                                arguments_str = arguments_raw if isinstance(arguments_raw, str) else json.dumps(arguments_raw)
                                function_args = json.loads(arguments_str)
                            else:
                                function_args = {}
                            if function_args is None:
                                function_args = {}
                        except (json.JSONDecodeError, TypeError):
                            function_args = {}

                        messages[-1]["tool_calls"] = [tool_calls[0]]

                        tool_call_dict = tool_calls[0]
                        results = await self._execute_tools_parallel_async(tool_executor, [tool_call_dict])
                        tool_result = results.get(function_name, f"Tool '{function_name}' execution failed")

                        tool_outputs = {function_name: tool_result}

                        first_tool_call_for_storage = tool_call_dict

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": first_tool_call.id,
                                "content": tool_result,
                            }
                        )

                        if hasattr(tool_executor, "_tool_outputs"):
                            tool_executor._tool_outputs.update(tool_outputs)
                        if hasattr(tool_executor, "_tool_calls"):
                            tool_executor._tool_calls.append(first_tool_call_for_storage)

                        return None
                else:
                    tool_outputs = await self._execute_tools_parallel_async(tool_executor, tool_calls)

                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

                    if hasattr(tool_executor, "_tool_outputs"):
                        tool_executor._tool_outputs.update(tool_outputs)
                    if hasattr(tool_executor, "_tool_calls"):
                        tool_executor._tool_calls.extend(tool_calls)

                    return None
            else:
                accumulated_content_parts = []

                for msg in messages:
                    if msg.get("role") == "assistant" and msg.get("content"):
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(str(item) if not isinstance(item, dict) else item.get("text", str(item)) for item in content).strip()
                        else:
                            content = str(content).strip()
                        if content:
                            accumulated_content_parts.append(content)

                final_content = message.content or ""
                if isinstance(final_content, list):
                    final_content = " ".join(
                        str(item) if not isinstance(item, dict) else item.get("text", str(item)) for item in final_content
                    ).strip()
                else:
                    final_content = str(final_content).strip()
                if final_content:
                    accumulated_content_parts.append(final_content)

                content = "\n\n".join(accumulated_content_parts) if accumulated_content_parts else ""

                messages.append({"role": message.role, "content": final_content})

                tool_calls = []
                if hasattr(tool_executor, "_tool_calls"):
                    tool_calls = tool_executor._tool_calls.copy()

                tool_outputs = {}
                if hasattr(tool_executor, "_tool_outputs"):
                    tool_outputs = tool_executor._tool_outputs.copy()

                final_usage = api_response.usage.__dict__ if api_response.usage else {}

                self._accumulate_metrics(final_usage)

                cumulative_metrics = self._get_cumulative_metrics()

                return self._create_llm_response(
                    content,
                    {
                        "input_tokens": cumulative_metrics.input_tokens,
                        "output_tokens": cumulative_metrics.output_tokens,
                        "model": self.model,
                        "finish_reason": api_response.choices[0].finish_reason or "stop",
                        "params": {
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "max_tokens": self.max_tokens,
                        },
                    },
                    tool_calls,
                    tool_outputs,
                )

        except HTTPValidationError as e:
            raise Exception(f"Mistral API Validation Error: {e!s}") from e
        except SDKError as e:
            raise Exception(f"Mistral SDK Error: {e!s}") from e
        except Exception as e:
            raise Exception(f"Unexpected error from Mistral: {e!s}") from e

        return None

    async def ahandle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        try:
            tool_calls: list[dict[str, Any]] = []
            usage = None

            formatted_messages = [self._format_files_for_mistral(msg.copy()) for msg in messages]

            mistral_messages = [self._format_message(msg) for msg in formatted_messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": mistral_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            }

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
                if response_format is not None:
                    raise ValueError("Mistral does not support structured output (response_format) when tools are present.")

            if response_format is not None and not tools:
                from pydantic import BaseModel

                if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                    pydantic_model = response_format
                    if hasattr(pydantic_model, "model_json_schema"):
                        json_schema = pydantic_model.model_json_schema()
                        request_params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {"name": pydantic_model.__name__, "schema": json_schema},
                        }
                    else:
                        request_params["response_format"] = response_format
                else:
                    request_params["response_format"] = response_format

            stream = await self.async_client.chat.stream_async(**request_params)

            try:
                async for chunk in stream:
                    try:
                        if hasattr(chunk, "data") and chunk.data:
                            if hasattr(chunk.data, "usage") and chunk.data.usage:
                                usage_obj = chunk.data.usage
                                usage = {
                                    "input_tokens": usage_obj.prompt_tokens if hasattr(usage_obj, "prompt_tokens") else 0,
                                    "output_tokens": usage_obj.completion_tokens if hasattr(usage_obj, "completion_tokens") else 0,
                                    "total_tokens": usage_obj.total_tokens if hasattr(usage_obj, "total_tokens") else 0,
                                }
                                self._accumulate_metrics(usage)
                                tool_executor._streaming_usage = usage

                            if not chunk.data.choices:
                                if usage:
                                    yield ResponseCompletedEvent(usage=usage)
                                continue

                            if chunk.data.choices and len(chunk.data.choices) > 0:
                                choice = chunk.data.choices[0]
                                delta = choice.delta

                                if hasattr(chunk.data, "usage") and chunk.data.usage and not usage:
                                    usage_obj = chunk.data.usage
                                    usage = {
                                        "input_tokens": usage_obj.prompt_tokens if hasattr(usage_obj, "prompt_tokens") else 0,
                                        "output_tokens": usage_obj.completion_tokens if hasattr(usage_obj, "completion_tokens") else 0,
                                        "total_tokens": usage_obj.total_tokens if hasattr(usage_obj, "total_tokens") else 0,
                                    }
                                    self._accumulate_metrics(usage)
                                    tool_executor._streaming_usage = usage

                                if delta.content:
                                    try:
                                        if isinstance(delta.content, str):
                                            yield ContentEvent(content=delta.content)
                                        elif isinstance(delta.content, list):
                                            text_content = ""
                                            for item in delta.content:
                                                try:
                                                    if hasattr(item, "text"):
                                                        text_content += item.text
                                                    elif isinstance(item, dict):
                                                        if "text" in item:
                                                            text_content += item["text"]
                                                        elif "type" in item and item["type"] == "reference":
                                                            if "text" in item:
                                                                text_content += item["text"]
                                                            elif "content" in item:
                                                                text_content += str(item["content"])
                                                        else:
                                                            text_content += str(item)
                                                    elif isinstance(item, str):
                                                        text_content += item
                                                    else:
                                                        text_content += str(item)
                                                except Exception:
                                                    continue
                                            if text_content:
                                                yield ContentEvent(content=text_content)
                                    except Exception:
                                        continue

                            if delta.tool_calls:
                                try:
                                    for tool_call_delta in delta.tool_calls:
                                        index = tool_call_delta.index or 0

                                        while len(tool_calls) <= index:
                                            tool_calls.append(
                                                {
                                                    "id": "",
                                                    "type": "function",
                                                    "function": {"name": "", "arguments": ""},
                                                }
                                            )

                                        if tool_call_delta.id:
                                            if tool_call_delta.id and not tool_calls[index]["id"]:
                                                tool_calls[index]["id"] = tool_call_delta.id
                                            else:
                                                tool_calls[index]["id"] += tool_call_delta.id

                                        if tool_call_delta.function:
                                            func_delta = tool_call_delta.function
                                            if func_delta.name:
                                                if func_delta.name and not tool_calls[index]["function"]["name"]:
                                                    tool_calls[index]["function"]["name"] = func_delta.name
                                                else:
                                                    tool_calls[index]["function"]["name"] += func_delta.name
                                            if func_delta.arguments:
                                                if func_delta.arguments and not tool_calls[index]["function"]["arguments"]:
                                                    tool_calls[index]["function"]["arguments"] = func_delta.arguments
                                                else:
                                                    tool_calls[index]["function"]["arguments"] += func_delta.arguments
                                except Exception:
                                    continue

                            if choice.finish_reason:
                                yield FinishReasonEvent(finish_reason=choice.finish_reason)
                                if usage:
                                    yield ResponseCompletedEvent(usage=usage)
                                break
                    except Exception:
                        continue
            except Exception as stream_error:
                if "validation error" in str(stream_error).lower() or "pydantic" in str(stream_error).lower():
                    logger.warning(
                        f"Mistral Pydantic validation error (continuing): {stream_error!s}",
                        exc_info=True,
                    )
                else:
                    logger.warning(f"Mistral streaming error (continuing): {stream_error!s}", exc_info=True)

            if tool_calls and any(tc["function"]["name"] for tc in tool_calls):
                tool_calls_to_yield = tool_calls
                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls and len(tool_calls) > 1:
                    tool_calls_to_yield = [tool_calls[0]]

                messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls_to_yield})

                yield ToolCallsEvent(tool_calls=tool_calls_to_yield)

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if tool_calls and tool_calls[0]["function"]["name"]:
                        first_tool = tool_calls[0]
                        function_name = first_tool["function"]["name"]
                        try:
                            function_args = json.loads(first_tool["function"]["arguments"]) if first_tool["function"]["arguments"] else {}
                            if function_args is None:
                                function_args = {}
                        except (json.JSONDecodeError, TypeError):
                            function_args = {}

                        tool_outputs_iter = await self._execute_tools_parallel_async(tool_executor, [first_tool])
                        tool_result = tool_outputs_iter.get(function_name, f"Tool '{function_name}' execution failed")

                        tool_outputs = {function_name: tool_result}

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": first_tool["id"],
                                "content": tool_result,
                            }
                        )

                        if hasattr(tool_executor, "_tool_outputs"):
                            tool_executor._tool_outputs.update(tool_outputs)
                        if hasattr(tool_executor, "_tool_calls"):
                            tool_executor._tool_calls.append(first_tool)

                        yield ToolOutputsEvent(tool_outputs=tool_outputs)

                        if usage:
                            yield ResponseCompletedEvent(usage=usage)

                        yield True
                        return
                else:
                    tool_outputs = await self._execute_tools_parallel_async(tool_executor, tool_calls)

                    for tool_call in tool_calls:
                        if tool_call["function"]["name"]:
                            function_name = tool_call["function"]["name"]
                            tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": tool_result,
                                }
                            )

                    yield ToolOutputsEvent(tool_outputs=tool_outputs)

                    if usage:
                        yield ResponseCompletedEvent(usage=usage)

                if hasattr(tool_executor, "_tool_outputs"):
                    tool_executor._tool_outputs.update(tool_outputs)
                if hasattr(tool_executor, "_tool_calls"):
                    tool_executor._tool_calls.extend(tool_calls)

                yield True
                return

            if usage:
                yield ResponseCompletedEvent(usage=usage)

            yield False

        except HTTPValidationError as e:
            raise Exception(f"Mistral API Validation Error: {e!s}") from e
        except SDKError as e:
            raise Exception(f"Mistral SDK Error: {e!s}") from e
        except Exception as e:
            raise Exception(f"Unexpected error from Mistral: {e!s}") from e

    def handle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ) -> Any | None:
        try:
            formatted_messages = [self._format_files_for_mistral(msg.copy()) for msg in messages]

            mistral_messages = [self._format_message(msg) for msg in formatted_messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": mistral_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            }

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
                if response_format is not None:
                    raise ValueError("Mistral does not support structured output (response_format) when tools are present.")

            if response_format is not None and not tools:
                from pydantic import BaseModel

                if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                    pydantic_model = response_format
                    if hasattr(pydantic_model, "model_json_schema"):
                        json_schema = pydantic_model.model_json_schema()
                        request_params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {"name": pydantic_model.__name__, "schema": json_schema},
                        }
                    else:
                        request_params["response_format"] = response_format
                else:
                    request_params["response_format"] = response_format

            api_response: ChatCompletionResponse = self.client.chat.complete(**request_params)

            message: AssistantMessage = api_response.choices[0].message

            if message.tool_calls:
                tool_usage: dict[str, Any] = api_response.usage.__dict__ if api_response.usage else {}
                self._accumulate_metrics(tool_usage)

                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    )

                messages.append({"role": message.role, "content": message.content, "tool_calls": tool_calls})

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if message.tool_calls:
                        first_tool_call = message.tool_calls[0]
                        function_name = first_tool_call.function.name
                        try:
                            arguments_raw = first_tool_call.function.arguments
                            if arguments_raw:
                                arguments_str = arguments_raw if isinstance(arguments_raw, str) else json.dumps(arguments_raw)
                                function_args = json.loads(arguments_str)
                            else:
                                function_args = {}
                            if function_args is None:
                                function_args = {}
                        except (json.JSONDecodeError, TypeError):
                            function_args = {}

                        messages[-1]["tool_calls"] = [tool_calls[0]]

                        tool_call_dict = tool_calls[0]
                        results = self._execute_tools_parallel_sync(tool_executor, [tool_call_dict])
                        tool_result = results.get(function_name, f"Tool '{function_name}' execution failed")

                        tool_outputs = {function_name: tool_result}

                        first_tool_call_for_storage = tool_call_dict

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": first_tool_call.id,
                                "content": tool_result,
                            }
                        )

                        if hasattr(tool_executor, "_tool_outputs"):
                            tool_executor._tool_outputs.update(tool_outputs)
                        if hasattr(tool_executor, "_tool_calls"):
                            tool_executor._tool_calls.append(first_tool_call_for_storage)

                        return None
                else:
                    tool_outputs = self._execute_tools_parallel_sync(tool_executor, tool_calls)

                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

                    if hasattr(tool_executor, "_tool_outputs"):
                        tool_executor._tool_outputs.update(tool_outputs)
                    if hasattr(tool_executor, "_tool_calls"):
                        tool_executor._tool_calls.extend(tool_calls)

                    return None
            else:
                accumulated_content_parts = []

                for msg in messages:
                    if msg.get("role") == "assistant" and msg.get("content"):
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(str(item) if not isinstance(item, dict) else item.get("text", str(item)) for item in content).strip()
                        else:
                            content = str(content).strip()
                        if content:
                            accumulated_content_parts.append(content)

                final_content = message.content or ""
                if isinstance(final_content, list):
                    final_content = " ".join(
                        str(item) if not isinstance(item, dict) else item.get("text", str(item)) for item in final_content
                    ).strip()
                else:
                    final_content = str(final_content).strip()
                if final_content:
                    accumulated_content_parts.append(final_content)

                content = "\n\n".join(accumulated_content_parts) if accumulated_content_parts else ""

                messages.append({"role": message.role, "content": final_content})

                tool_calls = []
                if hasattr(tool_executor, "_tool_calls"):
                    tool_calls = tool_executor._tool_calls.copy()

                tool_outputs = {}
                if hasattr(tool_executor, "_tool_outputs"):
                    tool_outputs = tool_executor._tool_outputs.copy()

                final_usage = api_response.usage.__dict__ if api_response.usage else {}

                self._accumulate_metrics(final_usage)

                cumulative_metrics = self._get_cumulative_metrics()

                return self._create_llm_response(
                    content,
                    {
                        "input_tokens": cumulative_metrics.input_tokens,
                        "output_tokens": cumulative_metrics.output_tokens,
                        "model": self.model,
                        "finish_reason": api_response.choices[0].finish_reason or "stop",
                        "params": {
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "max_tokens": self.max_tokens,
                        },
                    },
                    tool_calls,
                    tool_outputs,
                )

        except HTTPValidationError as e:
            raise Exception(f"Mistral API Validation Error: {e!s}") from e
        except SDKError as e:
            raise Exception(f"Mistral SDK Error: {e!s}") from e
        except Exception as e:
            raise Exception(f"Unexpected error from Mistral: {e!s}") from e

        return None

    def handle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        try:
            tool_calls: list[dict[str, Any]] = []
            usage = None

            formatted_messages = [self._format_files_for_mistral(msg.copy()) for msg in messages]

            mistral_messages = [self._format_message(msg) for msg in formatted_messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": mistral_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            }

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
                if response_format is not None:
                    raise ValueError("Mistral does not support structured output (response_format) when tools are present.")

            if response_format is not None and not tools:
                from pydantic import BaseModel

                if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                    pydantic_model = response_format
                    if hasattr(pydantic_model, "model_json_schema"):
                        json_schema = pydantic_model.model_json_schema()
                        request_params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {"name": pydantic_model.__name__, "schema": json_schema},
                        }
                    else:
                        request_params["response_format"] = response_format
                else:
                    request_params["response_format"] = response_format

            stream = self.client.chat.stream(**request_params)

            try:
                for chunk in stream:
                    try:
                        if hasattr(chunk, "data") and chunk.data:
                            if hasattr(chunk.data, "usage") and chunk.data.usage:
                                usage_obj = chunk.data.usage
                                usage = {
                                    "input_tokens": usage_obj.prompt_tokens if hasattr(usage_obj, "prompt_tokens") else 0,
                                    "output_tokens": usage_obj.completion_tokens if hasattr(usage_obj, "completion_tokens") else 0,
                                    "total_tokens": usage_obj.total_tokens if hasattr(usage_obj, "total_tokens") else 0,
                                }
                                self._accumulate_metrics(usage)
                                tool_executor._streaming_usage = usage

                            if not chunk.data.choices:
                                if usage:
                                    yield ResponseCompletedEvent(usage=usage)
                                continue

                            if chunk.data.choices and len(chunk.data.choices) > 0:
                                choice = chunk.data.choices[0]
                                delta = choice.delta

                                if hasattr(chunk.data, "usage") and chunk.data.usage and not usage:
                                    usage_obj = chunk.data.usage
                                    usage = {
                                        "input_tokens": usage_obj.prompt_tokens if hasattr(usage_obj, "prompt_tokens") else 0,
                                        "output_tokens": usage_obj.completion_tokens if hasattr(usage_obj, "completion_tokens") else 0,
                                        "total_tokens": usage_obj.total_tokens if hasattr(usage_obj, "total_tokens") else 0,
                                    }
                                    self._accumulate_metrics(usage)
                                    tool_executor._streaming_usage = usage

                                if delta.content:
                                    try:
                                        if isinstance(delta.content, str):
                                            yield ContentEvent(content=delta.content)
                                        elif isinstance(delta.content, list):
                                            text_content = ""
                                            for item in delta.content:
                                                try:
                                                    if hasattr(item, "text"):
                                                        text_content += item.text
                                                    elif isinstance(item, dict):
                                                        if "text" in item:
                                                            text_content += item["text"]
                                                        elif "type" in item and item["type"] == "reference":
                                                            if "text" in item:
                                                                text_content += item["text"]
                                                            elif "content" in item:
                                                                text_content += str(item["content"])
                                                        else:
                                                            text_content += str(item)
                                                    elif isinstance(item, str):
                                                        text_content += item
                                                    else:
                                                        text_content += str(item)
                                                except Exception:
                                                    continue
                                            if text_content:
                                                yield ContentEvent(content=text_content)
                                    except Exception:
                                        continue

                            if delta.tool_calls:
                                try:
                                    for tool_call_delta in delta.tool_calls:
                                        index = tool_call_delta.index or 0

                                        while len(tool_calls) <= index:
                                            tool_calls.append(
                                                {
                                                    "id": "",
                                                    "type": "function",
                                                    "function": {"name": "", "arguments": ""},
                                                }
                                            )

                                        if tool_call_delta.id:
                                            if tool_call_delta.id and not tool_calls[index]["id"]:
                                                tool_calls[index]["id"] = tool_call_delta.id
                                            else:
                                                tool_calls[index]["id"] += tool_call_delta.id

                                        if tool_call_delta.function:
                                            func_delta = tool_call_delta.function
                                            if func_delta.name:
                                                if func_delta.name and not tool_calls[index]["function"]["name"]:
                                                    tool_calls[index]["function"]["name"] = func_delta.name
                                                else:
                                                    tool_calls[index]["function"]["name"] += func_delta.name
                                            if func_delta.arguments:
                                                if func_delta.arguments and not tool_calls[index]["function"]["arguments"]:
                                                    tool_calls[index]["function"]["arguments"] = func_delta.arguments
                                                else:
                                                    tool_calls[index]["function"]["arguments"] += func_delta.arguments
                                except Exception:
                                    continue

                            if choice.finish_reason:
                                yield FinishReasonEvent(finish_reason=choice.finish_reason)
                                if usage:
                                    yield ResponseCompletedEvent(usage=usage)
                                break
                    except Exception:
                        continue
            except Exception as stream_error:
                if "validation error" in str(stream_error).lower() or "pydantic" in str(stream_error).lower():
                    logger.warning(
                        f"Mistral Pydantic validation error (continuing): {stream_error!s}",
                        exc_info=True,
                    )
                else:
                    logger.warning(f"Mistral streaming error (continuing): {stream_error!s}", exc_info=True)

            if tool_calls and any(tc["function"]["name"] for tc in tool_calls):
                tool_calls_to_yield = tool_calls
                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls and len(tool_calls) > 1:
                    tool_calls_to_yield = [tool_calls[0]]

                messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls_to_yield})

                yield ToolCallsEvent(tool_calls=tool_calls_to_yield)

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if tool_calls and tool_calls[0]["function"]["name"]:
                        first_tool = tool_calls[0]
                        function_name = first_tool["function"]["name"]
                        try:
                            function_args = json.loads(first_tool["function"]["arguments"]) if first_tool["function"]["arguments"] else {}
                            if function_args is None:
                                function_args = {}
                        except (json.JSONDecodeError, TypeError):
                            function_args = {}

                        tool_outputs_iter = self._execute_tools_parallel_sync(tool_executor, [first_tool])
                        tool_result = tool_outputs_iter.get(function_name, f"Tool '{function_name}' execution failed")

                        tool_outputs = {function_name: tool_result}

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": first_tool["id"],
                                "content": tool_result,
                            }
                        )

                        if hasattr(tool_executor, "_tool_outputs"):
                            tool_executor._tool_outputs.update(tool_outputs)
                        if hasattr(tool_executor, "_tool_calls"):
                            tool_executor._tool_calls.append(first_tool)

                        yield ToolOutputsEvent(tool_outputs=tool_outputs)

                        if usage:
                            yield ResponseCompletedEvent(usage=usage)

                        yield True
                        return
                else:
                    tool_outputs = self._execute_tools_parallel_sync(tool_executor, tool_calls)

                    for tool_call in tool_calls:
                        if tool_call["function"]["name"]:
                            function_name = tool_call["function"]["name"]
                            tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": tool_result,
                                }
                            )

                    yield ToolOutputsEvent(tool_outputs=tool_outputs)

                    if usage:
                        yield ResponseCompletedEvent(usage=usage)

                if hasattr(tool_executor, "_tool_outputs"):
                    tool_executor._tool_outputs.update(tool_outputs)
                if hasattr(tool_executor, "_tool_calls"):
                    tool_executor._tool_calls.extend(tool_calls)

                yield True
                return

            if usage:
                yield ResponseCompletedEvent(usage=usage)

            yield False

        except HTTPValidationError as e:
            raise Exception(f"Mistral API Validation Error: {e!s}") from e
        except SDKError as e:
            raise Exception(f"Mistral SDK Error: {e!s}") from e
        except Exception as e:
            raise Exception(f"Unexpected error from Mistral: {e!s}") from e

    def _format_files_for_mistral(self, message: dict[str, Any]) -> dict[str, Any]:
        from hypertic.utils.files import File, FileType

        if "_file_objects" not in message:
            return message

        file_objects: list[File] = message.get("_file_objects", [])
        if not file_objects:
            return message

        content_chunks: list[Any] = []
        message_content = message.get("content")
        if message_content:
            if isinstance(message_content, str):
                content_chunks.append(TextChunk(text=message_content))
            elif isinstance(message_content, list):
                if all(isinstance(item, (TextChunk, ImageURLChunk, DocumentURLChunk)) for item in message_content):
                    content_chunks = message_content
                else:
                    for item in message_content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                content_chunks.append(TextChunk(text=item.get("text", "")))
                            elif item.get("type") == "image_url":
                                image_url_val = item.get("image_url", {})
                                url = image_url_val.get("url", "") if isinstance(image_url_val, dict) else image_url_val
                                content_chunks.append(ImageURLChunk(image_url=url))
                            elif item.get("type") == "document_url":
                                doc_url_val = item.get("document_url", {})
                                doc_url = doc_url_val.get("url", "") if isinstance(doc_url_val, dict) else doc_url_val
                                content_chunks.append(DocumentURLChunk(document_url=doc_url))
                        else:
                            content_chunks.append(item)

        for file_obj in file_objects:
            if file_obj.file_type == FileType.IMAGE:
                if file_obj.url:
                    content_chunks.append(ImageURLChunk(image_url=file_obj.url))
                else:
                    base64_data = file_obj.to_base64()
                    if base64_data:
                        mime_type = file_obj.mime_type or "image/jpeg"
                        data_url = f"data:{mime_type};base64,{base64_data}"
                        content_chunks.append(ImageURLChunk(image_url=data_url))
            elif file_obj.file_type == FileType.DOCUMENT:
                if file_obj.url:
                    content_chunks.append(DocumentURLChunk(document_url=file_obj.url))
                else:
                    base64_data = file_obj.to_base64()
                    if base64_data:
                        mime_type = file_obj.mime_type or "application/pdf"
                        data_url = f"data:{mime_type};base64,{base64_data}"
                        content_chunks.append(DocumentURLChunk(document_url=data_url))
            else:
                from hypertic.utils.log import get_logger

                logger = get_logger(__name__)
                logger.warning(
                    f"File type {file_obj.file_type} not supported in Mistral Chat API "
                    f"(only images and documents are supported): {file_obj.url or file_obj.filepath}"
                )

        if not content_chunks:
            message["content"] = ""
        elif len(content_chunks) == 1 and isinstance(content_chunks[0], TextChunk):
            message["content"] = content_chunks[0].text
        else:
            message["content"] = content_chunks

        message.pop("_file_objects", None)
        message.pop("files", None)
        return message
