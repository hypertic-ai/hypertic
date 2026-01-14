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
    from groq import (
        APIError,
        APIResponseValidationError,
        APIStatusError,
        AsyncGroq as AsyncGroqClient,
        Groq as GroqClient,
    )
    from groq.types.chat import ChatCompletion
    from groq.types.chat.chat_completion_chunk import ChatCompletionChunk
except ImportError as err:
    raise ImportError("`groq` not installed. Please install using `pip install groq`") from err


@dataclass
class Groq(Base):
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None

    reasoning_format: str | None = None
    thinking: bool | None = None
    reasoning_effort: str | None = None

    client: Any = field(default=None, init=False)
    async_client: Any = field(default=None, init=False)

    def __post_init__(self):
        if self.reasoning_format is not None and self.thinking is not None:
            raise ValueError("reasoning_format and thinking are mutually exclusive. Use either one, not both.")

        self.api_key = self.api_key or getenv("GROQ_API_KEY")

        if self.api_key is not None:
            self.client = GroqClient(api_key=self.api_key)
            self.async_client = AsyncGroqClient(api_key=self.api_key)
        else:
            self.client = GroqClient()
            self.async_client = AsyncGroqClient()

    async def ahandle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ) -> Any | None:
        formatted_messages = [self._format_files_for_groq(msg.copy()) for msg in messages]

        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
        }

        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        if self.top_p is not None:
            request_params["top_p"] = self.top_p
        if self.presence_penalty is not None:
            request_params["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            request_params["frequency_penalty"] = self.frequency_penalty
        if self.max_tokens is not None:
            request_params["max_tokens"] = self.max_tokens

        if self.reasoning_format is not None:
            request_params["reasoning_format"] = self.reasoning_format
        if self.thinking is not None:
            request_params["include_reasoning"] = self.thinking
        if self.reasoning_effort is not None:
            request_params["reasoning_effort"] = self.reasoning_effort

        if tools:
            request_params["tools"] = tools
            if hasattr(tool_executor, "parallel_calls"):
                request_params["parallel_tool_calls"] = tool_executor.parallel_calls
            else:
                request_params["parallel_tool_calls"] = True
            if response_format is not None:
                raise ValueError("Groq does not support structured output (response_format) when tools are present.")

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

        try:
            api_response: ChatCompletion = await self.async_client.chat.completions.create(**request_params)

            message = api_response.choices[0].message

            if message.tool_calls:
                usage: dict[str, Any] = {}
                if api_response.usage is not None:
                    usage = {
                        "input_tokens": api_response.usage.prompt_tokens,
                        "output_tokens": api_response.usage.completion_tokens,
                        "total_tokens": api_response.usage.total_tokens,
                    }
                self._accumulate_metrics(usage)

                tool_calls = [tool_call.model_dump() for tool_call in message.tool_calls]

                messages.append({"role": message.role, "content": message.content, "tool_calls": tool_calls})

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if message.tool_calls:
                        first_tool_call = message.tool_calls[0]
                        function_name = first_tool_call.function.name
                        try:
                            function_args = json.loads(first_tool_call.function.arguments)
                            if function_args is None:
                                function_args = {}
                        except (json.JSONDecodeError, TypeError):
                            function_args = {}

                        messages[-1]["tool_calls"] = [first_tool_call.model_dump()]

                        tool_call_dict = first_tool_call.model_dump()
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
                tool_calls = []
                if hasattr(tool_executor, "_tool_calls"):
                    tool_calls = tool_executor._tool_calls.copy()

                tool_outputs = {}
                if hasattr(tool_executor, "_tool_outputs"):
                    tool_outputs = tool_executor._tool_outputs.copy()

                final_usage: dict[str, Any] = {}
                if api_response.usage is not None:
                    final_usage = {
                        "input_tokens": api_response.usage.prompt_tokens,
                        "output_tokens": api_response.usage.completion_tokens,
                        "total_tokens": api_response.usage.total_tokens,
                    }
                self._accumulate_metrics(final_usage)

                cumulative_metrics = self._get_cumulative_metrics()

                content = message.content or ""
                reasoning_content = ""

                if hasattr(message, "reasoning") and message.reasoning:
                    reasoning_content = message.reasoning

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
                if self.reasoning_format is not None:
                    params["reasoning_format"] = self.reasoning_format
                if self.thinking is not None:
                    params["thinking"] = self.thinking
                if self.reasoning_effort is not None:
                    params["reasoning_effort"] = self.reasoning_effort

                usage_info = {
                    "model": self.model,
                    "params": params,
                    "finish_reason": api_response.choices[0].finish_reason or "stop",
                    "input_tokens": cumulative_metrics.input_tokens,
                    "output_tokens": cumulative_metrics.output_tokens,
                }

                if reasoning_content:
                    usage_info["reasoning_content"] = reasoning_content

                return self._create_llm_response(content, usage_info, tool_calls, tool_outputs)

        except (APIResponseValidationError, APIStatusError) as e:
            error_message = str(e)
            status_code = getattr(e, "status_code", 500)
            if hasattr(e, "response") and e.response:
                try:
                    error_data = e.response.json() if hasattr(e.response, "json") else {}
                    error_message = error_data.get("error", {}).get("message", str(e))
                    status_code = e.response.status_code
                except Exception:
                    pass
            raise Exception(f"Groq API Error ({status_code}): {error_message}") from e
        except APIError as e:
            error_message = getattr(e, "message", str(e))
            raise Exception(f"Groq API Error: {error_message}") from e
        except Exception as e:
            raise Exception(f"Unexpected error from Groq: {e!s}") from e

    async def ahandle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        tool_calls: list[dict[str, Any]] = []

        try:
            formatted_messages = [self._format_files_for_groq(msg.copy()) for msg in messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": formatted_messages,
                "stream": True,
            }

            if self.temperature is not None:
                request_params["temperature"] = self.temperature
            if self.top_p is not None:
                request_params["top_p"] = self.top_p
            if self.presence_penalty is not None:
                request_params["presence_penalty"] = self.presence_penalty
            if self.frequency_penalty is not None:
                request_params["frequency_penalty"] = self.frequency_penalty
            if self.max_tokens is not None:
                request_params["max_tokens"] = self.max_tokens

            if self.reasoning_format is not None:
                request_params["reasoning_format"] = self.reasoning_format
            if self.thinking is not None:
                request_params["include_reasoning"] = self.thinking
            if self.reasoning_effort is not None:
                request_params["reasoning_effort"] = self.reasoning_effort

            if tools:
                request_params["tools"] = tools
                if hasattr(tool_executor, "parallel_calls"):
                    request_params["parallel_tool_calls"] = tool_executor.parallel_calls
                else:
                    request_params["parallel_tool_calls"] = True
                if response_format is not None:
                    raise ValueError("Groq does not support structured output (response_format) when tools are present.")

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

            stream = await self.async_client.chat.completions.create(**request_params)

            async for chunk in stream:
                async for event in self._process_streaming_event_async(chunk, tool_calls, tool_executor, messages):
                    yield event

        except (APIResponseValidationError, APIStatusError) as e:
            error_message = str(e)
            status_code = getattr(e, "status_code", 500)
            if hasattr(e, "response") and e.response:
                try:
                    error_data = e.response.json() if hasattr(e.response, "json") else {}
                    error_message = error_data.get("error", {}).get("message", str(e))
                    status_code = e.response.status_code
                except Exception:
                    pass
            raise Exception(f"Groq API Error ({status_code}): {error_message}") from e
        except APIError as e:
            error_message = getattr(e, "message", str(e))
            raise Exception(f"Groq API Error: {error_message}") from e
        except Exception as e:
            raise Exception(f"Unexpected error from Groq: {e!s}") from e

    async def _process_streaming_event_async(
        self,
        chunk: ChatCompletionChunk,
        tool_calls: list[dict[str, Any]],
        tool_executor: Any,
        messages: list[dict[str, Any]],
    ):
        if not chunk.choices:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage = {
                    "input_tokens": chunk.usage.prompt_tokens if hasattr(chunk.usage, "prompt_tokens") else 0,
                    "output_tokens": chunk.usage.completion_tokens if hasattr(chunk.usage, "completion_tokens") else 0,
                    "total_tokens": chunk.usage.total_tokens if hasattr(chunk.usage, "total_tokens") else 0,
                }
                self._accumulate_metrics(usage)
                tool_executor._streaming_usage = usage
                yield ResponseCompletedEvent(usage=usage)
            return

        choice = chunk.choices[0]
        delta = choice.delta

        if hasattr(chunk, "usage") and chunk.usage is not None:
            usage = {
                "input_tokens": chunk.usage.prompt_tokens if hasattr(chunk.usage, "prompt_tokens") else 0,
                "output_tokens": chunk.usage.completion_tokens if hasattr(chunk.usage, "completion_tokens") else 0,
                "total_tokens": chunk.usage.total_tokens if hasattr(chunk.usage, "total_tokens") else 0,
            }
            self._accumulate_metrics(usage)
            tool_executor._streaming_usage = usage

        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                index = tool_call_delta.index or 0

                while len(tool_calls) <= index:
                    tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

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

        content = delta.content
        reasoning_content = ""

        if hasattr(delta, "reasoning") and delta.reasoning:
            reasoning_content = delta.reasoning

        if content and content.strip():
            yield ContentEvent(content=content)

        if reasoning_content and reasoning_content.strip():
            yield ReasoningEvent(reasoning=reasoning_content)

        if choice.finish_reason:
            yield FinishReasonEvent(finish_reason=choice.finish_reason)

        if choice.finish_reason in ["stop", "tool_calls"]:
            if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

            if tool_calls and any(tc["function"]["name"] for tc in tool_calls):
                messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

                yield ToolCallsEvent(tool_calls=tool_calls)

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if tool_calls and tool_calls[0]["function"]["name"]:
                        first_tool = tool_calls[0]
                        function_name = first_tool["function"]["name"]
                        try:
                            function_args = json.loads(first_tool["function"]["arguments"])
                            if function_args is None:
                                function_args = {}
                        except (json.JSONDecodeError, TypeError):
                            function_args = {}

                        messages[-1]["tool_calls"] = [first_tool]

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

                if hasattr(tool_executor, "_tool_outputs"):
                    tool_executor._tool_outputs.update(tool_outputs)
                if hasattr(tool_executor, "_tool_calls"):
                    tool_executor._tool_calls.extend(tool_calls)

                yield True
                return

            if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

            yield False

    def handle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ) -> Any | None:
        formatted_messages = [self._format_files_for_groq(msg.copy()) for msg in messages]

        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
        }

        if self.temperature is not None:
            request_params["temperature"] = self.temperature
        if self.top_p is not None:
            request_params["top_p"] = self.top_p
        if self.presence_penalty is not None:
            request_params["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            request_params["frequency_penalty"] = self.frequency_penalty
        if self.max_tokens is not None:
            request_params["max_tokens"] = self.max_tokens

        if self.reasoning_format is not None:
            request_params["reasoning_format"] = self.reasoning_format
        if self.thinking is not None:
            request_params["include_reasoning"] = self.thinking
        if self.reasoning_effort is not None:
            request_params["reasoning_effort"] = self.reasoning_effort

        if tools:
            request_params["tools"] = tools
            if hasattr(tool_executor, "parallel_calls"):
                request_params["parallel_tool_calls"] = tool_executor.parallel_calls
            else:
                request_params["parallel_tool_calls"] = True
            if response_format is not None:
                raise ValueError("Groq does not support structured output (response_format) when tools are present.")

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

        try:
            api_response: ChatCompletion = self.client.chat.completions.create(**request_params)

            message = api_response.choices[0].message

            if message.tool_calls:
                usage: dict[str, Any] = {}
                if api_response.usage is not None:
                    usage = {
                        "input_tokens": api_response.usage.prompt_tokens,
                        "output_tokens": api_response.usage.completion_tokens,
                        "total_tokens": api_response.usage.total_tokens,
                    }
                self._accumulate_metrics(usage)

                tool_calls = [tool_call.model_dump() for tool_call in message.tool_calls]

                messages.append({"role": message.role, "content": message.content, "tool_calls": tool_calls})

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if message.tool_calls:
                        first_tool_call = message.tool_calls[0]
                        function_name = first_tool_call.function.name
                        try:
                            function_args = json.loads(first_tool_call.function.arguments)
                            if function_args is None:
                                function_args = {}
                        except (json.JSONDecodeError, TypeError):
                            function_args = {}

                        messages[-1]["tool_calls"] = [first_tool_call.model_dump()]

                        tool_call_dict = first_tool_call.model_dump()
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
                    tool_calls = [tc.model_dump() for tc in message.tool_calls]
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
                tool_calls = []
                if hasattr(tool_executor, "_tool_calls"):
                    tool_calls = tool_executor._tool_calls.copy()

                tool_outputs = {}
                if hasattr(tool_executor, "_tool_outputs"):
                    tool_outputs = tool_executor._tool_outputs.copy()

                final_usage: dict[str, Any] = {}
                if api_response.usage is not None:
                    final_usage = {
                        "input_tokens": api_response.usage.prompt_tokens,
                        "output_tokens": api_response.usage.completion_tokens,
                        "total_tokens": api_response.usage.total_tokens,
                    }
                self._accumulate_metrics(final_usage)

                cumulative_metrics = self._get_cumulative_metrics()

                content = message.content or ""
                reasoning_content = ""

                if hasattr(message, "reasoning") and message.reasoning:
                    reasoning_content = message.reasoning

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
                if self.reasoning_format is not None:
                    params["reasoning_format"] = self.reasoning_format
                if self.thinking is not None:
                    params["thinking"] = self.thinking
                if self.reasoning_effort is not None:
                    params["reasoning_effort"] = self.reasoning_effort

                usage_info = {
                    "model": self.model,
                    "params": params,
                    "finish_reason": api_response.choices[0].finish_reason or "stop",
                    "input_tokens": cumulative_metrics.input_tokens,
                    "output_tokens": cumulative_metrics.output_tokens,
                }

                if reasoning_content:
                    usage_info["reasoning_content"] = reasoning_content

                return self._create_llm_response(content, usage_info, tool_calls, tool_outputs)

        except (APIResponseValidationError, APIStatusError) as e:
            error_message = str(e)
            status_code = getattr(e, "status_code", 500)
            if hasattr(e, "response") and e.response:
                try:
                    error_data = e.response.json() if hasattr(e.response, "json") else {}
                    error_message = error_data.get("error", {}).get("message", str(e))
                    status_code = e.response.status_code
                except Exception:
                    pass
            raise Exception(f"Groq API Error ({status_code}): {error_message}") from e
        except APIError as e:
            error_message = getattr(e, "message", str(e))
            raise Exception(f"Groq API Error: {error_message}") from e
        except Exception as e:
            raise Exception(f"Unexpected error from Groq: {e!s}") from e

    def handle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        tool_calls: list[dict[str, Any]] = []

        try:
            formatted_messages = [self._format_files_for_groq(msg.copy()) for msg in messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": formatted_messages,
                "stream": True,
            }

            if self.temperature is not None:
                request_params["temperature"] = self.temperature
            if self.top_p is not None:
                request_params["top_p"] = self.top_p
            if self.presence_penalty is not None:
                request_params["presence_penalty"] = self.presence_penalty
            if self.frequency_penalty is not None:
                request_params["frequency_penalty"] = self.frequency_penalty
            if self.max_tokens is not None:
                request_params["max_tokens"] = self.max_tokens

            if self.reasoning_format is not None:
                request_params["reasoning_format"] = self.reasoning_format
            if self.thinking is not None:
                request_params["include_reasoning"] = self.thinking
            if self.reasoning_effort is not None:
                request_params["reasoning_effort"] = self.reasoning_effort

            if tools:
                request_params["tools"] = tools
                if hasattr(tool_executor, "parallel_calls"):
                    request_params["parallel_tool_calls"] = tool_executor.parallel_calls
                else:
                    request_params["parallel_tool_calls"] = True
                if response_format is not None:
                    raise ValueError("Groq does not support structured output (response_format) when tools are present.")

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

            stream = self.client.chat.completions.create(**request_params)

            for chunk in stream:
                yield from self._process_streaming_event_sync(chunk, tool_calls, tool_executor, messages)

        except (APIResponseValidationError, APIStatusError) as e:
            error_message = str(e)
            status_code = getattr(e, "status_code", 500)
            if hasattr(e, "response") and e.response:
                try:
                    error_data = e.response.json() if hasattr(e.response, "json") else {}
                    error_message = error_data.get("error", {}).get("message", str(e))
                    status_code = e.response.status_code
                except Exception:
                    pass
            raise Exception(f"Groq API Error ({status_code}): {error_message}") from e
        except APIError as e:
            error_message = getattr(e, "message", str(e))
            raise Exception(f"Groq API Error: {error_message}") from e
        except Exception as e:
            raise Exception(f"Unexpected error from Groq: {e!s}") from e

    def _process_streaming_event_sync(
        self,
        chunk: ChatCompletionChunk,
        tool_calls: list[dict[str, Any]],
        tool_executor: Any,
        messages: list[dict[str, Any]],
    ):
        if not chunk.choices:
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage = {
                    "input_tokens": chunk.usage.prompt_tokens if hasattr(chunk.usage, "prompt_tokens") else 0,
                    "output_tokens": chunk.usage.completion_tokens if hasattr(chunk.usage, "completion_tokens") else 0,
                    "total_tokens": chunk.usage.total_tokens if hasattr(chunk.usage, "total_tokens") else 0,
                }
                self._accumulate_metrics(usage)
                tool_executor._streaming_usage = usage
                yield ResponseCompletedEvent(usage=usage)
            return

        choice = chunk.choices[0]
        delta = choice.delta

        if hasattr(chunk, "usage") and chunk.usage is not None:
            usage = {
                "input_tokens": chunk.usage.prompt_tokens if hasattr(chunk.usage, "prompt_tokens") else 0,
                "output_tokens": chunk.usage.completion_tokens if hasattr(chunk.usage, "completion_tokens") else 0,
                "total_tokens": chunk.usage.total_tokens if hasattr(chunk.usage, "total_tokens") else 0,
            }
            self._accumulate_metrics(usage)
            tool_executor._streaming_usage = usage

        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                index = tool_call_delta.index or 0

                while len(tool_calls) <= index:
                    tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

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

        content = delta.content
        reasoning_content = ""

        if hasattr(delta, "reasoning") and delta.reasoning:
            reasoning_content = delta.reasoning

        if content and content.strip():
            yield ContentEvent(content=content)

        if reasoning_content and reasoning_content.strip():
            yield ReasoningEvent(reasoning=reasoning_content)

        if choice.finish_reason:
            yield FinishReasonEvent(finish_reason=choice.finish_reason)

        if choice.finish_reason in ["stop", "tool_calls"]:
            if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

            if tool_calls and any(tc["function"]["name"] for tc in tool_calls):
                messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

                yield ToolCallsEvent(tool_calls=tool_calls)

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if tool_calls and tool_calls[0]["function"]["name"]:
                        first_tool = tool_calls[0]
                        function_name = first_tool["function"]["name"]
                        try:
                            function_args = json.loads(first_tool["function"]["arguments"])
                            if function_args is None:
                                function_args = {}
                        except (json.JSONDecodeError, TypeError):
                            function_args = {}

                        messages[-1]["tool_calls"] = [first_tool]

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

                if hasattr(tool_executor, "_tool_outputs"):
                    tool_executor._tool_outputs.update(tool_outputs)
                if hasattr(tool_executor, "_tool_calls"):
                    tool_executor._tool_calls.extend(tool_calls)

                yield True
                return

            if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

            yield False

    def _format_files_for_groq(self, message: dict[str, Any]) -> dict[str, Any]:
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
                    content.append({"type": "image_url", "image_url": {"url": file_obj.url}})
                else:
                    base64_data = file_obj.to_base64()
                    if base64_data:
                        mime_type = file_obj.mime_type or "image/jpeg"
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                            }
                        )
            else:
                from hypertic.utils.log import get_logger

                logger = get_logger(__name__)
                logger.warning(
                    f"File type {file_obj.file_type} not supported in Groq Chat API "
                    f"(only images are supported in chat completions): {file_obj.url or file_obj.filepath}"
                )

        if not content:
            message["content"] = ""
        elif len(content) == 1 and content[0].get("type") == "text":
            message["content"] = content[0]["text"]
        else:
            message["content"] = content

        message.pop("_file_objects", None)
        message.pop("files", None)
        return message
