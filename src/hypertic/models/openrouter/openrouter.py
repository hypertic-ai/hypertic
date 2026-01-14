from dataclasses import dataclass, field
from os import getenv
from typing import Any

from hypertic.models.base import Base
from hypertic.models.events import (
    ContentEvent,
    FinishReasonEvent,
    ToolCallsEvent,
    ToolOutputsEvent,
)

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        AsyncOpenAI as AsyncOpenAIClient,
        OpenAI as OpenAIClient,
        RateLimitError,
    )
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
except (ImportError, ModuleNotFoundError) as err:
    raise ImportError("`openai` not installed. Please install using `pip install openai`") from err


@dataclass
class OpenRouter(Base):
    api_key: str | None = None
    model: str | None = None
    api_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None

    reasoning_effort: str | None = None

    async_client: Any = field(default=None, init=False)
    client: Any = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("OPENROUTER_API_KEY")

        base_url = "https://openrouter.ai/api/v1/"
        if self.api_key is not None:
            self.client = OpenAIClient(api_key=self.api_key, base_url=base_url)
            self.async_client = AsyncOpenAIClient(api_key=self.api_key, base_url=base_url)
        else:
            self.client = OpenAIClient(base_url=base_url)
            self.async_client = AsyncOpenAIClient(base_url=base_url)

    async def ahandle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ) -> Any | None:
        formatted_messages = [self._format_files_for_openrouter(msg.copy()) for msg in messages]

        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
        }

        if tools:
            request_params["tools"] = tools
            if hasattr(tool_executor, "parallel_calls"):
                request_params["parallel_tool_calls"] = tool_executor.parallel_calls
            else:
                request_params["parallel_tool_calls"] = True

        if response_format is not None:
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

        if self.reasoning_effort is not None:
            request_params["reasoning_effort"] = self.reasoning_effort
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

        try:
            response_obj: ChatCompletion = await self.async_client.chat.completions.create(**request_params)

            message = response_obj.choices[0].message

            content = message.content or ""

            messages.append(
                {
                    "role": message.role,
                    "content": content,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None,
                }
            )

            if message.tool_calls:
                usage_obj = response_obj.usage
                usage = {
                    "input_tokens": usage_obj.prompt_tokens if usage_obj else 0,
                    "output_tokens": usage_obj.completion_tokens if usage_obj else 0,
                    "total_tokens": usage_obj.total_tokens if usage_obj else 0,
                }
                self._accumulate_metrics(usage)

                tool_calls = [tool_call.model_dump() for tool_call in message.tool_calls]

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if len(tool_calls) > 1:
                        first_tool_call = message.tool_calls[0]
                        function = getattr(first_tool_call, "function", None)
                        if function is None:
                            return None
                        function_name = function.name

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
                tool_outputs = await self._execute_tools_parallel_async(tool_executor, tool_calls)

                for tool_call in message.tool_calls:
                    function = getattr(tool_call, "function", None)
                    if function is None:
                        continue
                    function_name = function.name
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

                usage_obj_final = response_obj.usage
                usage = {
                    "input_tokens": usage_obj_final.prompt_tokens if usage_obj_final else 0,
                    "output_tokens": usage_obj_final.completion_tokens if usage_obj_final else 0,
                    "total_tokens": usage_obj_final.total_tokens if usage_obj_final else 0,
                }
                self._accumulate_metrics(usage)

                cumulative_metrics = self._get_cumulative_metrics()

                content = message.content or ""

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
                    params["reasoning_effort"] = self.reasoning_effort

                return self._create_llm_response(
                    content,
                    {
                        "model": self.model,
                        "params": params,
                        "finish_reason": response_obj.choices[0].finish_reason or "stop",
                        "input_tokens": cumulative_metrics.input_tokens,
                        "output_tokens": cumulative_metrics.output_tokens,
                    },
                    tool_calls,
                    tool_outputs,
                )
        except RateLimitError as e:
            raise Exception(f"Rate limit error from OpenRouter SDK: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from OpenRouter SDK: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from OpenRouter SDK: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from OpenRouter SDK: {e}") from e

    async def ahandle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        try:
            formatted_messages = [self._format_files_for_openrouter(msg.copy()) for msg in messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": formatted_messages,
                "stream": True,
            }
            tool_calls: list[dict[str, Any]] = []

            if tools:
                request_params["tools"] = tools
                if hasattr(tool_executor, "parallel_calls"):
                    request_params["parallel_tool_calls"] = tool_executor.parallel_calls
                else:
                    request_params["parallel_tool_calls"] = True

            if response_format is not None:
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

            if self.reasoning_effort is not None:
                request_params["reasoning_effort"] = self.reasoning_effort
            if self.temperature is not None:
                request_params["temperature"] = self.temperature
            if self.top_p is not None:
                request_params["top_p"] = self.top_p
            if self.presence_penalty is not None:
                request_params["presence_penalty"] = self.presence_penalty
            if self.frequency_penalty is not None:
                request_params["frequency_penalty"] = self.frequency_penalty
            if self.max_tokens is not None:
                request_params["max_completion_tokens"] = self.max_tokens

            request_params["stream_options"] = {"include_usage": True}

            stream = await self.async_client.chat.completions.create(**request_params)

            async for chunk in stream:
                async for event in self._process_streaming_event_async(chunk, tool_calls, tool_executor, messages):
                    yield event

        except RateLimitError as e:
            raise Exception(f"Rate limit error from OpenRouter SDK: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from OpenRouter SDK: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from OpenRouter SDK: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from OpenRouter SDK: {e}") from e

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
        has_content = bool(content and content.strip()) if content else False

        if has_content or choice.finish_reason:
            if has_content:
                yield ContentEvent(content=content or "")
            if choice.finish_reason:
                yield FinishReasonEvent(finish_reason=choice.finish_reason)

        if choice.finish_reason in ["stop", "tool_calls"]:
            if tool_calls and any(tc["function"]["name"] for tc in tool_calls):
                is_sequential = hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls and len(tool_calls) > 1

                if is_sequential:
                    first_tool = tool_calls[0]
                    function_name = first_tool["function"]["name"]

                    messages.append({"role": "assistant", "content": None, "tool_calls": [first_tool]})

                    yield ToolCallsEvent(tool_calls=[first_tool])

                    tool_outputs = await self._execute_tools_parallel_async(tool_executor, [first_tool])
                    tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                    messages.append({"role": "tool", "tool_call_id": first_tool["id"], "content": tool_result})

                    yield ToolOutputsEvent(tool_outputs={function_name: tool_result})

                    if hasattr(tool_executor, "_tool_outputs"):
                        tool_executor._tool_outputs.update({function_name: tool_result})
                    if hasattr(tool_executor, "_tool_calls"):
                        tool_executor._tool_calls.append(first_tool)

                    yield True
                    return
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

                    yield ToolCallsEvent(tool_calls=tool_calls)

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

            yield False

    def handle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ) -> Any | None:
        formatted_messages = [self._format_files_for_openrouter(msg.copy()) for msg in messages]

        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
        }

        if tools:
            request_params["tools"] = tools
            if hasattr(tool_executor, "parallel_calls"):
                request_params["parallel_tool_calls"] = tool_executor.parallel_calls
            else:
                request_params["parallel_tool_calls"] = True

        if response_format is not None:
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

        if self.reasoning_effort is not None:
            request_params["reasoning_effort"] = self.reasoning_effort
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

        try:
            response_obj: ChatCompletion = self.client.chat.completions.create(**request_params)

            message = response_obj.choices[0].message

            content = message.content or ""

            messages.append(
                {
                    "role": message.role,
                    "content": content,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None,
                }
            )

            if message.tool_calls:
                usage_obj = response_obj.usage
                usage = {
                    "input_tokens": usage_obj.prompt_tokens if usage_obj else 0,
                    "output_tokens": usage_obj.completion_tokens if usage_obj else 0,
                    "total_tokens": usage_obj.total_tokens if usage_obj else 0,
                }
                self._accumulate_metrics(usage)

                tool_calls = [tool_call.model_dump() for tool_call in message.tool_calls]

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if len(tool_calls) > 1:
                        first_tool_call = message.tool_calls[0]
                        function = getattr(first_tool_call, "function", None)
                        if function is None:
                            return None
                        function_name = function.name

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

                tool_outputs = self._execute_tools_parallel_sync(tool_executor, tool_calls)

                for tool_call in message.tool_calls:
                    function = getattr(tool_call, "function", None)
                    if function is None:
                        continue
                    function_name = function.name
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

                usage_obj_final = response_obj.usage
                usage = {
                    "input_tokens": usage_obj_final.prompt_tokens if usage_obj_final else 0,
                    "output_tokens": usage_obj_final.completion_tokens if usage_obj_final else 0,
                    "total_tokens": usage_obj_final.total_tokens if usage_obj_final else 0,
                }
                self._accumulate_metrics(usage)

                cumulative_metrics = self._get_cumulative_metrics()

                content = message.content or ""

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
                    params["reasoning_effort"] = self.reasoning_effort

                return self._create_llm_response(
                    content,
                    {
                        "model": self.model,
                        "params": params,
                        "finish_reason": response_obj.choices[0].finish_reason or "stop",
                        "input_tokens": cumulative_metrics.input_tokens,
                        "output_tokens": cumulative_metrics.output_tokens,
                    },
                    tool_calls,
                    tool_outputs,
                )
        except RateLimitError as e:
            raise Exception(f"Rate limit error from OpenRouter SDK: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from OpenRouter SDK: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from OpenRouter SDK: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from OpenRouter SDK: {e}") from e

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
            formatted_messages = [self._format_files_for_openrouter(msg.copy()) for msg in messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": formatted_messages,
                "stream": True,
            }

            if tools:
                request_params["tools"] = tools
                if hasattr(tool_executor, "parallel_calls"):
                    request_params["parallel_tool_calls"] = tool_executor.parallel_calls
                else:
                    request_params["parallel_tool_calls"] = True

            if response_format is not None:
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

            if self.reasoning_effort is not None:
                request_params["reasoning_effort"] = self.reasoning_effort
            if self.temperature is not None:
                request_params["temperature"] = self.temperature
            if self.top_p is not None:
                request_params["top_p"] = self.top_p
            if self.presence_penalty is not None:
                request_params["presence_penalty"] = self.presence_penalty
            if self.frequency_penalty is not None:
                request_params["frequency_penalty"] = self.frequency_penalty
            if self.max_tokens is not None:
                request_params["max_completion_tokens"] = self.max_tokens

            request_params["stream_options"] = {"include_usage": True}

            stream = self.client.chat.completions.create(**request_params)

            for chunk in stream:
                yield from self._process_streaming_event_sync(chunk, tool_calls, tool_executor, messages)

        except RateLimitError as e:
            raise Exception(f"Rate limit error from OpenRouter SDK: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from OpenRouter SDK: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from OpenRouter SDK: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from OpenRouter SDK: {e}") from e

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
        has_content = bool(content and content.strip()) if content else False

        if has_content or choice.finish_reason:
            if has_content:
                yield ContentEvent(content=content or "")
            if choice.finish_reason:
                yield FinishReasonEvent(finish_reason=choice.finish_reason)

        if choice.finish_reason in ["stop", "tool_calls"]:
            if tool_calls and any(tc["function"]["name"] for tc in tool_calls):
                is_sequential = hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls and len(tool_calls) > 1

                if is_sequential:
                    first_tool = tool_calls[0]
                    function_name = first_tool["function"]["name"]

                    messages.append({"role": "assistant", "content": None, "tool_calls": [first_tool]})

                    yield ToolCallsEvent(tool_calls=[first_tool])

                    tool_outputs = self._execute_tools_parallel_sync(tool_executor, [first_tool])
                    tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                    messages.append({"role": "tool", "tool_call_id": first_tool["id"], "content": tool_result})

                    yield ToolOutputsEvent(tool_outputs={function_name: tool_result})

                    if hasattr(tool_executor, "_tool_outputs"):
                        tool_executor._tool_outputs.update({function_name: tool_result})
                    if hasattr(tool_executor, "_tool_calls"):
                        tool_executor._tool_calls.append(first_tool)

                    yield True
                    return
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

                    yield ToolCallsEvent(tool_calls=tool_calls)

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

            yield False

    def _format_files_for_openrouter(self, message: dict[str, Any]) -> dict[str, Any]:
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
                    f"File type {file_obj.file_type} may not be supported by the selected model. "
                    f"OpenRouter supports images for vision-capable models. "
                    f"File: {file_obj.url or file_obj.filepath}"
                )

        message["content"] = content

        message.pop("_file_objects", None)
        message.pop("files", None)
        return message
