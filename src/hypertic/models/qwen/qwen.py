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
class Qwen(Base):
    api_key: str | None = None
    model: str | None = None
    api_url: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None

    thinking: bool | None = None
    thinking_tokens: int | None = None

    async_client: Any = field(default=None, init=False)
    client: Any = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("QWEN_API_KEY")

        base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/"
        if self.api_key is not None:
            self.async_client = AsyncOpenAIClient(api_key=self.api_key, base_url=base_url)
            self.client = OpenAIClient(api_key=self.api_key, base_url=base_url)
        else:
            self.async_client = AsyncOpenAIClient(base_url=base_url)
            self.client = OpenAIClient(base_url=base_url)

    async def ahandle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ) -> Any | None:
        try:
            formatted_messages = [self._format_files_for_qwen(msg.copy()) for msg in messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": formatted_messages,
                "stream": False,
            }

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
                if hasattr(tool_executor, "parallel_calls"):
                    request_params["parallel_tool_calls"] = tool_executor.parallel_calls
                else:
                    request_params["parallel_tool_calls"] = True

            if response_format is not None:
                if hasattr(response_format, "model_json_schema"):
                    schema = response_format.model_json_schema()
                    request_params["response_format"] = {"type": "json_object", "schema": schema}
                else:
                    request_params["response_format"] = {"type": "json_object"}

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

            api_response: ChatCompletion = await self.async_client.chat.completions.create(**request_params)

            message = api_response.choices[0].message

            messages.append(
                {
                    "role": message.role,
                    "content": message.content,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None,
                }
            )

            if message.tool_calls:
                usage = {
                    "input_tokens": api_response.usage.prompt_tokens if api_response.usage else 0,
                    "output_tokens": api_response.usage.completion_tokens if api_response.usage else 0,
                    "total_tokens": api_response.usage.total_tokens if api_response.usage else 0,
                }
                self._accumulate_metrics(usage)

                tool_calls = [tool_call.model_dump() for tool_call in message.tool_calls]

                tool_outputs = await self._execute_tools_parallel_async(tool_executor, tool_calls)

                for tool_call in message.tool_calls:
                    if hasattr(tool_call, "function") and tool_call.function:
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

                usage_obj = api_response.usage if "api_response" in locals() else None
                usage = {
                    "input_tokens": usage_obj.prompt_tokens if usage_obj else 0,
                    "output_tokens": usage_obj.completion_tokens if usage_obj else 0,
                    "total_tokens": usage_obj.total_tokens if usage_obj else 0,
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
                metadata = {
                    "model": self.model,
                    "params": params,
                    "finish_reason": api_response.choices[0].finish_reason or "stop",
                    "input_tokens": cumulative_metrics.input_tokens,
                    "output_tokens": cumulative_metrics.output_tokens,
                }

                return self._create_llm_response(content, metadata, tool_calls, tool_outputs)

        except RateLimitError as e:
            raise Exception(f"Rate limit error from Qwen: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from Qwen: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from Qwen: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from Qwen: {e}") from e

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
            formatted_messages = [self._format_files_for_qwen(msg.copy()) for msg in messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": formatted_messages,
                "stream": True,
            }

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
                if hasattr(tool_executor, "parallel_calls"):
                    request_params["parallel_tool_calls"] = tool_executor.parallel_calls
                else:
                    request_params["parallel_tool_calls"] = True

            if response_format is not None:
                if hasattr(response_format, "model_json_schema"):
                    schema = response_format.model_json_schema()
                    request_params["response_format"] = {"type": "json_object", "schema": schema}
                else:
                    request_params["response_format"] = {"type": "json_object"}

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

            extra_body: dict[str, Any] = {}
            if self.thinking is not None:
                extra_body["enable_thinking"] = self.thinking
            if self.thinking_tokens is not None:
                extra_body["thinking_budget"] = self.thinking_tokens

            if extra_body:
                request_params["extra_body"] = extra_body

            request_params["stream_options"] = {"include_usage": True}

            stream = await self.async_client.chat.completions.create(**request_params)

            async for chunk in stream:
                async for event in self._process_streaming_event_async(chunk, tool_calls, tool_executor, messages):
                    yield event

        except RateLimitError as e:
            raise Exception(f"Rate limit error from Qwen: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from Qwen: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from Qwen: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from Qwen: {e}") from e

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
        has_content = bool(content and content.strip()) if content else False

        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_content = delta.reasoning_content
            yield ReasoningEvent(reasoning=reasoning_content)

        if has_content or choice.finish_reason:
            if has_content:
                yield ContentEvent(content=content or "")
            if choice.finish_reason:
                yield FinishReasonEvent(finish_reason=choice.finish_reason)
                if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                    yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

        if choice.finish_reason in ["stop", "tool_calls"]:
            if tool_calls and any(tc["function"]["name"] for tc in tool_calls):
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

                        yield ToolCallsEvent(tool_calls=[first_tool])

                        messages.append({"role": "assistant", "content": None, "tool_calls": [first_tool]})

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

                        if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                            yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

                        yield True
                        return
                else:
                    yield ToolCallsEvent(tool_calls=tool_calls)

                    messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

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

                if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                    yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

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
        try:
            formatted_messages = [self._format_files_for_qwen(msg.copy()) for msg in messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": formatted_messages,
                "stream": False,
            }

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
                if hasattr(tool_executor, "parallel_calls"):
                    request_params["parallel_tool_calls"] = tool_executor.parallel_calls
                else:
                    request_params["parallel_tool_calls"] = True

            if response_format is not None:
                if hasattr(response_format, "model_json_schema"):
                    schema = response_format.model_json_schema()
                    request_params["response_format"] = {"type": "json_object", "schema": schema}
                else:
                    request_params["response_format"] = {"type": "json_object"}

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

            api_response: ChatCompletion = self.client.chat.completions.create(**request_params)

            message = api_response.choices[0].message

            messages.append(
                {
                    "role": message.role,
                    "content": message.content,
                    "tool_calls": [tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None,
                }
            )

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

                if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                    if message.tool_calls:
                        first_tool_call = message.tool_calls[0]
                        if not hasattr(first_tool_call, "function") or first_tool_call.function is None:
                            return None
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

                usage_obj = api_response.usage if "api_response" in locals() else None
                usage = {
                    "input_tokens": usage_obj.prompt_tokens if usage_obj else 0,
                    "output_tokens": usage_obj.completion_tokens if usage_obj else 0,
                    "total_tokens": usage_obj.total_tokens if usage_obj else 0,
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
                metadata = {
                    "model": self.model,
                    "params": params,
                    "finish_reason": api_response.choices[0].finish_reason or "stop",
                    "input_tokens": cumulative_metrics.input_tokens,
                    "output_tokens": cumulative_metrics.output_tokens,
                }

                return self._create_llm_response(content, metadata, tool_calls, tool_outputs)

        except RateLimitError as e:
            raise Exception(f"Rate limit error from Qwen: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from Qwen: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from Qwen: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from Qwen: {e}") from e

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
            formatted_messages = [self._format_files_for_qwen(msg.copy()) for msg in messages]

            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": formatted_messages,
                "stream": True,
            }

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"
                if hasattr(tool_executor, "parallel_calls"):
                    request_params["parallel_tool_calls"] = tool_executor.parallel_calls
                else:
                    request_params["parallel_tool_calls"] = True

            if response_format is not None:
                if hasattr(response_format, "model_json_schema"):
                    schema = response_format.model_json_schema()
                    request_params["response_format"] = {"type": "json_object", "schema": schema}
                else:
                    request_params["response_format"] = {"type": "json_object"}

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

            extra_body: dict[str, Any] = {}
            if self.thinking is not None:
                extra_body["enable_thinking"] = self.thinking
            if self.thinking_tokens is not None:
                extra_body["thinking_budget"] = self.thinking_tokens

            if extra_body:
                request_params["extra_body"] = extra_body

            request_params["stream_options"] = {"include_usage": True}

            stream = self.client.chat.completions.create(**request_params)

            for chunk in stream:
                yield from self._process_streaming_event_sync(chunk, tool_calls, tool_executor, messages)

        except RateLimitError as e:
            raise Exception(f"Rate limit error from Qwen: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from Qwen: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from Qwen: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from Qwen: {e}") from e

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
        has_content = bool(content and content.strip()) if content else False

        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_content = delta.reasoning_content
            yield ReasoningEvent(reasoning=reasoning_content)

        if has_content or choice.finish_reason:
            if has_content:
                yield ContentEvent(content=content or "")
            if choice.finish_reason:
                yield FinishReasonEvent(finish_reason=choice.finish_reason)
                if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                    yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

        if choice.finish_reason in ["stop", "tool_calls"]:
            if tool_calls and any(tc["function"]["name"] for tc in tool_calls):
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

                        yield ToolCallsEvent(tool_calls=[first_tool])

                        messages.append({"role": "assistant", "content": None, "tool_calls": [first_tool]})

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

                        if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                            yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

                        yield True
                        return
                else:
                    yield ToolCallsEvent(tool_calls=tool_calls)

                    messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

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

                if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                    yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

                if hasattr(tool_executor, "_tool_outputs"):
                    tool_executor._tool_outputs.update(tool_outputs)
                if hasattr(tool_executor, "_tool_calls"):
                    tool_executor._tool_calls.extend(tool_calls)

                yield True
                return

            if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

            yield False

    def _format_files_for_qwen(self, message: dict[str, Any]) -> dict[str, Any]:
        from hypertic.utils.files import File, FileType

        if "_file_objects" not in message:
            return message

        file_objects: list[File] = message.get("_file_objects", [])
        if not file_objects:
            return message

        model_name = (self.model or "").lower()
        has_vision_support = "vl" in model_name

        content: list[dict[str, Any]] = []
        message_content = message.get("content")
        if message_content:
            if isinstance(message_content, str):
                content.append({"type": "text", "text": message_content})
            elif isinstance(message_content, list):
                content = message_content

        for file_obj in file_objects:
            if not has_vision_support and file_obj.file_type in (FileType.IMAGE, FileType.VIDEO):
                from hypertic.utils.log import get_logger

                logger = get_logger(__name__)
                logger.warning(
                    f"Model '{self.model}' does not support vision. "
                    f"Use a Qwen-VL model (e.g., qwen-vl-max, qwen-vl-plus) for image/video support. "
                    f"File: {file_obj.url or file_obj.filepath}"
                )
                continue

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
            elif file_obj.file_type == FileType.VIDEO:
                if file_obj.url:
                    content.append({"type": "video_url", "video_url": {"url": file_obj.url}})
                else:
                    base64_data = file_obj.to_base64()
                    if base64_data:
                        mime_type = file_obj.mime_type or "video/mp4"
                        content.append(
                            {
                                "type": "video_url",
                                "video_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                            }
                        )
            else:
                from hypertic.utils.log import get_logger

                logger = get_logger(__name__)
                logger.warning(
                    f"File type {file_obj.file_type} not supported in Qwen Chat API "
                    f"(only images and videos are supported in VL models): {file_obj.url or file_obj.filepath}"
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
