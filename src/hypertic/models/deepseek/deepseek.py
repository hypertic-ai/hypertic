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
class DeepSeek(Base):
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None

    async_client: Any = field(default=None, init=False)
    client: Any = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("DEEPSEEK_API_KEY")

        base_url = "https://api.deepseek.com/v1"
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
        for msg in messages:
            if "_file_objects" in msg and msg.get("_file_objects"):
                from hypertic.utils.log import get_logger

                logger = get_logger(__name__)
                file_objects = msg.get("_file_objects", [])
                for file_obj in file_objects:
                    logger.warning(
                        f"File type {file_obj.file_type} not supported in DeepSeek API "
                        f"(only text content is supported): {file_obj.url or file_obj.filepath}"
                    )

        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
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

        try:
            api_response: ChatCompletion = await self.async_client.chat.completions.create(**request_params)

            message = api_response.choices[0].message

            reasoning_content = None
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning_content = message.reasoning_content

            if message.tool_calls:
                usage: dict[str, Any] = {}
                if api_response.usage is not None:
                    usage = {
                        "input_tokens": api_response.usage.prompt_tokens,
                        "output_tokens": api_response.usage.completion_tokens,
                        "total_tokens": api_response.usage.total_tokens,
                    }
                self._accumulate_metrics(usage)

                tool_calls = [tc.model_dump() for tc in message.tool_calls]

                if (
                    hasattr(tool_executor, "parallel_calls")
                    and not tool_executor.parallel_calls
                    and not getattr(tool_executor, "_skip_handler_collection", False)
                ):
                    if tool_calls:
                        first_tool = tool_calls[0]

                        if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
                            messages[-1]["tool_calls"] = [first_tool]
                            messages[-1]["content"] = message.content
                        else:
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": message.content,
                                    "tool_calls": [first_tool],
                                }
                            )

                        tool_outputs_iter = await self._execute_tools_parallel_async(tool_executor, [first_tool])
                        function_name = first_tool["function"]["name"]
                        tool_result = tool_outputs_iter.get(function_name, f"Tool '{function_name}' execution failed")

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": first_tool["id"],
                                "content": tool_result,
                            }
                        )

                        if hasattr(tool_executor, "_tool_outputs"):
                            tool_executor._tool_outputs.update({function_name: tool_result})
                        if hasattr(tool_executor, "_tool_calls"):
                            tool_executor._tool_calls.append(first_tool)

                        return None
                    else:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": message.content,
                                "tool_calls": tool_calls,
                            }
                        )
                        return None
                else:
                    messages.append({"role": "assistant", "content": message.content, "tool_calls": tool_calls})

                    tool_outputs = await self._execute_tools_parallel_async(tool_executor, tool_calls)

                    for tool_call_dict in tool_calls:
                        function_name = tool_call_dict["function"]["name"]
                        tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_dict["id"],
                                "content": tool_result,
                            }
                        )

                if hasattr(tool_executor, "_tool_outputs"):
                    tool_executor._tool_outputs.update(tool_outputs)
                if hasattr(tool_executor, "_tool_calls"):
                    tool_executor._tool_calls.extend(tool_calls)

                return None
            else:
                message_dict = {
                    "role": message.role,
                    "content": message.content,
                    "tool_calls": None,
                }

                if reasoning_content:
                    message_dict["reasoning_content"] = reasoning_content

                messages.append(message_dict)

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

                if reasoning_content:
                    metadata["reasoning_content"] = reasoning_content

                return self._create_llm_response(content, metadata, tool_calls, tool_outputs)

        except RateLimitError as e:
            raise Exception(f"Rate limit error from DeepSeek: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from DeepSeek: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from DeepSeek: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from DeepSeek: {e}") from e

    async def ahandle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        for msg in messages:
            if "_file_objects" in msg and msg.get("_file_objects"):
                from hypertic.utils.log import get_logger

                logger = get_logger(__name__)
                file_objects = msg.get("_file_objects", [])
                for file_obj in file_objects:
                    logger.warning(
                        f"File type {file_obj.file_type} not supported in DeepSeek API "
                        f"(only text content is supported): {file_obj.url or file_obj.filepath}"
                    )

        tool_calls: list[dict[str, Any]] = []

        try:
            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
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

            request_params["stream_options"] = {"include_usage": True}

            stream = await self.async_client.chat.completions.create(**request_params)

            async for chunk in stream:
                async for event in self._process_streaming_event_async(chunk, tool_calls, tool_executor, messages):
                    yield event

        except RateLimitError as e:
            raise Exception(f"Rate limit error from DeepSeek: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from DeepSeek: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from DeepSeek: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from DeepSeek: {e}") from e

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

        reasoning_content = None
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_content = delta.reasoning_content
            yield ReasoningEvent(reasoning=reasoning_content)

        if has_content or choice.finish_reason:
            if has_content:
                yield ContentEvent(content=content or "")
            if choice.finish_reason:
                yield FinishReasonEvent(finish_reason=choice.finish_reason)

        if choice.finish_reason in ["stop", "tool_calls"]:
            if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

            if tool_calls and any(tc["function"]["name"] for tc in tool_calls):
                is_sequential = (
                    hasattr(tool_executor, "parallel_calls")
                    and not tool_executor.parallel_calls
                    and not getattr(tool_executor, "_skip_handler_collection", False)
                )

                messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

                tools_to_yield = [tool_calls[0]] if (is_sequential and len(tool_calls) > 1) else tool_calls
                yield ToolCallsEvent(tool_calls=tools_to_yield)

                if is_sequential:
                    all_tool_outputs = {}
                    if tool_calls:
                        first_tool = tool_calls[0]

                        messages[-1]["tool_calls"] = [first_tool]

                        tool_outputs_iter = await self._execute_tools_parallel_async(tool_executor, [first_tool])
                        function_name = first_tool["function"]["name"]
                        tool_result = tool_outputs_iter.get(function_name, f"Tool '{function_name}' execution failed")
                        all_tool_outputs[function_name] = tool_result

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": first_tool["id"],
                                "content": tool_result,
                            }
                        )

                        tool_calls = [first_tool]

                    tool_outputs = all_tool_outputs
                else:
                    tool_outputs = await self._execute_tools_parallel_async(tool_executor, tool_calls)

                    for tool_call in tool_calls:
                        if tool_call["function"]["name"]:  # Valid tool call
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

    # SYNC:
    def handle_non_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ) -> Any | None:
        for msg in messages:
            if "_file_objects" in msg and msg.get("_file_objects"):
                from hypertic.utils.log import get_logger

                logger = get_logger(__name__)
                file_objects = msg.get("_file_objects", [])
                for file_obj in file_objects:
                    logger.warning(
                        f"File type {file_obj.file_type} not supported in DeepSeek API "
                        f"(only text content is supported): {file_obj.url or file_obj.filepath}"
                    )

        request_params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
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

        try:
            api_response: ChatCompletion = self.client.chat.completions.create(**request_params)

            message = api_response.choices[0].message

            reasoning_content = None
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning_content = message.reasoning_content

            if message.tool_calls:
                usage: dict[str, Any] = {}
                if api_response.usage is not None:
                    usage = {
                        "input_tokens": api_response.usage.prompt_tokens,
                        "output_tokens": api_response.usage.completion_tokens,
                        "total_tokens": api_response.usage.total_tokens,
                    }
                self._accumulate_metrics(usage)

                tool_calls = [tc.model_dump() for tc in message.tool_calls]

                if (
                    hasattr(tool_executor, "parallel_calls")
                    and not tool_executor.parallel_calls
                    and not getattr(tool_executor, "_skip_handler_collection", False)
                ):
                    all_tool_calls: list[dict[str, Any]] = []
                    all_tool_outputs = {}
                    current_messages = messages.copy()

                    first_tool = tool_calls[0] if tool_calls else None

                    if first_tool:
                        current_messages.append(
                            {
                                "role": "assistant",
                                "content": message.content,
                                "tool_calls": [first_tool],
                            }
                        )
                    else:
                        current_messages.append(
                            {
                                "role": "assistant",
                                "content": message.content,
                                "tool_calls": tool_calls,
                            }
                        )

                    max_iterations = 10

                    for _iteration in range(max_iterations):
                        current_tool = None

                        if tool_calls and len(all_tool_calls) == 0:
                            current_tool = tool_calls[0]
                        else:
                            try:
                                temp_params = {
                                    "model": self.model,
                                    "messages": current_messages,
                                    "stream": False,
                                    "parallel_tool_calls": False,
                                }
                                if tools:
                                    temp_params["tools"] = tools
                                for k, v in request_params.items():
                                    if k not in [
                                        "model",
                                        "messages",
                                        "tools",
                                        "stream",
                                        "parallel_tool_calls",
                                    ]:
                                        temp_params[k] = v

                                temp_response: ChatCompletion = self.client.chat.completions.create(**temp_params)
                                temp_message = temp_response.choices[0].message

                                if temp_message.tool_calls and len(temp_message.tool_calls) > 0:
                                    next_tool_calls = [tc.model_dump() for tc in temp_message.tool_calls]
                                    current_tool = next_tool_calls[0]

                                    current_messages.append(
                                        {
                                            "role": "assistant",
                                            "content": temp_message.content,
                                            "tool_calls": [current_tool],
                                        }
                                    )
                                else:
                                    if temp_message.content:
                                        current_messages.append({"role": "assistant", "content": temp_message.content})
                                    break
                            except Exception:
                                break

                        if not current_tool:
                            break

                        tool_outputs_iter = self._execute_tools_parallel_sync(tool_executor, [current_tool])
                        function_name = current_tool["function"]["name"]
                        tool_result = tool_outputs_iter.get(function_name, f"Tool '{function_name}' execution failed")

                        all_tool_calls.append(current_tool)
                        all_tool_outputs[function_name] = tool_result

                        current_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": current_tool["id"],
                                "content": tool_result,
                            }
                        )

                        tool_calls = []

                    messages = current_messages
                    tool_calls = all_tool_calls
                    tool_outputs = all_tool_outputs

                    if messages and messages[-1].get("role") == "assistant" and messages[-1].get("content") and not messages[-1].get("tool_calls"):
                        final_content = messages[-1]["content"]
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

                        metadata = {
                            "model": self.model,
                            "params": params,
                            "finish_reason": "stop",
                            "input_tokens": cumulative_metrics.input_tokens,
                            "output_tokens": cumulative_metrics.output_tokens,
                        }

                        return self._create_llm_response(final_content, metadata, all_tool_calls, all_tool_outputs)

                else:
                    messages.append({"role": "assistant", "content": message.content, "tool_calls": tool_calls})

                    tool_outputs = self._execute_tools_parallel_sync(tool_executor, tool_calls)

                    for tool_call_dict in tool_calls:
                        function_name = tool_call_dict["function"]["name"]
                        tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_dict["id"],
                                "content": tool_result,
                            }
                        )

                if hasattr(tool_executor, "_tool_outputs"):
                    tool_executor._tool_outputs.update(tool_outputs)
                if hasattr(tool_executor, "_tool_calls"):
                    tool_executor._tool_calls.extend([tc.model_dump() for tc in message.tool_calls])

                return None
            else:
                message_dict = {
                    "role": message.role,
                    "content": message.content,
                    "tool_calls": None,
                }

                if reasoning_content:
                    message_dict["reasoning_content"] = reasoning_content

                messages.append(message_dict)

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
                sync_metadata_params: dict[str, Any] = {}
                if self.temperature is not None:
                    sync_metadata_params["temperature"] = self.temperature
                if self.top_p is not None:
                    sync_metadata_params["top_p"] = self.top_p
                if self.presence_penalty is not None:
                    sync_metadata_params["presence_penalty"] = self.presence_penalty
                if self.frequency_penalty is not None:
                    sync_metadata_params["frequency_penalty"] = self.frequency_penalty
                if self.max_tokens is not None:
                    sync_metadata_params["max_tokens"] = self.max_tokens

                metadata = {
                    "model": self.model,
                    "params": sync_metadata_params,
                    "finish_reason": api_response.choices[0].finish_reason or "stop",
                    "input_tokens": cumulative_metrics.input_tokens,
                    "output_tokens": cumulative_metrics.output_tokens,
                }

                if reasoning_content:
                    metadata["reasoning_content"] = reasoning_content

                return self._create_llm_response(content, metadata, tool_calls, tool_outputs)

        except RateLimitError as e:
            raise Exception(f"Rate limit error from DeepSeek: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from DeepSeek: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from DeepSeek: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from DeepSeek: {e}") from e

    def handle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        for msg in messages:
            if "_file_objects" in msg and msg.get("_file_objects"):
                from hypertic.utils.log import get_logger

                logger = get_logger(__name__)
                file_objects = msg.get("_file_objects", [])
                for file_obj in file_objects:
                    logger.warning(
                        f"File type {file_obj.file_type} not supported in DeepSeek API "
                        f"(only text content is supported): {file_obj.url or file_obj.filepath}"
                    )

        tool_calls: list[dict[str, Any]] = []

        try:
            request_params: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
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

            request_params["stream_options"] = {"include_usage": True}

            stream = self.client.chat.completions.create(**request_params)

            for chunk in stream:
                yield from self._process_streaming_event_sync(chunk, tool_calls, tool_executor, messages)

        except RateLimitError as e:
            raise Exception(f"Rate limit error from DeepSeek: {e}") from e
        except APIConnectionError as e:
            raise Exception(f"Connection error from DeepSeek: {e}") from e
        except APIStatusError as e:
            error_message = e.response.json().get("error", {})
            error_message = error_message.get("message", "Unknown model error") if isinstance(error_message, dict) else error_message
            raise Exception(f"Status error from DeepSeek: {error_message}") from e
        except Exception as e:
            raise Exception(f"Error from DeepSeek: {e}") from e

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

        reasoning_content = None
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_content = delta.reasoning_content
            yield ReasoningEvent(reasoning=reasoning_content)

        if has_content or choice.finish_reason:
            if has_content:
                yield ContentEvent(content=content or "")
            if choice.finish_reason:
                yield FinishReasonEvent(finish_reason=choice.finish_reason)

        if choice.finish_reason in ["stop", "tool_calls"]:
            if hasattr(tool_executor, "_streaming_usage") and tool_executor._streaming_usage:
                yield ResponseCompletedEvent(usage=tool_executor._streaming_usage)

            if tool_calls and any(tc["function"]["name"] for tc in tool_calls):
                is_sequential = (
                    hasattr(tool_executor, "parallel_calls")
                    and not tool_executor.parallel_calls
                    and not getattr(tool_executor, "_skip_handler_collection", False)
                )

                messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

                tools_to_yield = [tool_calls[0]] if (is_sequential and len(tool_calls) > 1) else tool_calls
                yield ToolCallsEvent(tool_calls=tools_to_yield)

                if is_sequential:
                    all_tool_outputs = {}
                    if tool_calls:
                        first_tool = tool_calls[0]

                        messages[-1]["tool_calls"] = [first_tool]

                        tool_outputs_iter = self._execute_tools_parallel_sync(tool_executor, [first_tool])
                        function_name = first_tool["function"]["name"]
                        tool_result = tool_outputs_iter.get(function_name, f"Tool '{function_name}' execution failed")
                        all_tool_outputs[function_name] = tool_result

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": first_tool["id"],
                                "content": tool_result,
                            }
                        )

                        tool_calls = [first_tool]

                    tool_outputs = all_tool_outputs
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

            yield False
