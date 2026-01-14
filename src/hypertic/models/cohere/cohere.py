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
    from cohere import AsyncClientV2 as CohereAsyncClient, ClientV2 as CohereClient, V2ChatResponse
except ImportError as err:
    raise ImportError("`cohere` not installed. Please install using `pip install cohere`") from err


@dataclass
class Cohere(Base):
    api_key: str | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    max_tokens: int | None = None

    thinking: bool | dict[str, Any] | None = None
    thinking_tokens: int | None = None

    client: Any = field(default=None, init=False)
    async_client: Any = field(default=None, init=False)

    def __post_init__(self):
        self.api_key = self.api_key or getenv("COHERE_API_KEY")

        if self.api_key is not None:
            self.async_client = CohereAsyncClient(api_key=self.api_key)
            self.client = CohereClient(api_key=self.api_key)
        else:
            self.async_client = CohereAsyncClient()
            self.client = CohereClient()

    def _convert_tools_to_cohere(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cohere_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                function = tool["function"]
                cohere_tool = {
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "description": function["description"],
                        "parameters": function["parameters"],
                    },
                }
                cohere_tools.append(cohere_tool)

        return cohere_tools

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
            formatted_msg = self._format_files_for_cohere(msg.copy())
            processed_messages.append(formatted_msg)

        data: dict[str, Any] = {"model": self.model, "messages": processed_messages}

        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.top_p is not None:
            data["p"] = self.top_p
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.presence_penalty is not None:
            data["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            data["frequency_penalty"] = self.frequency_penalty

        if self.thinking is not None or self.thinking_tokens is not None:
            thinking_config: dict[str, Any] = {}
            if self.thinking is not None:
                if isinstance(self.thinking, bool):
                    thinking_config["type"] = "enabled" if self.thinking else "disabled"
                else:
                    thinking_config = self.thinking if isinstance(self.thinking, dict) else {}
            if self.thinking_tokens is not None:
                thinking_config["token_budget"] = self.thinking_tokens
            data["thinking"] = thinking_config

        if tools:
            data["tools"] = self._convert_tools_to_cohere(tools)

        if response_format is not None:
            from pydantic import BaseModel

            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                pydantic_model = response_format
                if hasattr(pydantic_model, "model_json_schema"):
                    json_schema = pydantic_model.model_json_schema()
                    data["response_format"] = {"type": "json_object", "schema": json_schema}
                else:
                    data["response_format"] = {"type": "json_object"}
            else:
                data["response_format"] = {"type": "json_object"}

        try:
            api_response: V2ChatResponse = await self.async_client.v2.chat(**data)
        except Exception as e:
            raise Exception(f"Error from Cohere SDK: {e}") from e

        message = api_response.message
        tool_calls = message.tool_calls

        if tool_calls:
            usage: dict[str, Any] = {}
            if api_response.usage is not None:
                usage = (
                    api_response.usage.model_dump()
                    if hasattr(api_response.usage, "model_dump")
                    else api_response.usage
                    if isinstance(api_response.usage, dict)
                    else {}
                )
            self._accumulate_metrics(usage)

            is_sequential = hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls

            tool_calls_for_message = [tool_calls[0]] if (is_sequential and len(tool_calls) > 1) else tool_calls

            messages.append({"role": "assistant", "tool_calls": tool_calls_for_message})

            if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                if tool_calls:
                    first_tool_call = tool_calls[0]
                    function = first_tool_call.function
                    if function is None:
                        return None
                    function_name = function.name
                    if function_name is None:
                        return None
                    tool_call_dict = first_tool_call if isinstance(first_tool_call, dict) else first_tool_call.model_dump()
                    results = await self._execute_tools_parallel_async(tool_executor, [tool_call_dict])
                    tool_result = results.get(function_name, "")
                    tool_outputs = {function_name: tool_result}

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": first_tool_call.id,
                            "content": [{"type": "document", "document": {"data": str(tool_result)}}],
                        }
                    )
                else:
                    tool_outputs = {}
            else:
                tool_calls_dict = []
                for tool_call in tool_calls:
                    if tool_call.function is not None:
                        tool_calls_dict.append(
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        )

                tool_outputs = await self._execute_tools_parallel_async(tool_executor, tool_calls_dict)

                for tool_call in tool_calls:
                    if tool_call.function is None:
                        continue
                    function_name = tool_call.function.name
                    if function_name is None:
                        continue
                    tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": [{"type": "document", "document": {"data": str(tool_result)}}],
                        }
                    )

            if hasattr(tool_executor, "_tool_outputs"):
                tool_executor._tool_outputs.update(tool_outputs)
            if hasattr(tool_executor, "_tool_calls"):
                tool_calls_dict = []
                for tool_call in tool_calls:
                    if tool_call.function is not None:
                        tool_calls_dict.append(
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        )
                tool_executor._tool_calls.extend(tool_calls_dict)

            return None
        else:
            final_usage: dict[str, Any] = {}
            if api_response.usage is not None:
                final_usage = (
                    api_response.usage.model_dump()
                    if hasattr(api_response.usage, "model_dump")
                    else api_response.usage
                    if isinstance(api_response.usage, dict)
                    else {}
                )
            usage = final_usage

            self._accumulate_metrics(usage)

            cumulative_metrics = self._get_cumulative_metrics()

            content_blocks = message.content
            response_text = ""
            reasoning_content = ""
            tool_calls_list: list[dict[str, Any]] = []
            if content_blocks is not None:
                for block in content_blocks:
                    if block.type == "text":
                        if hasattr(block, "text"):
                            response_text += block.text
                    elif block.type == "thinking":
                        if hasattr(block, "thinking"):
                            reasoning_content += block.thinking
                    elif block.type == "tool_call":
                        if hasattr(block, "id") and hasattr(block, "name") and hasattr(block, "input"):
                            tool_calls_list.append(
                                {
                                    "id": block.id,
                                    "type": "function",
                                    "function": {
                                        "name": block.name,
                                        "arguments": json.dumps(block.input),
                                    },
                                }
                            )

            finish_reason = "stop"
            if api_response.finish_reason:
                finish_reason = api_response.finish_reason

            stored_tool_calls = []
            if hasattr(tool_executor, "_tool_calls"):
                stored_tool_calls = tool_executor._tool_calls.copy()

            tool_outputs = {}
            if hasattr(tool_executor, "_tool_outputs"):
                tool_outputs = tool_executor._tool_outputs.copy()

            params: dict[str, Any] = {}
            if self.temperature is not None:
                params["temperature"] = self.temperature
            if self.top_p is not None:
                params["top_p"] = self.top_p
            if self.max_tokens is not None:
                params["max_tokens"] = self.max_tokens
            if self.presence_penalty is not None:
                params["presence_penalty"] = self.presence_penalty
            if self.frequency_penalty is not None:
                params["frequency_penalty"] = self.frequency_penalty
            if self.thinking is not None:
                params["thinking"] = self.thinking
            if self.thinking_tokens is not None:
                params["thinking_tokens"] = self.thinking_tokens

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

    async def ahandle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        """Handler for streaming responses."""
        tool_calls: list[dict[str, Any]] = []
        current_text = ""
        content_buffer = ""

        processed_messages = []
        for msg in messages:
            formatted_msg = self._format_files_for_cohere(msg.copy())
            processed_messages.append(formatted_msg)

        data: dict[str, Any] = {"model": self.model, "messages": processed_messages}

        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.top_p is not None:
            data["p"] = self.top_p
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.presence_penalty is not None:
            data["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            data["frequency_penalty"] = self.frequency_penalty

        if self.thinking is not None or self.thinking_tokens is not None:
            thinking_config: dict[str, Any] = {}
            if self.thinking is not None:
                if isinstance(self.thinking, bool):
                    thinking_config["type"] = "enabled" if self.thinking else "disabled"
                else:
                    thinking_config = self.thinking if isinstance(self.thinking, dict) else {}
            if self.thinking_tokens is not None:
                thinking_config["token_budget"] = self.thinking_tokens
            data["thinking"] = thinking_config

        if tools:
            data["tools"] = self._convert_tools_to_cohere(tools)

        if response_format is not None:
            from pydantic import BaseModel

            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                pydantic_model = response_format
                if hasattr(pydantic_model, "model_json_schema"):
                    json_schema = pydantic_model.model_json_schema()
                    data["response_format"] = {"type": "json_object", "schema": json_schema}
                else:
                    data["response_format"] = {"type": "json_object"}
            else:
                data["response_format"] = {"type": "json_object"}

        try:
            stream = self.async_client.v2.chat_stream(**data)
        except Exception as e:
            raise Exception(f"Error from Cohere SDK (streaming): {e}") from e

        try:
            async for event in stream:
                event_type = event.type

                if event_type == "content-delta":
                    delta = event.delta.message.content

                    if hasattr(delta, "thinking") and delta.thinking:
                        reasoning_chunk = delta.thinking
                        yield ReasoningEvent(reasoning=reasoning_chunk)

                    if hasattr(delta, "text") and delta.text:
                        text_chunk = delta.text
                        current_text += text_chunk

                        if response_format is not None:
                            content_buffer += text_chunk

                        yield ContentEvent(content=text_chunk)

                elif event_type == "tool-call-start":
                    index = event.index
                    delta = event.delta.message.tool_calls

                    while len(tool_calls) <= index:
                        tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

                    if hasattr(delta, "id"):
                        tool_calls[index]["id"] = delta.id
                    if hasattr(delta, "type"):
                        tool_calls[index]["type"] = delta.type
                    if hasattr(delta, "function") and hasattr(delta.function, "name"):
                        tool_calls[index]["function"]["name"] = delta.function.name

                elif event_type == "tool-call-delta":
                    index = event.index
                    delta = event.delta.message.tool_calls

                    if index < len(tool_calls) and hasattr(delta, "function") and hasattr(delta.function, "arguments"):
                        tool_calls[index]["function"]["arguments"] += delta.function.arguments

                elif event_type == "tool-call-end":
                    pass

                elif event_type == "message-end":
                    usage = None
                    try:
                        if hasattr(event, "delta") and event.delta is not None:
                            if hasattr(event.delta, "usage") and event.delta.usage is not None:
                                usage_obj = event.delta.usage
                                if hasattr(usage_obj, "tokens") and usage_obj.tokens is not None:
                                    tokens = usage_obj.tokens
                                    usage = {
                                        "input_tokens": getattr(tokens, "input_tokens", 0),
                                        "output_tokens": getattr(tokens, "output_tokens", 0),
                                        "total_tokens": getattr(tokens, "input_tokens", 0) + getattr(tokens, "output_tokens", 0),
                                    }
                                    if hasattr(usage_obj, "billed_units") and usage_obj.billed_units is not None:
                                        billed = usage_obj.billed_units
                                        usage["billed_input_tokens"] = getattr(billed, "input_tokens", 0)
                                        usage["billed_output_tokens"] = getattr(billed, "output_tokens", 0)
                                elif hasattr(usage_obj, "model_dump"):
                                    usage = usage_obj.model_dump()
                                else:
                                    usage = usage_obj if isinstance(usage_obj, dict) else {}
                            elif hasattr(event.delta, "message") and event.delta.message is not None and hasattr(event.delta.message, "usage"):
                                usage_obj = event.delta.message.usage
                                usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else usage_obj
                        elif hasattr(event, "message") and event.message is not None and hasattr(event.message, "usage"):
                            usage_obj = event.message.usage
                            usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else usage_obj
                        elif hasattr(event, "usage") and event.usage is not None:
                            usage = event.usage.model_dump() if hasattr(event.usage, "model_dump") else event.usage
                        elif hasattr(event, "response") and event.response is not None and hasattr(event.response, "usage"):
                            usage = event.response.usage.model_dump() if hasattr(event.response.usage, "model_dump") else event.response.usage
                    except (AttributeError, Exception):
                        pass

                    if tool_calls and any(tc.get("function", {}).get("name") for tc in tool_calls):
                        for tool_call in tool_calls:
                            if tool_call.get("function", {}).get("name"):
                                args_str = tool_call["function"]["arguments"]
                                try:
                                    parsed_args = json.loads(args_str)
                                    tool_call["function"]["arguments"] = json.dumps(parsed_args)
                                except Exception:
                                    tool_call["function"]["arguments"] = "{}"

                        is_sequential = hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls

                        messages.append({"role": "assistant", "tool_calls": tool_calls})

                        tools_to_yield = [tool_calls[0]] if (is_sequential and len(tool_calls) > 1) else tool_calls
                        yield ToolCallsEvent(tool_calls=tools_to_yield)

                        if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                            if tool_calls and tool_calls[0].get("function", {}).get("name"):
                                first_tool_call = tool_calls[0]
                                function_name = first_tool_call["function"]["name"]

                                messages[-1]["tool_calls"] = [first_tool_call]

                                tool_call_dict = first_tool_call if isinstance(first_tool_call, dict) else first_tool_call.model_dump()
                                tool_results: dict[str, str] = await self._execute_tools_parallel_async(tool_executor, [tool_call_dict])
                                tool_result = tool_results.get(function_name, "")
                                tool_outputs = {function_name: tool_result}

                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": first_tool_call["id"],
                                        "content": [{"type": "document", "document": {"data": str(tool_result)}}],
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
                                    if usage:
                                        yield ResponseCompletedEvent(usage=usage)

                                yield True
                                return
                            else:
                                tool_outputs = {}
                        else:
                            tool_outputs = await self._execute_tools_parallel_async(tool_executor, tool_calls)

                            for tool_call in tool_calls:
                                if tool_call.get("function", {}).get("name"):
                                    function_name = tool_call["function"]["name"]
                                    tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                                    messages.append(
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_call["id"],
                                            "content": [
                                                {
                                                    "type": "document",
                                                    "document": {"data": str(tool_result)},
                                                }
                                            ],
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
                            if usage:
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
                        if usage:
                            yield ResponseCompletedEvent(usage=usage)

                    yield FinishReasonEvent(finish_reason="stop")
                    yield False
                    break
        finally:
            if hasattr(stream, "aclose"):
                try:
                    import asyncio

                    task = asyncio.current_task()
                    if task is not None and not task.cancelled():
                        await asyncio.shield(stream.aclose())
                except (RuntimeError, GeneratorExit):
                    pass
                except Exception:
                    pass

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
            formatted_msg = self._format_files_for_cohere(msg.copy())
            processed_messages.append(formatted_msg)

        data: dict[str, Any] = {"model": self.model, "messages": processed_messages}

        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.top_p is not None:
            data["p"] = self.top_p
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.presence_penalty is not None:
            data["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            data["frequency_penalty"] = self.frequency_penalty

        if self.thinking is not None or self.thinking_tokens is not None:
            thinking_config: dict[str, Any] = {}
            if self.thinking is not None:
                if isinstance(self.thinking, bool):
                    thinking_config["type"] = "enabled" if self.thinking else "disabled"
                else:
                    thinking_config = self.thinking if isinstance(self.thinking, dict) else {}
            if self.thinking_tokens is not None:
                thinking_config["token_budget"] = self.thinking_tokens
            data["thinking"] = thinking_config

        if tools:
            data["tools"] = self._convert_tools_to_cohere(tools)

        if response_format is not None:
            from pydantic import BaseModel

            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                pydantic_model = response_format
                if hasattr(pydantic_model, "model_json_schema"):
                    json_schema = pydantic_model.model_json_schema()
                    data["response_format"] = {"type": "json_object", "schema": json_schema}
                else:
                    data["response_format"] = {"type": "json_object"}
            else:
                data["response_format"] = {"type": "json_object"}

        try:
            api_response: V2ChatResponse = self.client.v2.chat(**data)
        except Exception as e:
            raise Exception(f"Error from Cohere SDK: {e}") from e

        message = api_response.message
        tool_calls = message.tool_calls

        if tool_calls:
            usage: dict[str, Any] = {}
            if api_response.usage is not None:
                usage = (
                    api_response.usage.model_dump()
                    if hasattr(api_response.usage, "model_dump")
                    else api_response.usage
                    if isinstance(api_response.usage, dict)
                    else {}
                )
            self._accumulate_metrics(usage)

            is_sequential = hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls

            tool_calls_for_message = [tool_calls[0]] if (is_sequential and len(tool_calls) > 1) else tool_calls

            messages.append({"role": "assistant", "tool_calls": tool_calls_for_message})

            if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                if tool_calls:
                    first_tool_call = tool_calls[0]
                    function = first_tool_call.function
                    if function is None:
                        return None
                    function_name = function.name
                    if function_name is None:
                        return None
                    tool_call_dict = first_tool_call if isinstance(first_tool_call, dict) else first_tool_call.model_dump()
                    results = self._execute_tools_parallel_sync(tool_executor, [tool_call_dict])
                    tool_result = results.get(function_name, "")
                    tool_outputs = {function_name: tool_result}

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": first_tool_call.id,
                            "content": [{"type": "document", "document": {"data": str(tool_result)}}],
                        }
                    )
                else:
                    tool_outputs = {}
            else:
                tool_calls_dict = []
                for tool_call in tool_calls:
                    if tool_call.function is not None:
                        tool_calls_dict.append(
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        )

                tool_outputs = self._execute_tools_parallel_sync(tool_executor, tool_calls_dict)

                for tool_call in tool_calls:
                    if tool_call.function is None:
                        continue
                    function_name = tool_call.function.name
                    if function_name is None:
                        continue
                    tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": [{"type": "document", "document": {"data": str(tool_result)}}],
                        }
                    )

            if hasattr(tool_executor, "_tool_outputs"):
                tool_executor._tool_outputs.update(tool_outputs)
            if hasattr(tool_executor, "_tool_calls"):
                tool_calls_dict = []
                for tool_call in tool_calls:
                    if tool_call.function is not None:
                        tool_calls_dict.append(
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                        )
                tool_executor._tool_calls.extend(tool_calls_dict)

            return None
        else:
            final_usage: dict[str, Any] = {}
            if api_response.usage is not None:
                final_usage = (
                    api_response.usage.model_dump()
                    if hasattr(api_response.usage, "model_dump")
                    else api_response.usage
                    if isinstance(api_response.usage, dict)
                    else {}
                )
            usage = final_usage

            self._accumulate_metrics(usage)

            cumulative_metrics = self._get_cumulative_metrics()

            content_blocks = message.content
            response_text = ""
            reasoning_content = ""
            tool_calls_list: list[dict[str, Any]] = []
            if content_blocks is not None:
                for block in content_blocks:
                    if block.type == "text":
                        if hasattr(block, "text"):
                            response_text += block.text
                    elif block.type == "thinking":
                        if hasattr(block, "thinking"):
                            reasoning_content += block.thinking
                    elif block.type == "tool_call":
                        if hasattr(block, "id") and hasattr(block, "name") and hasattr(block, "input"):
                            tool_calls_list.append(
                                {
                                    "id": block.id,
                                    "type": "function",
                                    "function": {
                                        "name": block.name,
                                        "arguments": json.dumps(block.input),
                                    },
                                }
                            )

            finish_reason = "stop"
            if api_response.finish_reason:
                finish_reason = api_response.finish_reason

            stored_tool_calls = []
            if hasattr(tool_executor, "_tool_calls"):
                stored_tool_calls = tool_executor._tool_calls.copy()

            tool_outputs = {}
            if hasattr(tool_executor, "_tool_outputs"):
                tool_outputs = tool_executor._tool_outputs.copy()

            params: dict[str, Any] = {}
            if self.temperature is not None:
                params["temperature"] = self.temperature
            if self.top_p is not None:
                params["top_p"] = self.top_p
            if self.max_tokens is not None:
                params["max_tokens"] = self.max_tokens
            if self.presence_penalty is not None:
                params["presence_penalty"] = self.presence_penalty
            if self.frequency_penalty is not None:
                params["frequency_penalty"] = self.frequency_penalty
            if self.thinking is not None:
                params["thinking"] = self.thinking
            if self.thinking_tokens is not None:
                params["thinking_tokens"] = self.thinking_tokens

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

    def handle_streaming(
        self,
        model: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: Any,
        response_format: Any | None = None,
    ):
        tool_calls: list[dict[str, Any]] = []
        current_text = ""
        content_buffer = ""

        processed_messages = []
        for msg in messages:
            formatted_msg = self._format_files_for_cohere(msg.copy())
            processed_messages.append(formatted_msg)

        data: dict[str, Any] = {"model": self.model, "messages": processed_messages}

        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.top_p is not None:
            data["p"] = self.top_p
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.presence_penalty is not None:
            data["presence_penalty"] = self.presence_penalty
        if self.frequency_penalty is not None:
            data["frequency_penalty"] = self.frequency_penalty

        if self.thinking is not None or self.thinking_tokens is not None:
            thinking_config: dict[str, Any] = {}
            if self.thinking is not None:
                if isinstance(self.thinking, bool):
                    thinking_config["type"] = "enabled" if self.thinking else "disabled"
                else:
                    thinking_config = self.thinking if isinstance(self.thinking, dict) else {}
            if self.thinking_tokens is not None:
                thinking_config["token_budget"] = self.thinking_tokens
            data["thinking"] = thinking_config

        if tools:
            data["tools"] = self._convert_tools_to_cohere(tools)

        if response_format is not None:
            from pydantic import BaseModel

            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                pydantic_model = response_format
                if hasattr(pydantic_model, "model_json_schema"):
                    json_schema = pydantic_model.model_json_schema()
                    data["response_format"] = {"type": "json_object", "schema": json_schema}
                else:
                    data["response_format"] = {"type": "json_object"}
            else:
                data["response_format"] = {"type": "json_object"}

        try:
            stream = self.client.v2.chat_stream(**data)
        except Exception as e:
            raise Exception(f"Error from Cohere SDK (streaming): {e}") from e

        for event in stream:
            event_type = event.type

            if event_type == "content-delta":
                delta = event.delta.message.content

                if hasattr(delta, "thinking") and delta.thinking:
                    reasoning_chunk = delta.thinking
                    yield ReasoningEvent(reasoning=reasoning_chunk)

                if hasattr(delta, "text") and delta.text:
                    text_chunk = delta.text
                    current_text += text_chunk

                    if response_format is not None:
                        content_buffer += text_chunk

                    yield ContentEvent(content=text_chunk)

            elif event_type == "tool-call-start":
                index = event.index
                delta = event.delta.message.tool_calls

                while len(tool_calls) <= index:
                    tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})

                if hasattr(delta, "id"):
                    tool_calls[index]["id"] = delta.id
                if hasattr(delta, "type"):
                    tool_calls[index]["type"] = delta.type
                if hasattr(delta, "function") and hasattr(delta.function, "name"):
                    tool_calls[index]["function"]["name"] = delta.function.name

            elif event_type == "tool-call-delta":
                index = event.index
                delta = event.delta.message.tool_calls

                if index < len(tool_calls) and hasattr(delta, "function") and hasattr(delta.function, "arguments"):
                    tool_calls[index]["function"]["arguments"] += delta.function.arguments

            elif event_type == "tool-call-end":
                pass

            elif event_type == "message-end":
                usage = None
                try:
                    if hasattr(event, "delta") and event.delta is not None:
                        if hasattr(event.delta, "usage") and event.delta.usage is not None:
                            usage_obj = event.delta.usage
                            if hasattr(usage_obj, "tokens") and usage_obj.tokens is not None:
                                tokens = usage_obj.tokens
                                usage = {
                                    "input_tokens": getattr(tokens, "input_tokens", 0),
                                    "output_tokens": getattr(tokens, "output_tokens", 0),
                                    "total_tokens": getattr(tokens, "input_tokens", 0) + getattr(tokens, "output_tokens", 0),
                                }
                                if hasattr(usage_obj, "billed_units") and usage_obj.billed_units is not None:
                                    billed = usage_obj.billed_units
                                    usage["billed_input_tokens"] = getattr(billed, "input_tokens", 0)
                                    usage["billed_output_tokens"] = getattr(billed, "output_tokens", 0)
                            elif hasattr(usage_obj, "model_dump"):
                                usage = usage_obj.model_dump()
                            else:
                                usage = usage_obj if isinstance(usage_obj, dict) else {}
                        elif hasattr(event.delta, "message") and event.delta.message is not None and hasattr(event.delta.message, "usage"):
                            usage_obj = event.delta.message.usage
                            usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else usage_obj
                    elif hasattr(event, "message") and event.message is not None and hasattr(event.message, "usage"):
                        usage_obj = event.message.usage
                        usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else usage_obj
                    elif hasattr(event, "usage") and event.usage is not None:
                        usage = event.usage.model_dump() if hasattr(event.usage, "model_dump") else event.usage
                    elif hasattr(event, "response") and event.response is not None and hasattr(event.response, "usage"):
                        usage = event.response.usage.model_dump() if hasattr(event.response.usage, "model_dump") else event.response.usage
                except (AttributeError, Exception):
                    pass

                if tool_calls and any(tc.get("function", {}).get("name") for tc in tool_calls):
                    for tool_call in tool_calls:
                        if tool_call.get("function", {}).get("name"):
                            args_str = tool_call["function"]["arguments"]
                            try:
                                parsed_args = json.loads(args_str)
                                tool_call["function"]["arguments"] = json.dumps(parsed_args)
                            except Exception:
                                tool_call["function"]["arguments"] = "{}"

                    is_sequential = hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls

                    messages.append({"role": "assistant", "tool_calls": tool_calls})

                    tools_to_yield = [tool_calls[0]] if (is_sequential and len(tool_calls) > 1) else tool_calls
                    yield ToolCallsEvent(tool_calls=tools_to_yield)

                    if hasattr(tool_executor, "parallel_calls") and not tool_executor.parallel_calls:
                        if tool_calls and tool_calls[0].get("function", {}).get("name"):
                            first_tool_call = tool_calls[0]
                            function_name = first_tool_call["function"]["name"]

                            messages[-1]["tool_calls"] = [first_tool_call]

                            tool_call_dict = first_tool_call if isinstance(first_tool_call, dict) else first_tool_call.model_dump()
                            results = self._execute_tools_parallel_sync(tool_executor, [tool_call_dict])
                            tool_result = results[function_name]
                            tool_outputs = {function_name: tool_result}

                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": first_tool_call["id"],
                                    "content": [{"type": "document", "document": {"data": str(tool_result)}}],
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
                                if usage:
                                    yield ResponseCompletedEvent(usage=usage)

                            yield True
                            return
                        else:
                            tool_outputs = {}
                    else:
                        tool_outputs = self._execute_tools_parallel_sync(tool_executor, tool_calls)

                        for tool_call in tool_calls:
                            if tool_call.get("function", {}).get("name"):
                                function_name = tool_call["function"]["name"]
                                tool_result = tool_outputs.get(function_name, f"Tool '{function_name}' execution failed")

                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call["id"],
                                        "content": [
                                            {
                                                "type": "document",
                                                "document": {"data": str(tool_result)},
                                            }
                                        ],
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
                        if usage:
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
                        if usage:
                            yield ResponseCompletedEvent(usage=usage)

                    yield FinishReasonEvent(finish_reason="stop")
                    yield False
                    break

    def _format_files_for_cohere(self, message: dict[str, Any]) -> dict[str, Any]:
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
                    f"File type {file_obj.file_type} not supported in Cohere API message content "
                    f"(only images are supported in messages; documents require separate 'documents' parameter): {file_obj.url or file_obj.filepath}"
                )

        message["content"] = content
        message.pop("_file_objects", None)
        message.pop("files", None)
        return message
