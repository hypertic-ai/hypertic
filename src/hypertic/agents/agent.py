import asyncio
import dataclasses
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, Optional, cast, get_origin, get_type_hints

from pydantic import BaseModel, create_model

from hypertic.models.base import LLMResponse
from hypertic.models.events import (
    MetadataEvent,
    StreamEvent,
)
from hypertic.tools.base import BaseToolkit, _ToolManager
from hypertic.tools.mcp.client import ExecutableTool
from hypertic.utils.exceptions import (
    ConfigurationError,
    GuardrailViolationError,
    MaxStepsError,
    RetrieverError,
    SchemaConversionError,
    ToolExecutionError,
    ToolNotFoundError,
)
from hypertic.utils.files import FileProcessor
from hypertic.utils.log import get_logger

logger = get_logger(__name__)


def _convert_to_pydantic(output_type: type[Any]) -> type[BaseModel]:
    if issubclass(output_type, BaseModel):
        return output_type

    if dataclasses.is_dataclass(output_type):
        hints = get_type_hints(output_type)
        field_definitions = {}
        for field in dataclasses.fields(output_type):
            field_type = hints.get(field.name, field.type)
            if field.default is not dataclasses.MISSING:
                field_definitions[field.name] = (field_type, field.default)
            elif field.default_factory is not dataclasses.MISSING:
                field_definitions[field.name] = (field_type, field.default_factory)
            else:
                field_definitions[field.name] = (field_type, ...)

        model_dataclass_base: Any = create_model(output_type.__name__, **cast(Any, field_definitions))
        model = cast(type[BaseModel], model_dataclass_base)

        def _str_with_class_name(self: BaseModel) -> str:
            return repr(self)

        model.__str__ = _str_with_class_name  # type: ignore
        model._hypertic_str_overridden = True  # type: ignore
        return model

    is_typed_dict = (
        hasattr(output_type, "__annotations__")
        and not dataclasses.is_dataclass(output_type)
        and not issubclass(output_type, BaseModel)
        and (
            hasattr(output_type, "__total__")
            or hasattr(output_type, "__required_keys__")
            or hasattr(output_type, "__optional_keys__")
            or output_type.__module__ in ("typing", "typing_extensions")
        )
    )

    if is_typed_dict:
        try:
            hints = get_type_hints(output_type, include_extras=True)
            if hints:
                field_definitions = {}

                required_keys = set()
                if hasattr(output_type, "__required_keys__"):
                    required_keys = output_type.__required_keys__
                elif hasattr(output_type, "__total__"):
                    if output_type.__total__:
                        required_keys = set(hints.keys())

                for field_name, field_type in hints.items():
                    is_required = field_name in required_keys if required_keys else True

                    if is_required:
                        field_definitions[field_name] = (field_type, ...)
                    else:
                        if get_origin(field_type) is not type(None):
                            field_type = Optional[field_type]
                        field_definitions[field_name] = (field_type, None)

                model_dataclass_typed: Any = create_model(output_type.__name__, **cast(Any, field_definitions))
                model = cast(type[BaseModel], model_dataclass_typed)

                def _str_with_class_name(self: BaseModel) -> str:
                    return repr(self)

                model.__str__ = _str_with_class_name  # type: ignore
                model._hypertic_str_overridden = True  # type: ignore
                return model
        except Exception:
            pass

    try:
        hints = get_type_hints(output_type)
        if hints:
            field_definitions = {name: (hint, ...) for name, hint in hints.items()}
            model_dataclass_fallback: Any = create_model(output_type.__name__, **cast(Any, field_definitions))
            model = cast(type[BaseModel], model_dataclass_fallback)

            def _str_with_class_name(self: BaseModel) -> str:
                return repr(self)

            model.__str__ = _str_with_class_name  # type: ignore
            model._hypertic_str_overridden = True  # type: ignore
            return model
    except Exception:
        pass

    raise ValueError(f"Cannot convert {output_type} to Pydantic model. Supported types: Pydantic BaseModel, dataclass, or TypedDict.")


@dataclass(init=False)
class Agent:
    model: Any
    instructions: str | None = None
    tools: list[Any] | None = None
    max_steps: int = 10
    output_type: type[Any] | None = None
    parallel_calls: bool = True
    retriever: Any | None = None
    memory: Any | None = None
    guardrails: list[Any] | None = None
    handler: Any = field(default=None, init=False, repr=False)
    mcp_tools: list[Any] = field(default_factory=list, init=False, repr=False)
    function_tools: list[Any] = field(default_factory=list, init=False, repr=False)
    _tool_manager: Any = field(default=None, init=False, repr=False)
    file_processor: FileProcessor = field(default_factory=FileProcessor, init=False, repr=False)
    _tool_outputs: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _tool_calls: list[Any] = field(default_factory=list, init=False, repr=False)

    def __init__(
        self,
        *,
        model: Any,
        instructions: str | None = None,
        tools: list[Any] | None = None,
        max_steps: int = 10,
        output_type: type[Any] | None = None,
        parallel_calls: bool = True,
        retriever: Any | None = None,
        memory: Any | None = None,
        guardrails: list[Any] | None = None,
    ):
        self.model = model
        self.instructions = instructions
        self.tools = tools
        self.max_steps = max_steps
        self.output_type = output_type
        self.parallel_calls = parallel_calls
        self.retriever = retriever
        self.memory = memory
        self.guardrails = guardrails or []

        self.handler = None
        self.mcp_tools = []
        self.function_tools = []
        self._tool_manager = None
        self.file_processor = FileProcessor()
        self._tool_outputs = {}
        self._tool_calls = []

        self._initialize()

    def _initialize(self) -> None:
        if self.tools is None:
            self.tools = []

        if not isinstance(self.parallel_calls, bool):
            raise ConfigurationError("parallel_calls must be True or False")

        if hasattr(self.model, "get_handler"):
            self.handler = self.model.get_handler()
        else:
            self.handler = self.model

        self._separate_tool_types()

        if self.function_tools:
            self._tool_manager = _ToolManager(self.function_tools)
        else:
            self._tool_manager = None

        if self.output_type is not None:
            provider_name = self.handler.__class__.__name__
            try:
                _convert_to_pydantic(self.output_type)
            except Exception as e:
                raise SchemaConversionError(f"Structured output not supported for {provider_name}: {e}") from e

    def _separate_tool_types(self):
        self.mcp_tools = []
        self.function_tools = []

        if self.tools is None:
            return

        all_tools: list[Any] = []

        def process_tool(tool: Any) -> None:
            if isinstance(tool, list):
                for item in tool:
                    process_tool(item)
            elif isinstance(tool, BaseToolkit):
                toolkit_tools = tool.get_tools()
                for toolkit_tool in toolkit_tools:
                    all_tools.append(toolkit_tool)
            else:
                all_tools.append(tool)

        if isinstance(self.tools, list):
            for tool in self.tools:
                process_tool(tool)
        else:
            process_tool(self.tools)

        for tool in all_tools:
            if hasattr(tool, "_tool_metadata"):
                self.function_tools.append(tool)
            elif isinstance(tool, ExecutableTool):
                self.mcp_tools.append(tool)
            elif isinstance(tool, type):
                raise ValueError(f"Tool must be an instance, not a class: {tool.__name__}.")
            elif hasattr(tool, "name") and hasattr(tool, "inputSchema"):
                self.mcp_tools.append(tool)
            elif isinstance(tool, (str, int, float, bool, list, dict)):
                continue
            elif hasattr(tool, "__dict__"):
                self.mcp_tools.append(tool)
            else:
                self.mcp_tools.append(tool)

    async def _get_all_tools_async(self) -> list[dict[str, Any]]:
        openai_tools = []

        if self.function_tools and self._tool_manager:
            openai_tools.extend(self._tool_manager.to_openai_format())

        for tool in self.mcp_tools:
            if isinstance(tool, dict) and "function" in tool:
                openai_tools.append(tool)
            elif hasattr(tool, "name"):
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": getattr(tool, "description", ""),
                        "parameters": getattr(tool, "inputSchema", {}),
                    },
                }
                openai_tools.append(openai_tool)

        return openai_tools

    def _load_memory_context(self, user_id: str | None, session_id: str | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        if not self.memory:
            return messages

        if session_id and not user_id:
            raise ValueError(
                "session_id requires user_id to be provided. "
                "For session-based memory, both session_id and user_id are required. "
                "For long-term memory, provide only user_id."
            )

        try:
            if session_id:
                messages = self.memory.get_messages(session_id=session_id, user_id=user_id)
            elif user_id:
                messages = self.memory.get_messages(session_id=None, user_id=user_id)
        except Exception as e:
            logger.warning(f"Error loading messages: {e}")

        cleaned_messages = []
        for msg in messages:
            cleaned_msg = {"role": msg.get("role"), "content": msg.get("content")}
            cleaned_messages.append(cleaned_msg)

        return cleaned_messages

    async def _aload_memory_context(self, user_id: str | None, session_id: str | None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        if not self.memory:
            return messages

        if session_id and not user_id:
            raise ValueError(
                "session_id requires user_id to be provided. "
                "For session-based memory, both session_id and user_id are required. "
                "For long-term memory, provide only user_id."
            )

        try:
            if session_id:
                messages = await self.memory.aget_messages(session_id=session_id, user_id=user_id)
            elif user_id:
                messages = await self.memory.aget_messages(session_id=None, user_id=user_id)
        except Exception as e:
            logger.warning(f"Error loading messages: {e}", exc_info=True)

        cleaned_messages = []
        for msg in messages:
            cleaned_msg = {"role": msg.get("role"), "content": msg.get("content")}
            cleaned_messages.append(cleaned_msg)

        return cleaned_messages

    def _save_to_memory(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        user_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_outputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.memory:
            return

        try:
            self.memory.save_message(session_id=session_id, role="user", content=user_message, user_id=user_id)

            self.memory.save_message(
                session_id=session_id,
                role="assistant",
                content=assistant_message,
                user_id=user_id,
                tool_calls=tool_calls,
                tool_outputs=tool_outputs,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Error saving to memory: {e}")

    async def _asave_to_memory(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        user_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_outputs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.memory:
            return

        try:
            await self.memory.asave_message(session_id=session_id, role="user", content=user_message, user_id=user_id)

            await self.memory.asave_message(
                session_id=session_id,
                role="assistant",
                content=assistant_message,
                user_id=user_id,
                tool_calls=tool_calls,
                tool_outputs=tool_outputs,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Error saving to memory: {e}", exc_info=True)

    async def _retrieve_knowledge_async(self, query: str) -> str:
        if not self.retriever:
            return ""

        try:
            if hasattr(self.retriever, "async_search"):
                docs = await self.retriever.async_search(query)
                results = [
                    type(
                        "RetrievalResult",
                        (),
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": getattr(doc, "score", 1.0),
                            "source": doc.metadata.get("source", "unknown"),
                        },
                    )()
                    for doc in docs
                ]
                return "\n\n".join([result.content for result in results])
            else:
                error_msg = f"Unknown knowledge source type: {type(self.retriever)}"
                logger.warning(error_msg)
                raise RetrieverError(error_msg)
        except RetrieverError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}", exc_info=True)
            raise RetrieverError(f"Failed to retrieve knowledge: {e}") from e

    async def _avalidate_input(self, query: str, user_id: str | None = None, session_id: str | None = None) -> str:
        if not self.guardrails:
            return query

        modified_query = query
        for guardrail in self.guardrails:
            result = guardrail.validate_input(query=modified_query, user_id=user_id, session_id=session_id)

            if not result.allowed:
                raise GuardrailViolationError(
                    reason=result.reason or "Input blocked by guardrail",
                    violation_type=result.violation_type,
                    details=result.details,
                )

            if result.modified_input:
                modified_query = result.modified_input

        return modified_query

    def _validate_input(self, query: str, user_id: str | None = None, session_id: str | None = None) -> str:
        if not self.guardrails:
            return query

        modified_query = query
        for guardrail in self.guardrails:
            result = guardrail.validate_input(query=modified_query, user_id=user_id, session_id=session_id)

            if not result.allowed:
                raise GuardrailViolationError(
                    reason=result.reason or "Input blocked by guardrail",
                    violation_type=result.violation_type,
                    details=result.details,
                )

            if result.modified_input:
                modified_query = result.modified_input

        return modified_query

    async def arun(
        self,
        query: str,
        files: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> LLMResponse:
        """Asynchronous run method - industry standard approach.

        Args:
            query: User query
            files: Optional list of file paths
            user_id: User identifier (required if session_id is provided, optional for long-term memory)
            session_id: Session identifier (requires user_id to be provided). Auto-generated if not provided.

        Returns:
            LLMResponse with the model's response

        Note:
            Memory behavior:
            - If session_id is provided → user_id is REQUIRED (short-term memory, session-specific)
            - If only user_id is provided (no session_id) → long-term memory (all user messages)
            - Only session_id (without user_id) → NOT ALLOWED (raises ValueError)
        """
        try:
            query = await self._avalidate_input(query, user_id=user_id, session_id=session_id)

            self._tool_calls = []
            self._tool_outputs = {}

            memory_messages = await self._aload_memory_context(
                user_id=user_id,
                session_id=session_id,
            )

            if session_id is None:
                session_id = f"session_{uuid.uuid4().hex[:8]}"

            if self.retriever:
                context = await self._retrieve_knowledge_async(query)
                if context:
                    query = f"{query}\n\n{context}"

            non_stream_response: LLMResponse = await self._arun_non_streaming(
                query=query, files=files, memory_messages=memory_messages, session_id=session_id
            )

            if non_stream_response.content:
                tool_calls: list[dict[str, Any]] | None = self._tool_calls if self._tool_calls else None
                tool_outputs: dict[str, Any] | None = self._tool_outputs if self._tool_outputs else None
                llm_metadata: dict[str, Any] | None = non_stream_response.metadata if non_stream_response.metadata else None
                await self._asave_to_memory(
                    session_id,
                    query,
                    non_stream_response.content,
                    user_id=user_id,
                    tool_calls=tool_calls,
                    tool_outputs=tool_outputs,
                    metadata=llm_metadata,
                )

            if self.output_type is not None and non_stream_response.content:
                try:
                    import json

                    if isinstance(non_stream_response.content, str):
                        parsed_dict = json.loads(non_stream_response.content)
                        pydantic_model = _convert_to_pydantic(self.output_type)
                        structured_output = pydantic_model(**parsed_dict)
                        if not hasattr(structured_output.__class__, "_hypertic_str_overridden"):

                            def _str_with_class_name(self: BaseModel) -> str:
                                return repr(self)

                            structured_output.__class__.__str__ = _str_with_class_name  # type: ignore
                            structured_output.__class__._hypertic_str_overridden = True  # type: ignore
                        object.__setattr__(non_stream_response, "structured_output", structured_output)
                        object.__setattr__(non_stream_response, "content", "")
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

            return non_stream_response
        finally:
            await self._auto_disconnect_clients()

    async def astream(
        self,
        query: str,
        files: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Asynchronous streaming run method - industry standard approach.

        Args:
            query: User query
            files: Optional list of file paths
            user_id: User identifier (required if session_id is provided, optional for long-term memory)
            session_id: Session identifier (requires user_id to be provided). Auto-generated if not provided.

        Note:
            Memory behavior:
            - If session_id is provided → user_id is REQUIRED (short-term memory, session-specific)
            - If only user_id is provided (no session_id) → long-term memory (all user messages)
            - Only session_id (without user_id) → NOT ALLOWED (raises ValueError)
        """
        try:
            memory_messages = await self._aload_memory_context(
                user_id=user_id,
                session_id=session_id,
            )

            if session_id is None:
                session_id = f"session_{uuid.uuid4().hex[:8]}"

            if self.retriever:
                try:
                    context = await self._retrieve_knowledge_async(query)
                    if context:
                        query = f"{query}\n\n{context}"
                except RetrieverError as e:
                    logger.warning(f"Continuing without knowledge retrieval: {e}")

            response_content = ""
            collected_tool_calls: list[dict[str, Any]] = []
            collected_tool_outputs: dict[str, Any] = {}
            collected_metadata: dict[str, Any] | None = None

            async for chunk in self._arun_streaming(query=query, files=files, memory_messages=memory_messages):
                if chunk.type == "content":
                    response_content += chunk.content
                    if self.output_type is None:
                        yield chunk
                elif chunk.type == "tool_calls":
                    if chunk.tool_calls:
                        collected_tool_calls.extend(chunk.tool_calls)
                    yield chunk
                elif chunk.type == "tool_outputs":
                    if chunk.tool_outputs:
                        collected_tool_outputs.update(chunk.tool_outputs)
                    yield chunk
                elif chunk.type == "metadata":
                    collected_metadata = chunk.metadata
                else:
                    yield chunk

            if self.output_type is not None and response_content:
                try:
                    import json

                    from hypertic.models.events import StructuredOutputEvent

                    if isinstance(response_content, str):
                        parsed_dict = json.loads(response_content)
                        pydantic_model = _convert_to_pydantic(self.output_type)
                        structured_output = pydantic_model(**parsed_dict)
                        if not hasattr(structured_output.__class__, "_hypertic_str_overridden"):

                            def _str_with_class_name(self: BaseModel) -> str:
                                return repr(self)

                            structured_output.__class__.__str__ = _str_with_class_name  # type: ignore
                            structured_output.__class__._hypertic_str_overridden = True  # type: ignore
                        yield StructuredOutputEvent(structured_output=structured_output)
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

            if collected_metadata is not None:
                from hypertic.models.events import MetadataEvent

                yield MetadataEvent(metadata=collected_metadata)

            if response_content and self.memory:
                await self._asave_to_memory(
                    session_id,
                    query,
                    response_content,
                    user_id=user_id,
                    tool_calls=collected_tool_calls if collected_tool_calls else None,
                    tool_outputs=collected_tool_outputs if collected_tool_outputs else None,
                    metadata=collected_metadata,
                )
        finally:
            await self._auto_disconnect_clients()

    async def _arun_non_streaming(
        self,
        query: str,
        files: list[str] | None = None,
        memory_messages: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, Any]] = []

        if self.instructions is not None and self.instructions.strip():
            messages.append({"role": "system", "content": self.instructions})

        user_message: dict[str, Any] = {"role": "user", "content": query}
        if files:
            user_message["files"] = files

        if files:
            provider_name = self.handler.__class__.__name__
            user_message = self.file_processor.process_message(user_message, provider_name)

        messages.append(user_message)

        openai_tools = await self._get_all_tools_async()

        result: LLMResponse = await self._ahandle_non_streaming_with_tools(messages, openai_tools, memory_messages=memory_messages)
        return result

    async def _arun_streaming(
        self,
        query: str,
        files: list[str] | None = None,
        memory_messages: list[dict[str, Any]] | None = None,
    ):
        messages: list[dict[str, Any]] = []

        if self.instructions is not None and self.instructions.strip():
            messages.append({"role": "system", "content": self.instructions})

        user_message: dict[str, Any] = {"role": "user", "content": query}
        if files:
            user_message["files"] = files

        if files:
            provider_name = self.handler.__class__.__name__
            user_message = self.file_processor.process_message(user_message, provider_name)

        messages.append(user_message)

        openai_tools = await self._get_all_tools_async()

        async for chunk in self._ahandle_streaming_with_tools(messages, openai_tools, memory_messages=memory_messages):
            yield chunk

    async def _aexecute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        if self.function_tools and self._tool_manager:
            try:
                loop = asyncio.get_event_loop()
                result: str = await loop.run_in_executor(None, self._tool_manager.execute_tool, tool_name, arguments)
                self._tool_outputs[tool_name] = result
                return result
            except ToolNotFoundError:
                pass
            except Exception as e:
                logger.error(f"Error executing function tool '{tool_name}': {e}", exc_info=True)
                raise ToolExecutionError(f"Failed to execute function tool '{tool_name}': {e}") from e

        for tool in self.mcp_tools:
            if hasattr(tool, "name") and tool.name == tool_name:
                if hasattr(tool, "call_tool"):
                    try:
                        if asyncio.iscoroutinefunction(tool.call_tool):
                            result = await tool.call_tool(arguments)
                        else:
                            result = tool.call_tool(arguments)

                        self._tool_outputs[tool_name] = str(result)
                        return str(result)
                    except Exception as e:
                        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
                        raise ToolExecutionError(f"Failed to execute tool '{tool_name}': {e}") from e
                else:
                    result = f"Tool '{tool_name}' executed with arguments: {arguments}"
                    self._tool_outputs[tool_name] = result
                    return result

        raise ToolNotFoundError(f"Tool '{tool_name}' not found")

    async def _aexecute_tools_parallel(
        self,
        tool_calls: list[dict[str, Any]],
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, str]:
        async def execute_single_tool(tool_call):
            function_name = tool_call["function"]["name"]
            try:
                function_args = json.loads(tool_call["function"]["arguments"])
                if function_args is None:
                    function_args = {}
            except (json.JSONDecodeError, TypeError):
                function_args = {}

            result = await self._aexecute_tool(function_name, function_args, user_id=user_id, session_id=session_id)
            return function_name, result

        tasks = [execute_single_tool(tool_call) for tool_call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tool_outputs: dict[str, Any] = {}
        for result in results:
            if isinstance(result, Exception):
                if isinstance(result, ToolExecutionError | ToolNotFoundError):
                    raise result
                logger.error(f"Unexpected error during tool execution: {result}", exc_info=True)
                raise ToolExecutionError(f"Unexpected error during tool execution: {result}") from result
            else:
                if isinstance(result, tuple) and len(result) == 2:
                    function_name, tool_result = result
                    tool_outputs[function_name] = tool_result
                else:
                    raise ToolExecutionError(f"Unexpected tool result: {result!r}")

        return tool_outputs

    async def _auto_disconnect_clients(self):
        for tool in self.mcp_tools:
            if hasattr(tool, "mcp_servers") and tool.mcp_servers.initialized:
                await tool.mcp_servers.disconnect()

    async def _ahandle_non_streaming_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        memory_messages: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.handler._reset_metrics()

        if memory_messages:
            messages = memory_messages + messages

        response_format = None
        if self.output_type is not None:
            try:
                response_format = _convert_to_pydantic(self.output_type)
            except Exception as e:
                logger.warning(
                    f"Could not convert structured output type: {e}",
                    exc_info=True,
                )
                response_format = None

        for _step in range(self.max_steps):
            response = await self.handler.ahandle_non_streaming(self.handler, messages, tools, self, response_format)

            if response is None:
                continue

            return cast(LLMResponse, response)

        raise MaxStepsError(f"Maximum steps ({self.max_steps}) reached without completion")

    async def _ahandle_streaming_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        memory_messages: list[dict[str, Any]] | None = None,
    ):
        self.handler._reset_metrics()

        if memory_messages:
            messages = memory_messages + messages

        response_format = None
        if self.output_type is not None:
            try:
                response_format = _convert_to_pydantic(self.output_type)
            except Exception as e:
                logger.warning(
                    f"Could not convert structured output type: {e}",
                    exc_info=True,
                )
                response_format = None

        finish_reason = "stop"

        for _step in range(self.max_steps):
            has_more_tools = False

            async for event in self.handler.ahandle_streaming(self.handler, messages, tools, self, response_format):
                if isinstance(event, bool):
                    has_more_tools = event
                else:
                    if event.type == "tool_calls":
                        yield event
                        has_more_tools = True
                        continue
                    elif event.type == "tool_outputs":
                        yield event
                        has_more_tools = True
                        break
                    else:
                        yield event

            if not has_more_tools:
                break

        cumulative_metrics = self.handler._get_cumulative_metrics()

        params = {}
        if self.handler.temperature is not None:
            params["temperature"] = self.handler.temperature
        if self.handler.top_p is not None:
            params["top_p"] = self.handler.top_p
        if self.handler.presence_penalty is not None:
            params["presence_penalty"] = self.handler.presence_penalty
        if self.handler.frequency_penalty is not None:
            params["frequency_penalty"] = self.handler.frequency_penalty
        if self.handler.max_tokens is not None:
            params["max_tokens"] = self.handler.max_tokens

        final_metadata = {
            "model": self.handler.model,
            "params": params,
            "finish_reason": finish_reason,
            "input_tokens": cumulative_metrics.input_tokens,
            "output_tokens": cumulative_metrics.output_tokens,
        }

        yield MetadataEvent(metadata=final_metadata)

    def _get_all_tools_sync(self) -> list[dict[str, Any]]:
        openai_tools = []

        if self.function_tools and self._tool_manager:
            openai_tools.extend(self._tool_manager.to_openai_format())

        return openai_tools

    def _retrieve_knowledge_sync(self, query: str) -> str:
        if not self.retriever:
            return ""

        try:
            if hasattr(self.retriever, "search"):
                docs = self.retriever.search(query)
                results = [
                    type(
                        "RetrievalResult",
                        (),
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": getattr(doc, "score", 1.0),
                            "source": doc.metadata.get("source", "unknown"),
                        },
                    )()
                    for doc in docs
                ]
                return "\n\n".join([result.content for result in results])
            else:
                error_msg = f"Unknown knowledge source type: {type(self.retriever)}"
                logger.warning(error_msg)
                raise RetrieverError(error_msg)
        except RetrieverError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving knowledge: {e}", exc_info=True)
            raise RetrieverError(f"Failed to retrieve knowledge: {e}") from e

    def run(
        self,
        query: str,
        files: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> LLMResponse:
        """Synchronous run method - industry standard approach.

        Args:
            query: User query
            files: Optional list of file paths
            user_id: User identifier (required if session_id is provided, optional for long-term memory)
            session_id: Session identifier (requires user_id to be provided). Auto-generated if not provided.

        Returns:
            LLMResponse with content and metadata

        Note:
            Memory behavior:
            - If session_id is provided → user_id is REQUIRED (short-term memory, session-specific)
            - If only user_id is provided (no session_id) → long-term memory (all user messages)
            - Only session_id (without user_id) → NOT ALLOWED (raises ValueError)
        """
        query = self._validate_input(query, user_id=user_id, session_id=session_id)

        self._tool_calls = []
        self._tool_outputs = {}

        memory_messages = self._load_memory_context(
            user_id=user_id,
            session_id=session_id,
        )

        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        if self.retriever:
            try:
                context = self._retrieve_knowledge_sync(query)
                if context:
                    query = f"{query}\n\n{context}"
            except RetrieverError as e:
                logger.warning(f"Continuing without knowledge retrieval: {e}")

        response: LLMResponse = self._run_non_streaming(query=query, files=files, memory_messages=memory_messages, session_id=session_id)

        if response.content:
            tool_calls = self._tool_calls if self._tool_calls else None
            tool_outputs = self._tool_outputs if self._tool_outputs else None
            metadata = response.metadata if response.metadata else None
            self._save_to_memory(
                session_id,
                query,
                response.content,
                user_id=user_id,
                tool_calls=tool_calls,
                tool_outputs=tool_outputs,
                metadata=metadata,
            )

        if self.output_type is not None and response.content:
            try:
                import json

                if isinstance(response.content, str):
                    parsed_dict = json.loads(response.content)
                    pydantic_model = _convert_to_pydantic(self.output_type)
                    structured_output = pydantic_model(**parsed_dict)
                    if not hasattr(structured_output.__class__, "_hypertic_str_overridden"):

                        def _str_with_class_name(self: BaseModel) -> str:
                            return repr(self)

                        structured_output.__class__.__str__ = _str_with_class_name  # type: ignore
                        structured_output.__class__._hypertic_str_overridden = True  # type: ignore
                    object.__setattr__(response, "structured_output", structured_output)
                    object.__setattr__(response, "content", "")
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return response

    def stream(
        self,
        query: str,
        files: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Iterator[StreamEvent]:
        """Synchronous streaming run method - industry standard approach.

        Args:
            query: User query
            files: Optional list of file paths
            user_id: User identifier (required if session_id is provided, optional for long-term memory)
            session_id: Session identifier (requires user_id to be provided). Auto-generated if not provided.

        Note:
            Memory behavior:
            - If session_id is provided → user_id is REQUIRED (short-term memory, session-specific)
            - If only user_id is provided (no session_id) → long-term memory (all user messages)
            - Only session_id (without user_id) → NOT ALLOWED (raises ValueError)
        """
        memory_messages = self._load_memory_context(
            user_id=user_id,
            session_id=session_id,
        )

        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        if self.retriever:
            try:
                context = self._retrieve_knowledge_sync(query)
                if context:
                    query = f"{query}\n\n{context}"
            except RetrieverError as e:
                logger.warning(f"Continuing without knowledge retrieval: {e}")

        response_content = ""
        collected_tool_calls: list[dict[str, Any]] = []
        collected_tool_outputs: dict[str, Any] = {}
        collected_metadata: dict[str, Any] | None = None

        for chunk in self._run_streaming(query=query, files=files, memory_messages=memory_messages):
            if chunk.type == "content":
                response_content += chunk.content
                if self.output_type is None:
                    yield chunk
            elif chunk.type == "tool_calls":
                if chunk.tool_calls:
                    collected_tool_calls.extend(chunk.tool_calls)
                yield chunk
            elif chunk.type == "tool_outputs":
                if chunk.tool_outputs:
                    collected_tool_outputs.update(chunk.tool_outputs)
                yield chunk
            elif chunk.type == "metadata":
                collected_metadata = chunk.metadata
            else:
                yield chunk

        if self.output_type is not None and response_content:
            try:
                import json

                from hypertic.models.events import StructuredOutputEvent

                if isinstance(response_content, str):
                    parsed_dict = json.loads(response_content)
                    pydantic_model = _convert_to_pydantic(self.output_type)
                    structured_output = pydantic_model(**parsed_dict)
                    if not hasattr(structured_output.__class__, "_hypertic_str_overridden"):

                        def _str_with_class_name(self: BaseModel) -> str:
                            return repr(self)

                        structured_output.__class__.__str__ = _str_with_class_name  # type: ignore
                        structured_output.__class__._hypertic_str_overridden = True  # type: ignore
                    yield StructuredOutputEvent(structured_output=structured_output)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        if collected_metadata is not None:
            from hypertic.models.events import MetadataEvent

            yield MetadataEvent(metadata=collected_metadata)

        if response_content and self.memory:
            self._save_to_memory(
                session_id,
                query,
                response_content,
                user_id=user_id,
                tool_calls=collected_tool_calls if collected_tool_calls else None,
                tool_outputs=collected_tool_outputs if collected_tool_outputs else None,
                metadata=collected_metadata,
            )

    def _run_non_streaming(
        self,
        query: str,
        files: list[str] | None = None,
        memory_messages: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, Any]] = []
        if self.instructions is not None and self.instructions.strip():
            messages.append({"role": "system", "content": self.instructions})

        user_message: dict[str, Any] = {"role": "user", "content": query}
        if files:
            user_message["files"] = files

        if files:
            provider_name = self.handler.__class__.__name__
            user_message = self.file_processor.process_message(user_message, provider_name)

        messages.append(user_message)

        openai_tools = self._get_all_tools_sync()

        return cast(
            LLMResponse,
            self._handle_non_streaming_with_tools(messages, openai_tools, memory_messages=memory_messages),
        )

    def _run_streaming(
        self,
        query: str,
        files: list[str] | None = None,
        memory_messages: list[dict[str, Any]] | None = None,
    ):
        messages: list[dict[str, Any]] = []
        if self.instructions is not None and self.instructions.strip():
            messages.append({"role": "system", "content": self.instructions})

        user_message: dict[str, Any] = {"role": "user", "content": query}
        if files:
            user_message["files"] = files

        if files:
            provider_name = self.handler.__class__.__name__
            user_message = self.file_processor.process_message(user_message, provider_name)

        messages.append(user_message)

        openai_tools = self._get_all_tools_sync()

        yield from self._handle_streaming_with_tools(messages, openai_tools, memory_messages=memory_messages)

    def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        if self.function_tools and self._tool_manager:
            try:
                result: str = str(self._tool_manager.execute_tool(tool_name, arguments))
                self._tool_outputs[tool_name] = result
                return result
            except ToolNotFoundError:
                pass
            except Exception as e:
                logger.error(f"Error executing function tool '{tool_name}': {e}", exc_info=True)
                raise ToolExecutionError(f"Failed to execute function tool '{tool_name}': {e}") from e

        for tool in self.mcp_tools:
            if hasattr(tool, "name") and tool.name == tool_name:
                if hasattr(tool, "mcp_servers"):
                    raise ToolExecutionError(
                        f"Tool '{tool_name}' is an MCP tool (async-only). Use agent.arun() or agent.astream() instead of run() or stream()."
                    )

                if hasattr(tool, "call_tool"):
                    try:
                        result = tool.call_tool(arguments)
                        self._tool_outputs[tool_name] = str(result)
                        return str(result)
                    except Exception as e:
                        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
                        raise ToolExecutionError(f"Failed to execute tool '{tool_name}': {e}") from e
                else:
                    result = f"Tool '{tool_name}' executed with arguments: {arguments}"
                    self._tool_outputs[tool_name] = result
                    return result

        raise ToolNotFoundError(f"Tool '{tool_name}' not found")

    def _execute_tools_parallel(
        self,
        tool_calls: list[dict[str, Any]],
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, str]:
        def execute_single_tool(tool_call):
            function_name = tool_call["function"]["name"]
            try:
                function_args = json.loads(tool_call["function"]["arguments"])
                if function_args is None:
                    function_args = {}
            except (json.JSONDecodeError, TypeError):
                function_args = {}

            result = self._execute_tool(function_name, function_args, user_id=user_id, session_id=session_id)
            return function_name, result

        results = [execute_single_tool(tool_call) for tool_call in tool_calls]

        tool_outputs = {}
        for result in results:
            if isinstance(result, Exception):
                if isinstance(result, ToolExecutionError | ToolNotFoundError):
                    raise result
                logger.error(f"Unexpected error during tool execution: {result}", exc_info=True)
                raise ToolExecutionError(f"Unexpected error during tool execution: {result}") from result
            else:
                function_name, tool_result = result
                tool_outputs[function_name] = tool_result

        return tool_outputs

    def _handle_non_streaming_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        memory_messages: list[dict[str, Any]] | None = None,
    ) -> Any:
        self.handler._reset_metrics()

        if memory_messages:
            messages = memory_messages + messages

        response_format = None
        if self.output_type is not None:
            try:
                response_format = _convert_to_pydantic(self.output_type)
            except Exception as e:
                logger.warning(
                    f"Could not convert structured output type: {e}",
                    exc_info=True,
                )
                response_format = None

        for _step in range(self.max_steps):
            response: LLMResponse | None = self.handler.handle_non_streaming(self.handler, messages, tools, self, response_format)

            if response is None:
                continue

            return response

        raise MaxStepsError(f"Maximum steps ({self.max_steps}) reached without completion")

    def _handle_streaming_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        memory_messages: list[dict[str, Any]] | None = None,
    ):
        self.handler._reset_metrics()

        if memory_messages:
            messages = memory_messages + messages

        response_format = None
        if self.output_type is not None:
            try:
                response_format = _convert_to_pydantic(self.output_type)
            except Exception as e:
                logger.warning(
                    f"Could not convert structured output type: {e}",
                    exc_info=True,
                )
                response_format = None

        finish_reason = "stop"

        for _step in range(self.max_steps):
            has_more_tools = False

            for event in self.handler.handle_streaming(self.handler, messages, tools, self, response_format):
                if isinstance(event, bool):
                    has_more_tools = event
                else:
                    if event.type == "tool_calls":
                        yield event
                        has_more_tools = True
                        continue
                    elif event.type == "tool_outputs":
                        yield event
                        has_more_tools = True
                        break
                    else:
                        yield event

            if not has_more_tools:
                break

        cumulative_metrics = self.handler._get_cumulative_metrics()

        params = {}
        if self.handler.temperature is not None:
            params["temperature"] = self.handler.temperature
        if self.handler.top_p is not None:
            params["top_p"] = self.handler.top_p
        if self.handler.presence_penalty is not None:
            params["presence_penalty"] = self.handler.presence_penalty
        if self.handler.frequency_penalty is not None:
            params["frequency_penalty"] = self.handler.frequency_penalty
        if self.handler.max_tokens is not None:
            params["max_tokens"] = self.handler.max_tokens

        final_metadata = {
            "model": self.handler.model,
            "params": params,
            "finish_reason": finish_reason,
            "input_tokens": cumulative_metrics.input_tokens,
            "output_tokens": cumulative_metrics.output_tokens,
        }

        yield MetadataEvent(metadata=final_metadata)
