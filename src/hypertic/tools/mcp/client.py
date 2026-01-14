import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

from hypertic.utils.log import get_logger

logger = get_logger(__name__)


def _get_streamable_http_client(url: str, headers: dict[str, str] | None, timeout: float, sse_read_timeout: float):  # noqa: ANN201
    """Get streamable HTTP client - wrapper for deprecated function.

    Note: streamablehttp_client is deprecated but has no replacement in current mcp version.
    This wrapper isolates the deprecation to a single location.
    """
    from mcp.client.streamable_http import streamablehttp_client  # noqa: PLC0415

    return streamablehttp_client(url, headers, timeout, sse_read_timeout)


@dataclass
class ExecutableTool:
    tool: Any
    mcp_servers: "MCPServers"
    name: str = field(init=False)
    description: str = field(init=False)
    input_schema: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.name = self.tool.name
        self.description = getattr(self.tool, "description", "")
        self.input_schema = getattr(self.tool, "inputSchema", {})

    def __repr__(self) -> str:
        tool_dict = {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }
        return str(tool_dict)

    async def call_tool(self, arguments: dict[str, Any]) -> Any:
        return await self.mcp_servers.call_tool(self.name, arguments)


@dataclass
class MCPServerConfig:
    name: str
    transport: str
    config: dict[str, Any]


@dataclass
class MCPServers:
    config: dict[str, Any] | list[dict[str, Any]]

    servers: dict[str, MCPServerConfig] = field(default_factory=dict, init=False)
    sessions: dict[str, ClientSession] = field(default_factory=dict, init=False)
    exit_stack: AsyncExitStack = field(default_factory=AsyncExitStack, init=False)
    tools: dict[str, Any] = field(default_factory=dict, init=False)
    resources: dict[str, Any] = field(default_factory=dict, init=False)
    prompts: dict[str, Any] = field(default_factory=dict, init=False)
    initialized: bool = field(default=False, init=False)
    _auto_initialized: bool = field(default=False, init=False)

    def __post_init__(self):
        configs = [self.config] if isinstance(self.config, dict) else self.config

        for server_config in configs:
            self._add_server_config(server_config)

    def _add_server_config(self, config: dict[str, Any]):
        for server_name, server_config in config.items():
            transport = server_config.get("transport", "stdio")

            if transport == "stdio" and "command" in server_config:
                pass
            elif "url" in server_config:
                if transport == "streamable_http":
                    pass
                elif server_config.get("streamable", False):
                    transport = "streamable_http"
                else:
                    transport = "sse"

            self.servers[server_name] = MCPServerConfig(name=server_name, transport=transport, config=server_config)

    @classmethod
    def from_config_file(cls, *config_files: str, tools: list[str] | None = None) -> "MCPServers":
        configs = []
        for config_file in config_files:
            with open(config_file) as f:
                if config_file.lower().endswith((".yaml", ".yml")):
                    try:
                        import yaml

                        configs.append(yaml.safe_load(f))
                    except ImportError as err:
                        raise ImportError(
                            "YAML processing requires PyYAML. This should be installed by default. If missing, install with: pip install PyYAML"
                        ) from err
                else:
                    configs.append(json.load(f))
        return cls(configs)

    async def initialize(self) -> "MCPServers":
        if self.initialized:
            return self

        for server_name in self.servers:
            server_config = self.servers[server_name]
            session = await self._create_session(server_config)
            self.sessions[server_name] = session

            await self._discover_capabilities(server_name, session)

        self.initialized = True
        return self

    async def _create_session(self, server_config: MCPServerConfig) -> ClientSession:
        if server_config.transport == "stdio":
            return await self._create_stdio_session(server_config.config)
        elif server_config.transport == "sse":
            return await self._create_sse_session(server_config.config)
        elif server_config.transport == "streamable_http":
            return await self._create_streamable_http_session(server_config.config)
        else:
            raise ValueError(f"Unsupported transport type: {server_config.transport}")

    async def _create_stdio_session(self, config: dict[str, Any]) -> ClientSession:
        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env", {}),
            cwd=config.get("cwd"),
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport

        session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        return session

    async def _create_sse_session(self, config: dict[str, Any]) -> ClientSession:
        url = config["url"]
        headers = config.get("headers", {})
        timeout = config.get("timeout", 5)
        sse_read_timeout = config.get("sse_read_timeout", 60 * 5)

        sse_transport = await self.exit_stack.enter_async_context(sse_client(url, headers, timeout, sse_read_timeout))
        read_stream, write_stream = sse_transport

        session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        return session

    async def _create_streamable_http_session(self, config: dict[str, Any]) -> ClientSession:
        url = config["url"]
        headers = config.get("headers", {})
        timeout = config.get("timeout", 5)
        sse_read_timeout = config.get("sse_read_timeout", 60 * 5)

        # Use wrapper function to isolate deprecated import
        http_transport: tuple[Any, Any, Any] = await self.exit_stack.enter_async_context(
            _get_streamable_http_client(url, headers, timeout, sse_read_timeout)
        )
        read_stream, write_stream, _ = http_transport

        session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        return session

    async def _discover_capabilities(self, server_name: str, session: ClientSession):
        try:
            try:
                tools_response = await session.list_tools()

                tools_to_add = tools_response.tools

                for tool in tools_to_add:
                    self.tools[tool.name] = ExecutableTool(tool, self)

                logger.debug(f"Discovered {len(tools_to_add)} tools from {server_name} (filtered from {len(tools_response.tools)})")
            except Exception as e:
                logger.debug(f"Server {server_name} does not support list_tools: {e}")

            try:
                resources_response = await session.list_resources()
                for resource in resources_response.resources:
                    self.resources[str(resource.uri)] = resource
                logger.debug(f"Discovered {len(resources_response.resources)} resources from {server_name}")
            except Exception as e:
                logger.debug(f"Server {server_name} does not support list_resources: {e}")

            try:
                prompts_response = await session.list_prompts()
                for prompt in prompts_response.prompts:
                    self.prompts[prompt.name] = prompt
                logger.debug(f"Discovered {len(prompts_response.prompts)} prompts from {server_name}")
            except Exception as e:
                logger.debug(f"Server {server_name} does not support list_prompts: {e}")

        except Exception as e:
            logger.error(f"Failed to discover capabilities from {server_name}: {e}")

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self.initialized:
            await self.initialize()

        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        try:
            server_name = None
            for name, session in self.sessions.items():
                tools_response = await session.list_tools()
                if any(t.name == tool_name for t in tools_response.tools):
                    server_name = name
                    break

            if not server_name:
                raise ValueError(f"Tool '{tool_name}' not found on any server")

            session = self.sessions[server_name]

            result = await session.call_tool(tool_name, arguments)

            if hasattr(result, "content") and result.content:
                content_parts = []
                for content in result.content:
                    if hasattr(content, "text"):
                        content_parts.append(content.text)
                    elif hasattr(content, "data"):
                        content_parts.append(str(content.data))

                return "\n".join(content_parts) if content_parts else "Tool call completed"
            else:
                return "Tool call completed"

        except Exception as e:
            logger.error(f"Failed to call tool {tool_name}: {e}")
            raise

    async def read_resource(self, uri: str) -> str:
        if not self.initialized:
            await self.initialize()

        resource = self.resources.get(uri)
        if not resource:
            raise ValueError(f"Resource '{uri}' not found")

        try:
            server_name = None
            for name, session in self.sessions.items():
                resources_response = await session.list_resources()
                if any(str(r.uri) == uri for r in resources_response.resources):
                    server_name = name
                    break

            if not server_name:
                raise ValueError(f"Resource '{uri}' not found on any server")

            session = self.sessions[server_name]

            uri_url = AnyUrl(uri)
            result = await session.read_resource(uri_url)

            if hasattr(result, "contents") and result.contents:
                content_parts = []
                for content in result.contents:
                    if hasattr(content, "text"):
                        content_parts.append(content.text)
                    elif hasattr(content, "data"):
                        content_parts.append(str(content.data))

                return "\n".join(content_parts) if content_parts else ""
            else:
                return ""

        except Exception as e:
            logger.error(f"Failed to read resource {uri}: {e}")
            raise

    async def get_prompt(self, prompt_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.initialized:
            await self.initialize()

        prompt = self.prompts.get(prompt_name)
        if not prompt:
            raise ValueError(f"Prompt '{prompt_name}' not found")

        try:
            server_name = None
            for name, session in self.sessions.items():
                prompts_response = await session.list_prompts()
                if any(p.name == prompt_name for p in prompts_response.prompts):
                    server_name = name
                    break

            if not server_name:
                raise ValueError(f"Prompt '{prompt_name}' not found on any server")

            session = self.sessions[server_name]

            result = await session.get_prompt(prompt_name, arguments or {})

            if hasattr(result, "messages"):
                messages = []
                for msg in result.messages:
                    content_text = ""
                    if msg.content:
                        if isinstance(msg.content, list | tuple) and len(msg.content) > 0:
                            first_content = msg.content[0]
                            if hasattr(first_content, "text"):
                                content_text = first_content.text
                        elif hasattr(msg.content, "text"):
                            content_text = msg.content.text
                    messages.append({"role": msg.role, "content": content_text})
                return {"messages": messages}
            else:
                return {}

        except Exception as e:
            logger.error(f"Failed to get prompt {prompt_name}: {e}")
            raise

    def get_servers_info(self) -> list[Any]:
        return list(self.tools.values())

    def get_resources(self) -> list[Any]:
        return list(self.resources.values())

    def get_prompts(self) -> list[Any]:
        return list(self.prompts.values())

    async def get_tools(self, tool_names: list[str] | None = None) -> list[Any]:
        if not self.initialized:
            await self.initialize()
            self._auto_initialized = True

        if tool_names:
            tool_names_lower = {name.lower() for name in tool_names}
            mcp_tools = []
            for tool_name, tool in self.tools.items():
                if tool_name.lower() in tool_names_lower:
                    mcp_tools.append(tool)
            return mcp_tools

        return list(self.tools.values())

    def to_openai_format(self) -> list[dict[str, Any]]:
        if not self.initialized:
            return []

        openai_tools = []
        for tool in self.tools.values():
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema or {},
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools

    def format_tools_for_prompt(self) -> str:
        if not self.initialized:
            return "MCP tools will be available after initialization"

        if not self.tools:
            return "No MCP tools available"

        tools_info = []
        for tool in self.tools.values():
            tool_info = f"- {tool.name}: {tool.description or 'No description'}"
            if hasattr(tool, "input_schema") and tool.input_schema and "properties" in tool.input_schema:
                params = ", ".join(tool.input_schema["properties"].keys())
                tool_info += f" (parameters: {params})"
            tools_info.append(tool_info)

        return "\n".join(tools_info)

    async def disconnect(self):
        try:
            for server_name, session in list(self.sessions.items()):
                try:
                    if hasattr(session, "close"):
                        await session.close()
                except Exception as e:
                    logger.debug(f"Error closing session {server_name}: {e}")

            self.sessions.clear()

            try:
                await self.exit_stack.aclose()
            except Exception as e:
                if any(
                    keyword in str(e).lower()
                    for keyword in [
                        "cancel scope",
                        "taskgroup",
                        "generator didn't stop",
                        "cancellederror",
                        "generatorexit",
                        "baseexceptiongroup",
                        "unhandled errors in a taskgroup",
                    ]
                ):
                    logger.debug(f"MCP client exit stack cleanup error (suppressed): {e}")
                else:
                    logger.error(f"Error closing exit stack: {e}")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            self.initialized = False

    async def __aenter__(self):
        if not self.initialized:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
