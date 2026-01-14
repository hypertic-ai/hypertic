from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from hypertic.tools.mcp.client import ExecutableTool, MCPServerConfig, MCPServers


class TestMCPServerConfig:
    def test_server_config_creation(self):
        config = MCPServerConfig(
            name="test_server",
            transport="stdio",
            config={"command": "python", "args": ["server.py"]},
        )
        assert config.name == "test_server"
        assert config.transport == "stdio"
        assert config.config == {"command": "python", "args": ["server.py"]}


class TestExecutableTool:
    @pytest.fixture
    def mock_tool(self):
        tool = Mock()
        tool.name = "test_tool"
        tool.description = "Test tool description"
        tool.inputSchema = {"type": "object", "properties": {}}
        return tool

    @pytest.fixture
    def mock_mcp_client(self):
        client = Mock()
        client.call_tool = AsyncMock(return_value="result")
        return client

    @pytest.fixture
    def executable_tool(self, mock_tool, mock_mcp_client):
        return ExecutableTool(tool=mock_tool, mcp_servers=mock_mcp_client)

    def test_executable_tool_creation(self, executable_tool):
        assert executable_tool.name == "test_tool"
        assert executable_tool.description == "Test tool description"
        assert executable_tool.input_schema == {"type": "object", "properties": {}}

    def test_executable_tool_creation_no_description(self):
        """Test ExecutableTool with tool that has no description."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        del mock_tool.description
        mock_mcp_client = Mock()
        tool = ExecutableTool(tool=mock_tool, mcp_servers=mock_mcp_client)
        assert tool.description == ""

    def test_executable_tool_creation_no_input_schema(self):
        """Test ExecutableTool with tool that has no inputSchema."""
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        del mock_tool.inputSchema
        mock_mcp_client = Mock()
        tool = ExecutableTool(tool=mock_tool, mcp_servers=mock_mcp_client)
        assert tool.input_schema == {}

    def test_executable_tool_repr(self, executable_tool):
        """Test ExecutableTool __repr__."""
        repr_str = repr(executable_tool)
        assert "test_tool" in repr_str
        assert "Test tool description" in repr_str

    @pytest.mark.asyncio
    async def test_executable_tool_call_tool(self, executable_tool, mock_mcp_client):
        result = await executable_tool.call_tool({"param": "value"})
        assert result == "result"
        mock_mcp_client.call_tool.assert_called_once_with("test_tool", {"param": "value"})


class TestMCPServers:
    @pytest.fixture
    def mcp_config(self):
        return {
            "test_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["server.py"],
            }
        }

    @pytest.fixture
    def mcp_servers(self, mcp_config):
        return MCPServers(config=mcp_config)

    def test_mcp_servers_creation(self, mcp_servers, mcp_config):
        assert mcp_servers.config == mcp_config
        assert len(mcp_servers.servers) > 0

    def test_mcp_servers_parse_config_dict(self):
        config = {"server1": {"transport": "stdio", "command": "python"}}
        servers = MCPServers(config=config)
        assert "server1" in servers.servers

    def test_mcp_servers_parse_config_list(self):
        config = [{"server1": {"transport": "stdio", "command": "python"}}]
        servers = MCPServers(config=config)
        assert "server1" in servers.servers

    def test_add_server_config_stdio(self):
        """Test _add_server_config with stdio transport."""
        config = {"server1": {"transport": "stdio", "command": "python"}}
        servers = MCPServers(config=config)
        assert servers.servers["server1"].transport == "stdio"

    def test_add_server_config_sse(self):
        """Test _add_server_config with SSE transport."""
        config = {"server1": {"transport": "sse", "url": "http://localhost:8000"}}
        servers = MCPServers(config=config)
        assert servers.servers["server1"].transport == "sse"

    def test_add_server_config_streamable_http(self):
        """Test _add_server_config with streamable_http transport."""
        config = {"server1": {"transport": "streamable_http", "url": "http://localhost:8000"}}
        servers = MCPServers(config=config)
        assert servers.servers["server1"].transport == "streamable_http"

    def test_add_server_config_streamable_flag(self):
        """Test _add_server_config with streamable flag."""
        config = {"server1": {"url": "http://localhost:8000", "streamable": True}}
        servers = MCPServers(config=config)
        assert servers.servers["server1"].transport == "streamable_http"

    def test_add_server_config_default_transport(self):
        """Test _add_server_config with default transport."""
        config = {"server1": {"url": "http://localhost:8000"}}
        servers = MCPServers(config=config)
        assert servers.servers["server1"].transport == "sse"

    @pytest.mark.asyncio
    async def test_mcp_servers_initialize(self, mcp_servers):
        mock_session = MagicMock()
        with (
            patch.object(mcp_servers, "_create_session", AsyncMock(return_value=mock_session)),
            patch.object(mcp_servers, "_discover_capabilities", AsyncMock(return_value=None)),
        ):
            result = await mcp_servers.initialize()
            assert result is mcp_servers
            assert mcp_servers.initialized is True

    @pytest.mark.asyncio
    async def test_mcp_servers_initialize_already_initialized(self, mcp_servers):
        """Test initialize when already initialized."""
        mcp_servers.initialized = True
        result = await mcp_servers.initialize()
        assert result is mcp_servers

    @pytest.mark.asyncio
    async def test_create_stdio_session(self, mcp_servers):
        """Test _create_stdio_session."""
        config = {"command": "python", "args": ["server.py"], "env": {}, "cwd": "/tmp"}
        mock_transport = (MagicMock(), MagicMock())
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()

        with patch("hypertic.tools.mcp.client.stdio_client", return_value=AsyncMock()) as mock_stdio:
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_transport)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
            with patch("hypertic.tools.mcp.client.ClientSession", return_value=mock_session) as mock_session_class:
                mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
                session = await mcp_servers._create_stdio_session(config)
                assert session == mock_session

    @pytest.mark.asyncio
    async def test_create_sse_session(self, mcp_servers):
        """Test _create_sse_session."""
        config = {"url": "http://localhost:8000", "headers": {}, "timeout": 5, "sse_read_timeout": 300}
        mock_transport = (MagicMock(), MagicMock())
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()

        with patch("hypertic.tools.mcp.client.sse_client", return_value=AsyncMock()) as mock_sse:
            mock_sse.return_value.__aenter__ = AsyncMock(return_value=mock_transport)
            mock_sse.return_value.__aexit__ = AsyncMock(return_value=None)
            with patch("hypertic.tools.mcp.client.ClientSession", return_value=mock_session) as mock_session_class:
                mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
                session = await mcp_servers._create_sse_session(config)
                assert session == mock_session

    @pytest.mark.asyncio
    async def test_create_streamable_http_session(self, mcp_servers):
        """Test _create_streamable_http_session."""
        config = {"url": "http://localhost:8000", "headers": {}, "timeout": 5, "sse_read_timeout": 300}
        mock_transport = (MagicMock(), MagicMock(), MagicMock())
        mock_session = MagicMock()
        mock_session.initialize = AsyncMock()

        with patch("hypertic.tools.mcp.client._get_streamable_http_client", return_value=AsyncMock()) as mock_http:
            mock_http.return_value.__aenter__ = AsyncMock(return_value=mock_transport)
            mock_http.return_value.__aexit__ = AsyncMock(return_value=None)
            with patch("hypertic.tools.mcp.client.ClientSession", return_value=mock_session) as mock_session_class:
                mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_class.return_value.__aexit__ = AsyncMock(return_value=None)
                session = await mcp_servers._create_streamable_http_session(config)
                assert session == mock_session

    @pytest.mark.asyncio
    async def test_create_session_unsupported_transport(self, mcp_servers):
        """Test _create_session with unsupported transport."""
        server_config = MCPServerConfig(name="test", transport="unsupported", config={})
        with pytest.raises(ValueError, match="Unsupported transport type"):
            await mcp_servers._create_session(server_config)

    @pytest.mark.asyncio
    async def test_discover_capabilities_tools(self, mcp_servers):
        """Test _discover_capabilities discovers tools."""
        mock_session = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tools_response = MagicMock()
        mock_tools_response.tools = [mock_tool]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)
        mock_session.list_resources = AsyncMock(side_effect=Exception("Not supported"))
        mock_session.list_prompts = AsyncMock(side_effect=Exception("Not supported"))

        await mcp_servers._discover_capabilities("test_server", mock_session)
        assert "test_tool" in mcp_servers.tools

    @pytest.mark.asyncio
    async def test_discover_capabilities_resources(self, mcp_servers):
        """Test _discover_capabilities discovers resources."""
        mock_session = MagicMock()
        mock_resource = MagicMock()
        mock_resource.uri = "test://resource"
        mock_resources_response = MagicMock()
        mock_resources_response.resources = [mock_resource]
        mock_session.list_tools = AsyncMock(side_effect=Exception("Not supported"))
        mock_session.list_resources = AsyncMock(return_value=mock_resources_response)
        mock_session.list_prompts = AsyncMock(side_effect=Exception("Not supported"))

        await mcp_servers._discover_capabilities("test_server", mock_session)
        assert "test://resource" in mcp_servers.resources

    @pytest.mark.asyncio
    async def test_discover_capabilities_prompts(self, mcp_servers):
        """Test _discover_capabilities discovers prompts."""
        mock_session = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.name = "test_prompt"
        mock_prompts_response = MagicMock()
        mock_prompts_response.prompts = [mock_prompt]
        mock_session.list_tools = AsyncMock(side_effect=Exception("Not supported"))
        mock_session.list_resources = AsyncMock(side_effect=Exception("Not supported"))
        mock_session.list_prompts = AsyncMock(return_value=mock_prompts_response)

        await mcp_servers._discover_capabilities("test_server", mock_session)
        assert "test_prompt" in mcp_servers.prompts

    @pytest.mark.asyncio
    async def test_call_tool(self, mcp_servers):
        """Test call_tool method."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mcp_servers.tools["test_tool"] = ExecutableTool(tool=mock_tool, mcp_servers=mcp_servers)
        mock_session = MagicMock()
        mock_tools_response = MagicMock()
        mock_tool_obj = MagicMock()
        mock_tool_obj.name = "test_tool"
        mock_tools_response.tools = [mock_tool_obj]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)
        mock_content = MagicMock()
        mock_content.text = "result text"
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mcp_servers.sessions["test_server"] = mock_session
        mcp_servers.initialized = True

        result = await mcp_servers.call_tool("test_tool", {"param": "value"})
        assert result == "result text"

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self, mcp_servers):
        """Test call_tool with tool not found."""
        mcp_servers.initialized = True
        with pytest.raises(ValueError, match="Tool 'unknown_tool' not found"):
            await mcp_servers.call_tool("unknown_tool", {})

    @pytest.mark.asyncio
    async def test_call_tool_content_data(self, mcp_servers):
        """Test call_tool with content.data instead of content.text."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mcp_servers.tools["test_tool"] = ExecutableTool(tool=mock_tool, mcp_servers=mcp_servers)
        mock_session = MagicMock()
        mock_tools_response = MagicMock()
        mock_tool_obj = MagicMock()
        mock_tool_obj.name = "test_tool"
        mock_tools_response.tools = [mock_tool_obj]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)
        # Create a mock that has data attribute but no text attribute
        mock_content = MagicMock()
        del mock_content.text  # Remove text attribute
        type(mock_content).data = "result data"  # Set data as a property
        mock_result = MagicMock()
        mock_result.content = [mock_content]
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mcp_servers.sessions["test_server"] = mock_session
        mcp_servers.initialized = True

        result = await mcp_servers.call_tool("test_tool", {})
        assert "result data" in result

    @pytest.mark.asyncio
    async def test_call_tool_no_content(self, mcp_servers):
        """Test call_tool with no content."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mcp_servers.tools["test_tool"] = ExecutableTool(tool=mock_tool, mcp_servers=mcp_servers)
        mock_session = MagicMock()
        mock_tools_response = MagicMock()
        mock_tool_obj = MagicMock()
        mock_tool_obj.name = "test_tool"
        mock_tools_response.tools = [mock_tool_obj]
        mock_session.list_tools = AsyncMock(return_value=mock_tools_response)
        mock_result = MagicMock()
        mock_result.content = None
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        mcp_servers.sessions["test_server"] = mock_session
        mcp_servers.initialized = True

        result = await mcp_servers.call_tool("test_tool", {})
        assert result == "Tool call completed"

    @pytest.mark.asyncio
    async def test_read_resource(self, mcp_servers):
        """Test read_resource method."""
        mock_resource = MagicMock()
        mock_resource.uri = "test://resource"
        mcp_servers.resources["test://resource"] = mock_resource
        mock_session = MagicMock()
        mock_resources_response = MagicMock()
        mock_resources_response.resources = [mock_resource]
        mock_session.list_resources = AsyncMock(return_value=mock_resources_response)
        mock_content = MagicMock()
        mock_content.text = "resource content"
        mock_result = MagicMock()
        mock_result.contents = [mock_content]
        mock_session.read_resource = AsyncMock(return_value=mock_result)
        mcp_servers.sessions["test_server"] = mock_session
        mcp_servers.initialized = True

        result = await mcp_servers.read_resource("test://resource")
        assert result == "resource content"

    @pytest.mark.asyncio
    async def test_get_prompt(self, mcp_servers):
        """Test get_prompt method."""
        mock_prompt = MagicMock()
        mock_prompt.name = "test_prompt"
        mcp_servers.prompts["test_prompt"] = mock_prompt
        mock_session = MagicMock()
        mock_prompts_response = MagicMock()
        mock_prompts_response.prompts = [mock_prompt]
        mock_session.list_prompts = AsyncMock(return_value=mock_prompts_response)
        mock_message = MagicMock()
        mock_message.role = "user"
        mock_content = MagicMock()
        mock_content.text = "prompt text"
        mock_message.content = [mock_content]
        mock_result = MagicMock()
        mock_result.messages = [mock_message]
        mock_session.get_prompt = AsyncMock(return_value=mock_result)
        mcp_servers.sessions["test_server"] = mock_session
        mcp_servers.initialized = True

        result = await mcp_servers.get_prompt("test_prompt", {})
        assert result == {"messages": [{"role": "user", "content": "prompt text"}]}

    def test_get_servers_info(self, mcp_servers):
        """Test get_servers_info method."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mcp_servers.tools["test_tool"] = ExecutableTool(tool=mock_tool, mcp_servers=mcp_servers)
        info = mcp_servers.get_servers_info()
        assert len(info) == 1

    def test_get_resources(self, mcp_servers):
        """Test get_resources method."""
        mock_resource = MagicMock()
        mcp_servers.resources["test://resource"] = mock_resource
        resources = mcp_servers.get_resources()
        assert len(resources) == 1

    def test_get_prompts(self, mcp_servers):
        """Test get_prompts method."""
        mock_prompt = MagicMock()
        mcp_servers.prompts["test_prompt"] = mock_prompt
        prompts = mcp_servers.get_prompts()
        assert len(prompts) == 1

    @pytest.mark.asyncio
    async def test_get_tools(self, mcp_servers):
        """Test get_tools method."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mcp_servers.tools["test_tool"] = ExecutableTool(tool=mock_tool, mcp_servers=mcp_servers)
        mcp_servers.initialized = True
        tools = await mcp_servers.get_tools()
        assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_get_tools_with_names(self, mcp_servers):
        """Test get_tools with specific tool names."""
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mcp_servers.tools["tool1"] = ExecutableTool(tool=mock_tool1, mcp_servers=mcp_servers)
        mcp_servers.tools["tool2"] = ExecutableTool(tool=mock_tool2, mcp_servers=mcp_servers)
        mcp_servers.initialized = True
        tools = await mcp_servers.get_tools(["tool1"])
        assert len(tools) == 1
        assert tools[0].name == "tool1"

    def test_to_openai_format(self, mcp_servers):
        """Test to_openai_format method."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test description"
        mock_tool.inputSchema = {"type": "object"}
        mcp_servers.tools["test_tool"] = ExecutableTool(tool=mock_tool, mcp_servers=mcp_servers)
        mcp_servers.initialized = True
        openai_tools = mcp_servers.to_openai_format()
        assert len(openai_tools) == 1
        assert openai_tools[0]["function"]["name"] == "test_tool"

    def test_to_openai_format_not_initialized(self, mcp_servers):
        """Test to_openai_format when not initialized."""
        openai_tools = mcp_servers.to_openai_format()
        assert openai_tools == []

    def test_format_tools_for_prompt(self, mcp_servers):
        """Test format_tools_for_prompt method."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "Test description"
        mock_tool.inputSchema = {"properties": {"param1": {}, "param2": {}}}
        mcp_servers.tools["test_tool"] = ExecutableTool(tool=mock_tool, mcp_servers=mcp_servers)
        mcp_servers.initialized = True
        formatted = mcp_servers.format_tools_for_prompt()
        assert "test_tool" in formatted
        assert "Test description" in formatted

    def test_format_tools_for_prompt_not_initialized(self, mcp_servers):
        """Test format_tools_for_prompt when not initialized."""
        formatted = mcp_servers.format_tools_for_prompt()
        assert "MCP tools will be available after initialization" in formatted

    def test_format_tools_for_prompt_no_tools(self, mcp_servers):
        """Test format_tools_for_prompt when no tools."""
        mcp_servers.initialized = True
        formatted = mcp_servers.format_tools_for_prompt()
        assert "No MCP tools available" in formatted

    @pytest.mark.asyncio
    async def test_disconnect(self, mcp_servers):
        """Test disconnect method."""
        mock_session = MagicMock()
        mock_session.close = AsyncMock()
        mcp_servers.sessions["test_server"] = mock_session
        mcp_servers.initialized = True
        mcp_servers.exit_stack.aclose = AsyncMock()

        await mcp_servers.disconnect()
        assert mcp_servers.initialized is False
        assert len(mcp_servers.sessions) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self, mcp_servers):
        """Test async context manager."""
        with patch.object(mcp_servers, "initialize", new_callable=AsyncMock) as mock_init:
            async with mcp_servers:
                mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_from_config_file_json(self, tmp_path):
        """Test from_config_file with JSON file."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"server1": {"transport": "stdio", "command": "python"}}')
        servers = MCPServers.from_config_file(str(config_file))
        assert "server1" in servers.servers

    @pytest.mark.asyncio
    async def test_from_config_file_yaml(self, tmp_path):
        """Test from_config_file with YAML file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("server1:\n  transport: stdio\n  command: python\n")
        # yaml is imported inside the method, so we need to patch it at import time
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *args, **kwargs: __import__(name, *args, **kwargs)
            if name != "yaml"
            else MagicMock(safe_load=lambda f: {"server1": {"transport": "stdio", "command": "python"}}),
        ):
            # Actually, let's just test that it works if yaml is available
            try:
                import yaml  # noqa: F401

                servers = MCPServers.from_config_file(str(config_file))
                assert "server1" in servers.servers
            except ImportError:
                # If yaml is not available, test that it raises ImportError
                with pytest.raises(ImportError, match="PyYAML"):
                    MCPServers.from_config_file(str(config_file))
