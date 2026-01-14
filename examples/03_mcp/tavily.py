import asyncio

from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.models import Anthropic
from hypertic.tools import MCPServers

load_dotenv()


async def main():
    # MCP configuration
    config = {
        "tavily-remote": {"url": "https://mcp.tavily.com/mcp/?tavilyApiKey=", "transport": "streamable_http"},
    }

    # Create MCP client
    mcp_tools = await MCPServers(config).get_tools(["web_search_tavily"])
    tools = [mcp_tools]
    print(f"Available Tools: {tools}")

    agent = Agent(
        model=Anthropic(model="claude-haiku-4-5"),
        instructions="",
        tools=tools,
    )

    print("\nStreaming response:")
    async for event in agent.astream("find me good restaurants in san francisco"):
        if event.type == "content":
            print(event.content, end="", flush=True)
        elif event.type == "tool_calls":
            print(f"\nTool Calls: {event.tool_calls}")
        elif event.type == "tool_outputs":
            print(f"\nTool Outputs: {event.tool_outputs}")
        elif event.type == "metadata":
            print(f"\nMetadata: {event.metadata}")


if __name__ == "__main__":
    asyncio.run(main())
