import asyncio

from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.models import OpenAIChat
from hypertic.tools import MCPServers

load_dotenv()


async def main():
    # MCP configuration
    config = {"playwright": {"command": "npx", "args": ["@playwright/mcp@latest"], "env": {"DISPLAY": ":1"}}}

    # Create MCP client
    mcp_tools = await MCPServers(config).get_tools()
    tools = [mcp_tools]
    print(f"Available Tools: {tools}")

    model = OpenAIChat(
        model="gpt-4.1",
    )

    agent = Agent(
        model=model,
        instructions="",
        tools=tools,
    )

    # Test non-streaming
    print("\nNon-streaming response:")
    response = await agent.arun("find me good restaurants in san francisco")
    print(f"Final response: {response.content}")
    print(f"Final tool calls: {response.tool_calls}")
    print(f"Final tool outputs: {response.tool_outputs}")
    print(f"Final metadata: {response.metadata}")


if __name__ == "__main__":
    asyncio.run(main())
