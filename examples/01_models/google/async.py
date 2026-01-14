import asyncio
from datetime import datetime

from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.models import GoogleAI
from hypertic.tools import tool

load_dotenv()


@tool
def today_date() -> str:
    """Returns the today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


tools = [today_date]
print(f"Available Tools: {tools}")

# models: gemini-3-pro-preview, gemini-2.5-pro, gemini-2.0-flash, etc.
agent = Agent(
    model=GoogleAI(model="gemini-3-pro-preview"),
    instructions="",
    tools=tools,
    # parallel_calls=False,
)


async def main():
    print("\nNon-streaming response:")
    response = await agent.arun("what is today's date?")
    print(response)
    # print(f"Final response: {response.content}")
    # print(f"Final tool calls: {response.tool_calls}")
    # print(f"Final tool outputs: {response.tool_outputs}")
    # print(f"Final metadata: {response.metadata}")


if __name__ == "__main__":
    asyncio.run(main())
