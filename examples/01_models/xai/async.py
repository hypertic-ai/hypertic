import asyncio
from datetime import datetime

from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.models import XAI
from hypertic.tools import tool

load_dotenv()


@tool
def today_date() -> str:
    """Returns the today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


tools = [today_date]
print(f"Available Tools: {tools}")

# models: grok-3, grok-3-mini, grok-4-fast-non-reasoning, etc.
agent = Agent(
    model=XAI(model="grok-4-fast-non-reasoning"),
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
