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
    print("\nStreaming response:")
    async for event in agent.astream("what is today's date?"):
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
