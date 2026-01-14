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
