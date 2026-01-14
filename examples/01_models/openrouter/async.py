import asyncio
from datetime import datetime

from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.models import OpenRouter
from hypertic.tools import tool

load_dotenv()


@tool
def today_date() -> str:
    """Returns the today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


tools = [today_date]
print(f"Available Tools: {tools}")

# models: mistralai/mistral-small-creative, nvidia/nemotron-3-nano-30b-a3b:free, xiaomi/mimo-v2-flash:free etc.
agent = Agent(
    model=OpenRouter(model="openai/gpt-4o"),
    instructions="",
    tools=tools,
    # parallel_calls=False,
)


async def main():
    print("\nNon-streaming response:")
    response = await agent.arun("what is today's date?")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
