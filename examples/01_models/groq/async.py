import asyncio
from datetime import datetime

from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.models import Groq
from hypertic.tools import tool

load_dotenv()


@tool
def today_date() -> str:
    """Returns the today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


tools = [today_date]
print(f"Available Tools: {tools}")

# models: meta-llama/llama-4-scout-17b-16e-instruct, moonshotai/kimi-k2-instruct-0905, llama-3.3-70b-versatile, etc.
agent = Agent(
    model=Groq(model="meta-llama/llama-4-maverick-17b-128e-instruct"),
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
