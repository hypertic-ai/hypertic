import asyncio
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel

from hypertic.agents import Agent
from hypertic.models import OpenAIResponse
from hypertic.tools import tool

load_dotenv()

# Note: Only Anthropic, Openai, XAI and Cohere supports tools + structured output.


class Date(BaseModel):
    day: int
    month: int
    year: int


@tool
def today_date() -> str:
    """Returns the today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


async def main():
    agent = Agent(model=OpenAIResponse(model="gpt-5.2"), tools=[today_date], output_type=Date)

    print("non streaming response:")
    response = await agent.arun("what is today's date?")
    print(response.structured_output)
    # print(response.structured_output.day)
    # print(response.structured_output.month)
    # print(response.structured_output.year)
    print(response.tool_calls)
    print(response.tool_outputs)
    print(response.metadata)

    print("\nstreaming response:")
    async for event in agent.astream("what is yesterday's date?"):
        if event.type == "structured_output":
            structured_output = event.structured_output
            print(f"\n{structured_output}")
            # print(structured_output.day)
            # print(structured_output.month)
            # print(structured_output.year)
        elif event.type == "tool_calls":
            print(f"\nTool Calls: {event.tool_calls}")
        elif event.type == "tool_outputs":
            print(f"\nTool Outputs: {event.tool_outputs}")
        elif event.type == "metadata":
            print(f"\nMetadata: {event.metadata}")


if __name__ == "__main__":
    asyncio.run(main())
