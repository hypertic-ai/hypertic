import asyncio

from dotenv import load_dotenv
from file_loader import FileLoaderTools

from hypertic.agents import Agent, RunOptions
from hypertic.models import OpenAIChat

load_dotenv()


async def main():
    agent = Agent(
        model=OpenAIChat(model="gpt-5.2"),
        tools=[FileLoaderTools(path="data/financial_data.json")],
        skills=["financial"],
    )

    async for event in agent.astream(
        "Calculate financial ratios from the file", options=RunOptions(enabled_skills=["analyzing-financial-statements"])
    ):
        if event.type == "content":
            print(event.content, end="", flush=True)
        elif event.type == "tool_calls":
            print(f"\n Tool Calls: {event.tool_calls}")


if __name__ == "__main__":
    asyncio.run(main())
