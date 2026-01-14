from datetime import datetime

from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.models import Cohere
from hypertic.tools import tool

load_dotenv()


@tool
def today_date() -> str:
    """Returns the today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


tools = [today_date]
print(f"Available Tools: {tools}")

# models: command-r7b-12-2024, command-a-03-2025, etc.
agent = Agent(
    model=Cohere(model="command-a-03-2025"),
    instructions="",
    tools=tools,
    # parallel_calls=False,
)

print("\nStreaming response:")
for event in agent.stream("what is today's date?"):
    if event.type == "content":
        print(event.content, end="", flush=True)
    elif event.type == "tool_calls":
        print(f"\nTool Calls: {event.tool_calls}")
    elif event.type == "tool_outputs":
        print(f"\nTool Outputs: {event.tool_outputs}")
    elif event.type == "metadata":
        print(f"\nMetadata: {event.metadata}")
