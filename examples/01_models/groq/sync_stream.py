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
