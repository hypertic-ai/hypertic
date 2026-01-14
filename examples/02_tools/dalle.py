from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.models import OpenAIChat
from hypertic.tools import DalleTools

load_dotenv()

dalle = DalleTools(
    model="dall-e-3",
    size="1024x1024",
    quality="standard",
    style="vivid",
)

tools = [dalle]
print(f"Tools: {tools}")

agent = Agent(
    model=OpenAIChat(model="gpt-5.2"),
    instructions="",
    tools=tools,
    # parallel_calls=False,
)

response = agent.run("Generate a picture of a dog playing with a ball")
print(response)
# print(f"Final response: {response.content}")
# print(f"Final tool calls: {response.tool_calls}")
# print(f"Final tool outputs: {response.tool_outputs}")
# print(f"Final metadata: {response.metadata}")
