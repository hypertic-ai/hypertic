from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.models import OpenAIChat
from hypertic.tools import DuckDuckGoTools

load_dotenv()

duckduckgo = DuckDuckGoTools(safesearch="off", region="us-en", time="d")
tools = [duckduckgo]
print(f"Tools: {tools}")

agent = Agent(
    model=OpenAIChat(model="gpt-5.2"),
    instructions="",
    tools=tools,
    # parallel_calls=False,
)

# Test: Text search (tools: search)
response = agent.run("Use search to search for 'Python programming language'")
print(f"Text search: {response.content}\n")

# Test: News search (tools: news)
response = agent.run("Use news to search for recent news about 'artificial intelligence'")
print(f"News search: {response.content}\n")

# Test: Image search (tools: images)
response = agent.run("Use images to search for images of 'cute puppies'")
print(f"Image search: {response.content}\n")
