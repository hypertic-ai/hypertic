from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.memory import InMemory
from hypertic.models import OpenAIChat

load_dotenv()

# InMemory - simple in-memory storage (data lost on restart)
memory = InMemory()

agent = Agent(
    model=OpenAIChat(model="gpt-5.2"),
    memory=memory,
    instructions="",
)

user_id = "user_123"
session_id = "session_001"

print("\nFirst message:")
response1 = agent.run("My favorite color is blue.", session_id=session_id, user_id=user_id)
print(f"Response: {response1.content}\n")

print("\nSecond message (with memory):")
response2 = agent.run("What is my favorite color?", session_id=session_id, user_id=user_id)
print(f"Response: {response2.content}\n")
