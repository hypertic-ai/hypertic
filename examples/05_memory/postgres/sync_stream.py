import os

from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.memory import PostgresServer
from hypertic.models import OpenAIChat

load_dotenv()

# PostgreSQL connection string
DB_URL = os.getenv("DATABASE_URL")

# PostgresServer - persistent storage in PostgreSQL
memory = PostgresServer(db_url=DB_URL, table_name="agent_memory")

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

print("\nSecond message (with memory, streaming):")
for event in agent.stream("What is my favorite color?", session_id=session_id, user_id=user_id):
    if event.type == "content":
        print(event.content, end="", flush=True)
