import asyncio
import os

from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.memory import AsyncPostgresServer
from hypertic.models import OpenAIChat

load_dotenv()


async def main():
    # PostgreSQL connection string
    db_url = os.getenv("DATABASE_URL")

    # AsyncPostgresServer - persistent storage in PostgreSQL
    async with AsyncPostgresServer.create(db_url=db_url, table_name="agent_memory") as memory:
        agent = Agent(
            model=OpenAIChat(model="gpt-5.2"),
            memory=memory,
            instructions="",
        )

        user_id = "user_123"
        session_id = "session_001"

        print("\nFirst message:")
        response1 = await agent.arun("My favorite color is blue.", session_id=session_id, user_id=user_id)
        print(f"Response: {response1.content}\n")

        print("\nSecond message (with memory):")
        response2 = await agent.arun("What is my favorite color?", session_id=session_id, user_id=user_id)
        print(f"Response: {response2.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
