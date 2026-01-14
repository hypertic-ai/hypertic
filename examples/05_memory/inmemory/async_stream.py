import asyncio

from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.memory import InMemory
from hypertic.models import OpenAIChat

load_dotenv()


async def main():
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
    response1 = await agent.arun("My favorite color is blue.", session_id=session_id, user_id=user_id)
    print(f"Response: {response1.content}\n")

    print("\nSecond message (with memory, streaming):")
    async for event in agent.astream("What is my favorite color?", session_id=session_id, user_id=user_id):
        if event.type == "content":
            print(event.content, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
