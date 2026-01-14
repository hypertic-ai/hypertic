import asyncio

from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.models import OpenAIResponse

load_dotenv()


async def main():
    agent = Agent(
        model=OpenAIResponse(model="gpt-5.2", max_tokens=4096),
    )

    print("non streaming response:")
    response = await agent.arun(
        query="Describe this image?",
        files=[
            "https://yavuzceliker.github.io/sample-images/image-1021.jpg",
        ],
    )
    print(f"Response: {response.content}")
    print(f"Metadata: {response.metadata}")

    print("streaming response:")
    async for event in agent.astream(query="Short summary of this pdf?", files=["https://www.berkshirehathaway.com/letters/2024ltr.pdf"]):
        if event.type == "content":
            print(event.content, end="", flush=True)
        elif event.type == "metadata":
            print(f"Metadata: {event.metadata}")


if __name__ == "__main__":
    asyncio.run(main())
