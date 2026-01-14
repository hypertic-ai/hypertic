import asyncio

from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.embedders import OpenAIEmbedder
from hypertic.models import OpenAIChat
from hypertic.vectordb import ChromaDB

load_dotenv()


async def main():
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    vector_db = ChromaDB(collection="demo", embedder=embedder, path="./chromadb")

    # await vector_db.async_add(
    # files=["data/sample.txt"],
    # metadatas=[{"source": "sample.txt", "type": "document"}]
    # )

    text = "My secret is I am the best AI Engineer in the world."

    await vector_db.async_add(texts=[text], metadatas=[{"source": "sample.txt", "type": "document"}])

    retriever = vector_db.as_retriever(k=3)

    rag_agent = Agent(model=OpenAIChat(model="gpt-5.2"), retriever=retriever)

    response = await rag_agent.arun("What is my secret?")

    print(response.content)


asyncio.run(main())
