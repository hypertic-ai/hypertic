from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.embedders import OpenAIEmbedder
from hypertic.models import OpenAIChat
from hypertic.vectordb import MongoDBAtlas

load_dotenv()


# Create an embedder
embedder = OpenAIEmbedder(model="text-embedding-3-small")
vector_db = MongoDBAtlas(collection="demo", embedder=embedder, connection_string="mongodb://localhost:27017")

# vector_db.add(
#     files=["data/sample.txt"],
#     metadatas=[{"source": "sample.txt", "type": "document"}]
# )

text = "My secret is I am the best AI Engineer in the world."
vector_db.add(texts=[text], metadatas=[{"source": "sample.txt", "type": "document"}])

retriever = vector_db.as_retriever(k=3)

rag_agent = Agent(model=OpenAIChat(model="gpt-5.2"), retriever=retriever)

for event in rag_agent.stream("What is my secret?"):
    if event.type == "content":
        print(event.content, end="", flush=True)
