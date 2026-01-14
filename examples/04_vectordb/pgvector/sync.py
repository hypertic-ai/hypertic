from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.embedders import OpenAIEmbedder
from hypertic.models import OpenAIChat
from hypertic.vectordb import PgVectorDB

load_dotenv()


# Create an embedder
embedder = OpenAIEmbedder(model="text-embedding-3-small")
vector_db = PgVectorDB(collection="demo", embedder=embedder, db_url="postgresql://user:password@localhost:5432/vectordb")

# vector_db.add(
#     files=["data/sample.txt"],
#     metadatas=[{"source": "sample.txt", "type": "document"}]
# )

text = "My secret is I am the best AI Engineer in the world."
vector_db.add(texts=[text], metadatas=[{"source": "sample.txt", "type": "document"}])

retriever = vector_db.as_retriever(k=3)

rag_agent = Agent(model=OpenAIChat(model="gpt-5.2"), retriever=retriever)

response = rag_agent.run("What is my secret?")
print(response.content)
