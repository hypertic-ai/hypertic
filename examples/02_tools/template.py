import os

from dotenv import load_dotenv
from tavily import TavilyClient

from hypertic.agents import Agent
from hypertic.models import OpenAIChat
from hypertic.tools import BaseToolkit, tool

load_dotenv()


class TavilyTools(BaseToolkit):
    max_results: int = 5

    @tool
    def search(self, query: str) -> str:
        """Search the web using Tavily API.
        Args:
            query: The query to search for.
        Returns:
            JSON string containing search results with citations.
        """
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query=query, max_results=self.max_results)
        return str(response)


tavily = TavilyTools(max_results=3)
tools = [tavily]
print(f"Tools: {tools}")

agent = Agent(
    model=OpenAIChat(model="gpt-5.2"),
    tools=tools,
    instructions="",
    # parallel_calls=False,
)

response = agent.run("Search for latest AI news")
print(response.content)
