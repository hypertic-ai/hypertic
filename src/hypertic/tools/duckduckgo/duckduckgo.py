import json
from typing import Optional

from hypertic.tools.base import BaseToolkit, tool


class DuckDuckGoTools(BaseToolkit):
    modifier: Optional[str] = None
    max_results: int = 5
    proxy: Optional[str] = None
    timeout: Optional[int] = 10
    backend: str = "auto"
    verify_ssl: bool = True
    region: Optional[str] = "us-en"
    safesearch: str = "moderate"
    time: Optional[str] = "y"

    @tool
    def search(self, query: str) -> str:
        """Search DuckDuckGo for a query.

        Uses the max_results configured during initialization (default: 5).

        Args:
            query: The query to search for.

        Returns:
            JSON string containing search results.
        """
        try:
            from ddgs import DDGS
        except ImportError as err:
            raise ImportError("ddgs library not installed. Install with: pip install ddgs") from err

        search_query = f"{self.modifier} {query}" if self.modifier else query

        try:
            with DDGS(proxy=self.proxy, timeout=self.timeout, verify=self.verify_ssl) as ddgs:
                results = list(
                    ddgs.text(
                        query=search_query,
                        region=self.region,
                        safesearch=self.safesearch,
                        timelimit=self.time,
                        max_results=self.max_results,
                        backend=self.backend,
                    )
                )

            return json.dumps(results, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to search: {e!s}") from e

    @tool
    def news(self, query: str) -> str:
        """Get the latest news from DuckDuckGo.

        Uses the max_results configured during initialization (default: 5).

        Args:
            query: The query to search for.

        Returns:
            JSON string containing news results.
        """
        try:
            from ddgs import DDGS
        except ImportError as err:
            raise ImportError("ddgs library not installed. Install with: pip install ddgs") from err

        try:
            with DDGS(proxy=self.proxy, timeout=self.timeout, verify=self.verify_ssl) as ddgs:
                results = list(
                    ddgs.news(query=query, region=self.region, safesearch=self.safesearch, timelimit=self.time, max_results=self.max_results)
                )

            return json.dumps(results, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to search news: {e!s}") from e

    @tool
    def images(self, query: str) -> str:
        """Search for images using DuckDuckGo.

        Uses the max_results configured during initialization (default: 5).

        Args:
            query: The query to search for.

        Returns:
            JSON string containing image search results.
        """
        try:
            from ddgs import DDGS
        except ImportError as err:
            raise ImportError("ddgs library not installed. Install with: pip install ddgs") from err

        try:
            with DDGS(proxy=self.proxy, timeout=self.timeout, verify=self.verify_ssl) as ddgs:
                results = list(ddgs.images(query=query, region=self.region, safesearch=self.safesearch, max_results=self.max_results))

            return json.dumps(results, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to search images: {e!s}") from e
