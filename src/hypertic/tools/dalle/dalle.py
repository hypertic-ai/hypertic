import json
from os import getenv
from typing import Literal, Optional

from pydantic import model_validator

from hypertic.tools.base import BaseToolkit, tool

try:
    from openai import OpenAI
except ImportError as err:
    raise ImportError("openai library not installed. Install with: pip install openai") from err


class DalleTools(BaseToolkit):
    api_key: Optional[str] = None
    model: str = "dall-e-3"
    size: Optional[Literal["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]] = "1024x1024"
    quality: Literal["standard", "hd"] = "standard"
    style: Literal["vivid", "natural"] = "vivid"
    n: int = 1

    @model_validator(mode="after")
    def _check_api_key(self):
        if not self.api_key:
            self.api_key = getenv("OPENAI_API_KEY")
        return self

    @tool
    def generate(self, query: str) -> str:
        """Generate an image from a text query using DALL-E.

        Args:
            query: The text description of the image to generate.

        Returns:
            JSON string containing the generated image URL(s) and metadata.
        """
        if not self.api_key:
            raise ValueError("API key is required. Set it when initializing DalleTools or via OPENAI_API_KEY environment variable.")

        if self.api_key is not None:
            client = OpenAI(api_key=self.api_key)
        else:
            client = OpenAI()

        try:
            n_images = 1 if self.model == "dall-e-3" else min(self.n, 10)

            if self.model == "dall-e-3":
                response = client.images.generate(
                    model=self.model,
                    prompt=query,
                    size=self.size,
                    quality=self.quality,
                    style=self.style,
                    n=1,
                )
            else:
                response = client.images.generate(
                    model=self.model,
                    prompt=query,
                    size=self.size,
                    n=n_images,
                )

            images = []
            if response.data:
                for image in response.data:
                    images.append(
                        {
                            "url": image.url,
                            "revised_prompt": getattr(image, "revised_prompt", None),
                        }
                    )

            result = {
                "model": self.model,
                "query": query,
                "images": images,
                "count": len(images),
            }

            return json.dumps(result, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to generate image: {e!s}") from e
