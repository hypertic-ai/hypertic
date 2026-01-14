<div align="center" style="margin: 0 auto; max-width: 80%;">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="./static/logowhite.png">
      <source media="(prefers-color-scheme: light)" srcset="./static/logoblack.png">
      <img 
        src="./static/logoblack.png" 
        style="width: 300px; height: auto; margin: 20px auto;"
      >
    </picture>
</div>


<div align="center">

[![PyPI](https://img.shields.io/pypi/v/hypertic?label=pypi%20package)](https://pypi.org/project/hypertic/)

</div>

Hypertic is the fastest and easiest way to build AI agent applications. It provides a clean, simple interface for connecting models, tools, vector databases, memory, and more.

### Key Features:

1. **[Tools](https://docs.hypertic.ai/tools)**: Create custom tools with Python functions or connect to MCP servers
2. **[Memory](https://docs.hypertic.ai/memory)**: Store conversation history with in-memory, PostgreSQL, MongoDB, or Redis backends
3. **[Retriever](https://docs.hypertic.ai/retriever)**: Connect agent to your documents and data for RAG capabilities
4. **[Structured Output](https://docs.hypertic.ai/structured-output)**: Get validated, structured responses using Pydantic models or schemas
5. **[Guardrails](https://docs.hypertic.ai/guardrails)**: Add safety and validation rules to control agent behavior

Check out the [examples](https://github.com/hypertic/hypertic/tree/main/examples) to see how Hypertic works, and visit our [documentation](https://docs.hypertic.ai) to learn more.

## Get Started

To get started, set up your Python environment (Python 3.10 or newer required), and then install the Hypertic package.

### venv

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install hypertic
```

For specific model providers, install the corresponding packages:

```bash
pip install openai          # For OpenAI
pip install anthropic       # For Anthropic
pip install google-genai    # For Google Gemini
```

### uv

If you're familiar with [uv](https://docs.astral.sh/uv/), installing the package would be even easier:

```bash
uv init
uv add hypertic
```

For specific model providers:

```bash
uv add openai          # For OpenAI
uv add anthropic       # For Anthropic
uv add google-genai    # For Google Gemini
```

### Quick Start

**Sync (non-streaming):**

Use `run()` for synchronous, non-streaming responses. This returns the complete response after the agent finishes processing:

```python
from hypertic import Agent, tool
from openai import OpenAI

# Define a tool
@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"Sunny, 72°F in {city}"

# Create agent
model = OpenAI(model="gpt-4")
agent = Agent(
    model=model,
    tools=[get_weather],
    instructions="You are a helpful assistant."
)

# Use it
response = agent.run("What's the weather in San Francisco?")
print(response.content)
```

**Sync (streaming):**

Use `stream()` for synchronous streaming. This yields events in real-time as the agent generates responses, improving user experience for longer outputs:

```python
from hypertic import Agent, tool
from openai import OpenAI

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"Sunny, 72°F in {city}"

model = OpenAI(model="gpt-4")
agent = Agent(
    model=model,
    tools=[get_weather],
    instructions="You are a helpful assistant."
)

# Stream responses in real-time
for event in agent.stream("What's the weather in San Francisco?"):
    if event.type == "content":
        print(event.content, end="", flush=True)
```

**Async (non-streaming):**

Use `arun()` for asynchronous, non-streaming responses. This is ideal for concurrent operations and non-blocking I/O:

```python
import asyncio
from hypertic import Agent, tool
from openai import OpenAI

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"Sunny, 72°F in {city}"

model = OpenAI(model="gpt-4")
agent = Agent(
    model=model,
    tools=[get_weather],
    instructions="You are a helpful assistant."
)

async def main():
    response = await agent.arun("What's the weather in San Francisco?")
    print(response.content)

asyncio.run(main())
```

**Async (streaming):**

Use `astream()` for asynchronous streaming. This combines the benefits of async operations with real-time response streaming:

```python
import asyncio
from hypertic import Agent, tool
from openai import OpenAI

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"Sunny, 72°F in {city}"

model = OpenAI(model="gpt-4")
agent = Agent(
    model=model,
    tools=[get_weather],
    instructions="You are a helpful assistant."
)

async def main():
    async for event in agent.astream("What's the weather in San Francisco?"):
        if event.type == "content":
            print(event.content, end="", flush=True)

asyncio.run(main())
```

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.


## License

This project is licensed under the [Apache License 2.0](LICENSE).