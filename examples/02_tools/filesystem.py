from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.models import OpenAIChat
from hypertic.tools import FileSystemTools

load_dotenv()

filesystem_tools = FileSystemTools(root_dir="./data")
tools = [filesystem_tools]
print(f"Tools: {tools}")

agent = Agent(
    model=OpenAIChat(model="gpt-5.2"),
    tools=tools,
    instructions="",
    # parallel_calls=False,
)

# Test 1: List Directory
result = agent.run("List all the contents")
print(f"List directory: {result.content}")

# # Test 2: Read File
# result = agent.run("Read the contents of sample.txt")
# print(f"Read file: {result.content}")

# # # Test 3: Write File
# result = agent.run("Create a new file called 'test_output.txt' with the content 'This is a test file created by FileSystem tools'")
# print(f"Write file: {result.content}")

# # # Test 4: Copy File
# result = agent.run("Copy sample.txt to data/sample_copy.txt")
# print(f"Copy file: {result.content}")

# # # Test 5: Move File
# result = agent.run("Move test_output.txt to moved_output.txt")
# print(f"Move file: {result.content}")

# # # Test 6: Search Files
# result = agent.run("Search for all .txt files in the current directory")
# print(f"Search files: {result.content}")

# # # Test 7: Delete File
# result = agent.run("Delete the file moved_output.txt")
# print(f"Delete file: {result.content}")
