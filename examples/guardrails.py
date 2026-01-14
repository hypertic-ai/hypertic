from dotenv import load_dotenv

from hypertic.agents import Agent
from hypertic.guardrails import Guardrail
from hypertic.models import XAI

load_dotenv()

guardrail = Guardrail(
    block_toxic=True,
    email="redact",
    credit_card="mask",
)

query = "My email is john.doe@example.com and card is 5105-1051-0510-5100."
result = guardrail.validate_input(query)
print(result)

agent = Agent(
    model=XAI(model="grok-4-1-fast-non-reasoning"),
    guardrails=[guardrail],
)

response = agent.run(query)
print(response.content)
