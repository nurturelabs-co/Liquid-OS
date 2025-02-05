<div align="center">
  <a href="https://hyperlauncher.ai/liquid-os">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://res.cloudinary.com/dd9z3q1v8/image/upload/w_100/v1738754167/g8aphb7uohfjoks01yu0.svg">
      <img src="https://res.cloudinary.com/dd9z3q1v8/image/upload/w_100/v1738754167/g8aphb7uohfjoks01yu0.svg" alt="LiquidOS">
    </picture>
  </a>
</div>
<div align="center">
  <em>Agent Framework / shim to use Pydantic with LLMs</em>
</div>

---

**Documentation**: [hyperlauncher.ai/liquid-os](https://hyperlauncher.ai/liquid-os) Coming Soon!

---

LiquidOS is a Python agent framework designed to make it less painful to build production grade applications with Generative AI.

## Why use LiquidOS

* __Model-agnostic__
Supports OpenAI, Anthropic, Gemini, Deepseek, Ollama, Groq, Cohere, and Mistral, and there is a simple interface to implement support for [other models](https://hyperlauncher.ai/liquid-osmodels/).

* __Pydantic Logfire Integration__
Seamlessly [integrates](https://hyperlauncher.ai/liquid-oslogfire/) with [Pydantic Logfire](https://pydantic.dev/logfire) for real-time debugging, performance monitoring, and behavior tracking of your LLM-powered applications.

* __Type-safe__
Designed to make [type checking](https://hyperlauncher.ai/liquid-osagents/#static-type-checking) as powerful and informative as possible for you.

* __Python-centric Design__
Leverages Python's familiar control flow and agent composition to build your AI-driven projects, making it easy to apply standard Python best practices you'd use in any other (non-AI) project.

* __Structured Responses__
Harnesses the power of [Pydantic](https://docs.pydantic.dev/latest/) to [validate and structure](https://hyperlauncher.ai/liquid-osresults/#structured-result-validation) model outputs, ensuring responses are consistent across runs.

* __Dependency Injection System__
Offers an optional [dependency injection](https://hyperlauncher.ai/liquid-osdependencies/) system to provide data and services to your agent's [system prompts](https://hyperlauncher.ai/liquid-osagents/#system-prompts), [tools](https://hyperlauncher.ai/liquid-ostools/) and [result validators](https://hyperlauncher.ai/liquid-osresults/#result-validators-functions).
This is useful for testing and eval-driven iterative development.

* __Streamed Responses__
Provides the ability to [stream](https://hyperlauncher.ai/liquid-osresults/#streamed-results) LLM outputs continuously, with immediate validation, ensuring rapid and accurate results.

* __Graph Support__
[Pydantic Graph](https://hyperlauncher.ai/liquid-osgraph) provides a powerful way to define graphs using typing hints, this is useful in complex applications where standard control flow can degrade to spaghetti code.

## In Beta!

PydanticAI is in early beta, the API is still subject to change and there's a lot more to do.
[Feedback](https://github.com/pydantic/pydantic-ai/issues) is very welcome!

## Hello World Example

Here's a minimal example of PydanticAI:

```python
from liquid_os import Agent

# Define a very simple agent including the model to use, you can also set the model when running the agent.
agent = Agent(
    'google-gla:gemini-1.5-flash',
    # Register a static system prompt using a keyword argument to the agent.
    # For more complex dynamically-generated system prompts, see the example below.
    system_prompt='Be concise, reply with one sentence.',
)

# Run the agent synchronously, conducting a conversation with the LLM.
# Here the exchange should be very short: PydanticAI will send the system prompt and the user query to the LLM,
# the model will return a text response. See below for a more complex run.
result = agent.run_sync('Where does "hello world" come from?')
print(result.data)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

_(This example is complete, it can be run "as is")_

Not very interesting yet, but we can easily add "tools", dynamic system prompts, and structured responses to build more powerful agents.

## Tools & Dependency Injection Example

Here is a concise example using PydanticAI to build a support agent for a bank:

**(Better documented example [in the docs](https://hyperlauncher.ai/liquid-os#tools-dependency-injection-example))**

```python
from dataclasses import dataclass

from pydantic import BaseModel, Field
from liquid_os import Agent, RunContext

from bank_database import DatabaseConn


# SupportDependencies is used to pass data, connections, and logic into the model that will be needed when running
# system prompt and tool functions. Dependency injection provides a type-safe way to customise the behavior of your agents.
@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


# This pydantic model defines the structure of the result returned by the agent.
class SupportResult(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


# This agent will act as first-tier support in a bank.
# Agents are generic in the type of dependencies they accept and the type of result they return.
# In this case, the support agent has type `Agent[SupportDependencies, SupportResult]`.
support_agent = Agent(
    'openai:gpt-4o',
    deps_type=SupportDependencies,
    # The response from the agent will, be guaranteed to be a SupportResult,
    # if validation fails the agent is prompted to try again.
    result_type=SupportResult,
    system_prompt=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)


# Dynamic system prompts can make use of dependency injection.
# Dependencies are carried via the `RunContext` argument, which is parameterized with the `deps_type` from above.
# If the type annotation here is wrong, static type checkers will catch it.
@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


# `tool` let you register functions which the LLM may call while responding to a user.
# Again, dependencies are carried via `RunContext`, any other arguments become the tool schema passed to the LLM.
# Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.
@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""
    # The docstring of a tool is also passed to the LLM as the description of the tool.
    # Parameter descriptions are extracted from the docstring and added to the parameter schema sent to the LLM.
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return balance


...  # In a real use case, you'd add more tools and a longer system prompt


async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    # Run the agent asynchronously, conducting a conversation with the LLM until a final response is reached.
    # Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve a result.
    result = await support_agent.run('What is my balance?', deps=deps)
    # The result will be validated with Pydantic to guarantee it is a `SupportResult`, since the agent is generic,
    # it'll also be typed as a `SupportResult` to aid with static type checking.
    print(result.data)
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = await support_agent.run('I just lost my card!', deps=deps)
    print(result.data)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """
```
