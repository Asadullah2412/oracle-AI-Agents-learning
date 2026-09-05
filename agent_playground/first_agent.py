import os

from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

# loads local variables
load_dotenv() 

# step1 : brain (LLM)


model = ChatOpenRouter(
    model= 'liquid/lfm-2.5-2.6b:free',
)

# step 2 Define your tools 

from langchain_core.tools import tool
import math


@tool
def add(a:float,b:float):
    """
    Add two numbers together,
    The agent will use this when it detectes an additon problem.
    """
    return a+b

@tool
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers together.
    Used for multiplication tasks.
    """
    return a * b


@tool
def divide(a: float, b: float) -> str:
    """
    Divide the first number by the second.
    Includes error handling for division by zero.
    """
    if b == 0:
        return "Error: Cannot divide by zero"
    return str(a / b)


@tool
def square_root(number: float) -> str:
    """
    Calculate the square root of a number.
    Includes error handling for negative inputs.
    """
    if number < 0:
        return "Error: Cannot take square root of a negative number"
    return str(math.sqrt(number))


# combining all the tools 

tools = [add,multiply,divide,square_root]

print("=== Available Tools ===")
for t in tools:
    print(f"  • {t.name}: {t.description}")
print()


# step:3  create the Agent

from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools =tools
)


# step:4 run the Agent

def run_agent(question:str):
    """Run the agent and print a clean , begineer-friendly execution trace."""

    print(f"\n🧑 User: {question}")
    print("-" * 60)

    result = agent.invoke({
        "messages": [("user", question)]
    })

    print("🔎 Clean Agent Execution Trace")
    print("-" * 60)

    step = 1
    for msg in result["messages"]:

        # 1. Human message = original user question
        if msg.type == "human":
            print(f"{step}. User asked:")
            print(f"   {msg.content}")
            step += 1

        # 2. AI message with tool_calls = agent decided to use a tool
        elif msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tool_call in msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"{step}. Agent decision:")
                print(f"   I need to use the tool: {tool_name}")
                print(f"   Tool input: {tool_args}")
                step += 1

        # 3. Tool message = result returned by the tool
        elif msg.type == "tool":
            print(f"{step}. Tool observation:")
            print(f"   Tool returned: {msg.content}")
            step += 1

        # 4. Final AI message = final response to user
        elif msg.type == "ai" and msg.content:
            print(f"{step}. Final answer:")
            print(f"   {msg.content}")
            step += 1

    print("=" * 60)

