"""We will be creating tools using LangChain's tool system, which allows us to define functions that can be called by the language model. Below is an example of how to create a calculator tool that can perform basic arithmetic operations."""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent
from langchain.tools import tool
from math_func import add, subtract, multiply, divide

load_dotenv()

add_tool = tool(add)
subtract_tool = tool(subtract)
multiply_tool = tool(multiply)
divide_tool = tool(divide)

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# create an agent with the tools
calc_agent = initialize_agent(
    tools=[add_tool, subtract_tool, multiply_tool, divide_tool],
    llm=llm,
    agent="structured-chat-zero-shot-react-description",
)
result = calc_agent.invoke("What is 5 plus 3 multiplied by 2?")
print("Final Result:", result["output"])
