import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent
from langchain.tools import tool
from weather_func import get_weather
from logging_config import setup_logging

setup_logging()
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

weather_tool = tool(get_weather)

agent = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    agent="structured-chat-zero-shot-react-description",
)

response = agent.invoke("What is the weather like in New York and Jaipur today?")
print("Final Result:", response["output"])