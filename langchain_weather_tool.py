import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent
from langchain.tools import tool
from weather_func import get_weather
from logging_config import setup_logging
from flask import Flask, jsonify, render_template, request

setup_logging()
load_dotenv()

weather_tool = tool(get_weather)


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index_page():
        return render_template("index.html", title="Gemini Weather Tool")

    @app.route("/gemini", methods=["GET"])
    def fetch_result():
        query = request.args.get("query", "").strip()
        if not query:
            return jsonify({"error": "Missing query"}), 400

        llm = ChatGoogleGenerativeAI(
            model=os.getenv("MODEL_NAME"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        agent = initialize_agent(
            tools=[weather_tool],
            llm=llm,
            agent="structured-chat-zero-shot-react-description",
        )

        response = agent.invoke(query)
        return jsonify({"result": response["output"]})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
