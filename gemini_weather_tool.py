import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google.generativeai import types
from logging_config import setup_logging
from weather_func import get_weather

setup_logging()
load_dotenv()

# Constants
MAX_ITERATIONS = 5
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"


# Define the weather tool
weather_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get current weather information for a given location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location for which to get weather information (city, country, etc.)",
                    },
                },
                "required": ["location"],
            },
        ),
    ],
)

# Tool registry for easy extension
tool_registry = {
    "weather": {
        "tool": weather_tool,
        "implementation": {
            "get_weather": get_weather,
        },
    }
}


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        The configured Flask application instance.
    """
    app = Flask(__name__)

    @app.route("/")
    def index_page():
        """Render the HTML form for weather queries."""
        return render_template("index.html", title="Gemini Weather Tool")

    @app.route("/gemini")
    def fetch_gemini_response():
        """Fetch the Gemini response for a query and handle tool calls.

        Returns:
            JSON response with the result or error.
        """
        query = request.args.get("query", "").strip()
        if not query:
            return jsonify({"error": "Missing query"}), 400

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(
            model_name=os.getenv("MODEL_NAME"), tools=[weather_tool]
        )

        # Start conversation with the user's query
        chat = model.start_chat()
        response = chat.send_message(query)

        # Handle function calls in a loop with safety limit
        iteration = 0
        while iteration < MAX_ITERATIONS:
            iteration += 1
            function_result = None

            # Check for function calls in the response
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            function_call = part.function_call
                            func_name = function_call.name
                            func_args = function_call.args

                            if func_name in tool_registry["weather"]["implementation"]:
                                func = tool_registry["weather"]["implementation"][
                                    func_name
                                ]
                                try:
                                    result = func(**func_args)
                                    function_result = (
                                        f"{func_name}({func_args}) = {result}"
                                    )
                                    logging.info(f"Executed {func_name} successfully")
                                except Exception as e:
                                    error_msg = (
                                        f"Error in {func_name}({func_args}): {str(e)}"
                                    )
                                    function_result = error_msg
                                    logging.error(error_msg)
                            else:
                                function_result = f"Unknown function: {func_name}"
                                logging.warning(f"Unknown function called: {func_name}")

            if function_result is not None:
                # Send function result back to the model
                result_message = f"Function call results:\n{function_result}"
                logging.info(f"Sending result back to Gemini: {result_message}")
                response = chat.send_message(result_message)
            else:
                # No more function calls, return the final response
                break

        if iteration >= MAX_ITERATIONS:
            logging.warning("Reached maximum iterations in tool call loop")

        return jsonify({"result": response.text})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
