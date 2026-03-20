import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types
from flask import Flask, jsonify, render_template, request
from math_func import add, subtract, multiply, divide

logging.basicConfig(level=logging.INFO)

load_dotenv()

# Define the calculator tool using the correct Gemini API format
calculator_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="add",
            description="Add two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number"},
                    "b": {"type": "number", "description": "The second number"},
                },
                "required": ["a", "b"],
            },
        ),
        types.FunctionDeclaration(
            name="subtract",
            description="Subtract two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number"},
                    "b": {"type": "number", "description": "The second number"},
                },
                "required": ["a", "b"],
            },
        ),
        types.FunctionDeclaration(
            name="multiply",
            description="Multiply two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number"},
                    "b": {"type": "number", "description": "The second number"},
                },
                "required": ["a", "b"],
            },
        ),
        types.FunctionDeclaration(
            name="divide",
            description="Divide two numbers",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number"},
                    "b": {"type": "number", "description": "The second number"},
                },
                "required": ["a", "b"],
            },
        ),
    ],
)

tool_registry = {
    "calculator": {
        "tool": calculator_tool,
        "implementation": {
            "add": add,
            "subtract": subtract,
            "multiply": multiply,
            "divide": divide,
        },
    }
}


def create_app() -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)

    @app.route("/")
    def index_page():
        """Render the HTML form used to send requests to the Gemini backend."""

        return render_template("index.html", title="Gemini Calculator")

    @app.route("/gemini")
    def fetch_gemini_response():
        """Fetch the Gemini response for a query and return it as JSON."""

        query = request.args.get("query", "").strip()
        if not query:
            return jsonify({"error": "Missing query"}), 400

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel(model_name=os.getenv("MODEL_NAME"), tools=[calculator_tool])

        # Start conversation with the user's query
        chat = model.start_chat()
        response = chat.send_message(query)

        # Handle function calls in a loop
        while True:
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

                            if (
                                func_name
                                in tool_registry["calculator"]["implementation"]
                            ):
                                func = tool_registry["calculator"]["implementation"][
                                    func_name
                                ]
                                try:
                                    result = func(**func_args)
                                    function_result = (
                                        f"{func_name}({func_args}) = {result}"
                                    )
                                except Exception as e:
                                    function_result = (
                                        f"Error in {func_name}({func_args}): {str(e)}"
                                    )
                            else:
                                function_result = f"Unknown function: {func_name}"

            if function_result is not None:
                # Send function results back to the model
                result_message = f"Function call results:\n{function_result}"
                logging.info(f"Sending function result back to Gemini: {result_message}")
                response = chat.send_message(result_message)
            else:
                # No more function calls, return the final response
                break

        return jsonify({"result": response.text})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
