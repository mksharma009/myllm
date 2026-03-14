import os
from dotenv import load_dotenv
import google.generativeai as genai
from flask import Flask, jsonify, render_template, request


load_dotenv()


def create_app() -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)

    @app.route("/")
    def index_page():
        """Render the HTML form used to send requests to the Gemini backend."""

        return render_template("index.html")

    @app.route("/gemini")
    def fetch_gemini_response():
        """Fetch the Gemini response for a query and return it as JSON."""

        query = request.args.get("query", "").strip()
        if not query:
            return jsonify({"error": "Missing query"}), 400

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(query)
        return jsonify({"result": response.text})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
