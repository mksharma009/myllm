import os
import google.generativeai as genai


def load_dotenv():
    """Load environment variables from a .env file."""
    # Implementation for loading environment variables
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value


load_dotenv()


def fetch_gemini_response(query):
    """Connect to Gemini API and fetch response data."""
    # Implementation for fetching Gemini response
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(query)
    return response.text


if __name__ == "__main__":
    while True:
        query = input("Enter your query for Gemini: ")
        gemini_response = fetch_gemini_response(query)
        print(gemini_response)
        if input("Do you want to ask another question? (yes/no): ").lower() != "yes":
            break