import os
import time
import httpx
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)

# --- Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
SYSTEM_PROMPT = (
    "You are a movie recommendation assistant. You will be given a genre and must "
    "suggest one single movie that fits that genre. You must respond with only "
    "the movie title on the first line, followed by a one-sentence summary on the "
    "second line. Do not add any conversational text like 'Here is a movie...' "
    "or 'I recommend...'."
)

# --- Routes ---

@app.route('/')
def index():
    """Serves the main index.html file."""
    # Serves files from the same directory as the script.
    return send_from_directory('.', 'index.html')

@app.route('/api/get-movie', methods=['POST'])
def get_movie_suggestion():
    """
    API endpoint to get a movie suggestion.
    The browser sends a POST request here with a JSON body like: {"genre": "Sci-Fi"}
    """
    if not API_KEY:
        return jsonify({"error": "GOOGLE_API_KEY not configured on the server."}), 500

    # 1. Get genre from the incoming JSON request
    data = request.json
    genre = data.get('genre')

    if not genre:
        return jsonify({"error": "Genre is required."}), 400

    # 2. Construct the payload for the Gemini API
    user_query = f"Suggest one popular movie for the genre \"{genre}\"."
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "tools": [{"google_search": {}}]
    }

    # 3. Call the Gemini API with retry logic
    response_data, error, status_code = call_gemini_with_backoff(payload)

    if error:
        return jsonify({"error": error}), status_code

    # 4. Parse the response from Gemini
    try:
        candidate = response_data.get('candidates', [{}])[0]
        text = candidate.get('content', {}).get('parts', [{}])[0].get('text')

        if not text:
            return jsonify({"error": "Invalid response structure from API."}), 500

        # Split into title and summary
        parts = text.split('\n', 1)
        title = parts[0]
        summary = parts[1] if len(parts) > 1 else "No summary available."

        # Get sources
        grounding_metadata = candidate.get('groundingMetadata', {})
        attributions = grounding_metadata.get('groundingAttributions', [])
        sources = [
            {"uri": attr['web']['uri'], "title": attr['web']['title']}
            for attr in attributions if attr.get('web', {}).get('uri')
        ]

        # 5. Send the final, clean data back to the front-end
        return jsonify({
            "title": title,
            "summary": summary,
            "sources": sources
        })

    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return jsonify({"error": "Failed to parse API response."}), 500

def call_gemini_with_backoff(payload, retries=3, delay=1):
    """
    Calls the Gemini API with exponential backoff for rate limiting (429)
    and network errors.
    """
    headers = {'Content-Type': 'application/json'}
    url = f"{GEMINI_API_URL}?key={API_KEY}"

    with httpx.Client(timeout=30.0) as client:
        for i in range(retries):
            try:
                response = client.post(url, headers=headers, json=payload)

                # Rate limited?
                if response.status_code == 429 and i < retries - 1:
                    time.sleep(delay * (2**i)) # Exponential backoff
                    continue

                # Other non-OK status
                if not response.is_success:
                    return None, f"API error: {response.status_code} {response.text}", response.status_code

                # Success
                return response.json(), None, 200

            except httpx.RequestError as e:
                # Network-level error
                if i < retries - 1:
                    time.sleep(delay * (2**i))
                    continue
                else:
                    return None, f"Network error: {e}", 500

    return None, "Max retries exceeded.", 500

# --- Error Handling ---
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not Found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error"}), 500

# --- Run the App ---
if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:5000
    app.run(debug=True, port=5000)
