# app_chat.py
from flask import Blueprint, render_template, request, jsonify
import json # For potential future use with more complex AI responses

# Import AI configuration from the main config.py
# This assumes config.py is in the same directory as app_chat.py or accessible via Python path.
from config import (
    OLLAMA_SERVER_URL, OLLAMA_MODEL_NAME,
    GEMINI_MODEL_NAME, # GEMINI_API_KEY will be passed from client for this endpoint
    AI_MODEL_PROVIDER
)
from ai import get_gemini_playlist_name # Import the function to call Gemini

# Create a Blueprint for chat-related routes
chat_bp = Blueprint('chat_bp', __name__,
                    template_folder='templates', # Specifies where to look for templates like chat.html
                    static_folder='static') # For chat-specific static files, if any

@chat_bp.route('/')
def chat_home():
    """
    Serves the main chat page.
    """
    return render_template('chat.html')

@chat_bp.route('/api/chatPlaylist', methods=['POST'])
def chat_playlist_api():
    """
    API endpoint to handle chat input and (mock) AI interaction.
    This is a synchronous endpoint.
    """
    data = request.get_json()
    if not data or 'userInput' not in data:
        return jsonify({"error": "Missing userInput in request"}), 400

    user_input = data.get('userInput')
    # Use AI provider from request, or fallback to global config, then to "NONE"
    ai_provider = data.get('ai_provider', AI_MODEL_PROVIDER).upper() # Use the imported constant
    ai_model_from_request = data.get('ai_model') # Model selected by user on chat page

    ai_response_message = f"Received your request: '{user_input}'.\n"
    actual_model_used = None

    # Define the prompt structure once, to be used by any provider that needs it.
    # The [USER INPUT] placeholder will be replaced dynamically.
    base_expert_playlist_creator_prompt = """
    You are an expert PostgreSQL query writer. Your sole purpose is to convert a user's natural language request for a music playlist into a valid PostgreSQL query for the table defined below.
    
    You MUST adhere to the following rules:
    1.  ONLY return the raw SQL query.
    2.  Do NOT add any explanations, comments, or markdown formatting (like ```sql). Your entire response must be ONLY the query itself.
    3.  The query should select the `title` and `author` columns.
    4.  Always add `LIMIT 25` to the end of the query to avoid overly long playlists.
    5.  Use `ILIKE` for case-insensitive matching on text fields when appropriate.
    ---
    
    ### **DATABASE SCHEMA:**
    
    ```sql
    CREATE TABLE IF NOT EXISTS public.score
    (
        item_id text NOT NULL,
        title text,
        author text,
        tempo real,
        key text,
        scale text,
        mood_vector text, -- Example: "jazz:0.278,blues:0.117,Hip-Hop:0.108"
        other_features text, -- Example: "danceable:0.53,party:0.88,relaxed:0.20"
        energy real,
        CONSTRAINT score_pkey PRIMARY KEY (item_id)
    );
    ```
    ---
    
    ### **QUERYING INSTRUCTIONS:**
    
    * **`mood_vector` and `other_features`**: These are TEXT fields containing comma-separated key:value pairs. To check if a mood or feature is present, you MUST use the `ILIKE` operator to find the key. For example, to find rock songs, use `WHERE mood_vector ILIKE '%rock:%'`. To find danceable songs, use `WHERE other_features ILIKE '%danceable:%'`.
    * **Tempo**: Use the `tempo` (real) column. Interpret user requests like this:
        * "Slow": `tempo < 100`
        * "Medium" or "Mid-tempo": `tempo BETWEEN 100 AND 140`
        * "Fast" or "Uptempo": `tempo > 140`
        * "Very Fast": `tempo > 160`
    * **Energy**: Use the `energy` (real) column (values 0.0 to 0.15).
        * "Low energy" or "Calm": `energy < 0.05`
        * "Moderate energy": `energy BETWEEN 0.05 AND 0.10`
        * "High energy": `energy > 0.10`
    * **Artist/Author**: Use the `author` column with an exact match: `WHERE author = 'Artist Name'`.
    * **Combining filters**: Your approach to combining filters is crucial.
        * Use **`AND`** when the user combines **different types of criteria**. For example, a mood AND a tempo (`mood_vector ILIKE '%rock:%' AND tempo > 140`).
        * Use **`OR`** when the user lists **multiple similar items**, like several moods or genres. For example, for "rock or pop music," you would use `(mood_vector ILIKE '%rock:%' OR mood_vector ILIKE '%pop:%')`.
        * When you have a mix, **group the `OR` conditions in parentheses**. For example, "fast rock or metal" should be `WHERE tempo > 140 AND (mood_vector ILIKE '%rock:%' OR mood_vector ILIKE '%metal:%')`.
    * For shuffling, you can add `ORDER BY random()`.
    * **EXAMPLE OF ROW IN THE TABLE:** `"b97e12654599c24d6dcf45042a439467" "Strictly for the Tardcore (skit)" "Bloodhound Gang" 156.60513 "C" "minor" "jazz:0.278,blues:0.117,Hip-Hop:0.108,rock:0.098,experimental:0.086" "danceable:0.53,aggressive:0.06,happy:0.09,party:0.88,relaxed:0.20,sad:0.41" 0.00963396`
    
    ---
    
    ### **EXAMPLES:**
    
    **User Request:** "I want some fast rock music"
    **Your Response:**
    `SELECT title, author FROM public.score WHERE tempo > 140 AND mood_vector ILIKE '%rock:%' ORDER BY random() LIMIT 25;`
    
    **User Request:** "A playlist of jazz or blues"
    **Your Response:**
    `SELECT title, author FROM public.score WHERE mood_vector ILIKE '%jazz:%' OR mood_vector ILIKE '%blues:%' ORDER BY random() LIMIT 25;`
    
    **User Request:** "I need something for a party that is high energy and either electronic or dance"
    **Your Response:**
    `SELECT title, author FROM public.score WHERE other_features ILIKE '%party:%' AND energy > 0.7 AND (mood_vector ILIKE '%electronic:%' OR mood_vector ILIKE '%dance:%') ORDER BY random() LIMIT 25;`
    
    **User Request:** "Just songs by Daft Punk"
    **Your Response:**
    `SELECT title, author FROM public.score WHERE author = 'Daft Punk' LIMIT 25;`
    
    ---
    
    Now, based on all the above, convert the following user request into a single PostgreSQL query.
    
    **User Request:** "{user_input_placeholder}"
    """

    # --- OLLAMA ---
    if ai_provider == "OLLAMA":
        actual_model_used = ai_model_from_request or OLLAMA_MODEL_NAME
        ai_response_message += f"Processing with OLLAMA model: {actual_model_used} (at {OLLAMA_SERVER_URL}).\n"
        # TODO: Implement actual Ollama call here using the prompt
        # full_prompt_for_ollama = base_expert_playlist_creator_prompt.replace("{user_input_placeholder}", user_input)
        # Example (requires ollama library and running server):
        # try:
        #   import ollama
        #   client = ollama.Client(host=OLLAMA_SERVER_URL)
        #   response = client.chat(model=actual_model_used, messages=[{'role': 'user', 'content': user_input}])
        #   ai_response_message += f"Ollama actual response: {response['message']['content']}"
        # except Exception as e:
        #   ai_response_message += f"Mock Response: Error connecting/querying Ollama: {str(e)}"
        ai_response_message += "(This is a mock response. Actual Ollama call not implemented in this version.)"

    # --- GEMINI ---
    elif ai_provider == "GEMINI":
        actual_model_used = ai_model_from_request or GEMINI_MODEL_NAME
        gemini_api_key_from_request = data.get('gemini_api_key')

        if not gemini_api_key_from_request:
            ai_response_message += "Error: Gemini API key was not provided in the request.\n"
            return jsonify({"response": {"message": ai_response_message, "original_request": user_input, "ai_provider_used": ai_provider, "ai_model_selected": actual_model_used}}), 400

        ai_response_message += f"Processing with GEMINI model: {actual_model_used} (using API key from request).\n"

        # Prepare the full prompt by inserting the user's input
        full_prompt_for_gemini = base_expert_playlist_creator_prompt.replace("{user_input_placeholder}", user_input)

        # Call Gemini (using the function from ai.py)
        # Use the API key provided in the request
        gemini_response = get_gemini_playlist_name(gemini_api_key_from_request, actual_model_used, full_prompt_for_gemini)
        ai_response_message += f"Gemini response: {gemini_response}"

    elif ai_provider == "NONE":
        ai_response_message += "No AI provider selected. Input acknowledged."
    else:
        ai_response_message += f"AI Provider '{ai_provider}' is not recognized."

    return jsonify({"response": {"message": ai_response_message, "original_request": user_input, "ai_provider_used": ai_provider, "ai_model_selected": actual_model_used}}), 200