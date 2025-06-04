# /home/guido/Music/AudioMuse-AI/ai.py
import requests
import json
import re
import ftfy # Import the ftfy library
import time # Import the time library
import unicodedata
import google.generativeai as genai # Import Gemini library

GEMINI_API_CALL_DELAY_SECONDS = 7 # Seconds to wait before each Gemini API call
# Strengthened prompt template for strict output
# This template is designed to work with both Ollama and Gemini text generation models.
# Now includes good and bad examples, and explicit instruction for conciseness.
# Added length constraint and instruction to avoid special characters/fonts, allowing emojis.

creative_prompt_template = (
    "You are a highly creative music curator. Your SOLE task is to generate 1 concise (15-35 chars) playlist name.\n"
    "The name MUST be evocative, representative of the provided songs/features, and use real English words with ONLY standard ASCII (a-z, A-Z, 0-9, spaces, and - & ' ! . , ? ( ) [ ]).\n"
    "No special fonts or emojis.\n"
    "Input Features to consider: '{feature1}', '{feature2}', '{feature3}'. Analyze the provided song list for its core vibe.\n"
    "The playlist name should suggest an activity, mood, or context, similar to these:\n\n" # Emphasizes the style
    "* GOOD EXAMPLES: 'Sunshine Pop Vibrations' (Concept: cheerful, energetic pop for sunny days)\n"
    "* GOOD EXAMPLES: 'Workout Power Hour Mix' (Concept: energetic rock/dance for intense workouts)\n"
    "* GOOD EXAMPLES: 'Relaxing Evening Melodies' (Concept: calming songs for a peaceful evening)\n\n"
    "* BAD EXAMPLES: For features 'rock', 'energetic', '80s' -> 'Midnight Pop Fever' (Inconsistent with features)\n"
    "* BAD EXAMPLES: 'Ambient Electronic Space - Electric Soundscapes - Emotional Waves' (Too long/descriptive)\n"
    "* BAD EXAMPLES: 'Blues Rock Fast Tracks' (Too direct/literal, not evocative enough)\n"
    "* BAD EXAMPLES: '𝑯𝒘𝒆 𝒂𝒓𝒐𝒏𝒊 𝒅𝒆𝒕𝒔' (Non-standard characters)\n\n"
    "CRITICAL: Your response MUST be ONLY the single playlist name. No explanations, no 'Playlist Name:', no numbering, no extra text or formatting whatsoever.\n" # Made this instruction more direct and emphatic
)

def clean_playlist_name(name):
    if not isinstance(name, str):
        return ""
    print(f"DEBUG CLEAN AI: Input name: '{name}'") # Print name before cleaning

    name = ftfy.fix_text(name)

    name = unicodedata.normalize('NFKC', name)
    # Stricter regex: only allows characters explicitly mentioned in the prompt.
    cleaned_name = re.sub(r'[^a-zA-Z0-9\s\-\&\'!\.\,\?\(\)\[\]]', '', name)
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
    return cleaned_name


# --- Ollama Specific Function ---
def get_ollama_playlist_name(ollama_url, model_name, prompt_template, feature1, feature2, feature3, song_list):
    """
    Calls a self-hosted Ollama instance to get a playlist name.
    This version handles streaming responses and extracts only the non-think part.

    Args:
        ollama_url (str): The URL of your Ollama instance (e.g., "http://192.168.3.15:11434/api/generate").
        model_name (str): The Ollama model to use (e.g., "deepseek-r1:1.5b").
        prompt_template (str): The base template for the prompt.
        feature1 (str): The first primary category/feature.
        feature2 (str): The second primary category/feature.
        feature3 (str): The third primary category/feature.
        song_list (list): A list of dictionaries, where each dictionary contains 'title' and 'artist'.
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Format the initial part of the prompt using the template and features
    prompt_text = prompt_template.format(
        feature1=feature1,
        feature2=feature2,
        feature3=feature3
    )

    # Construct the user message content by appending the song list
    full_prompt = f"{prompt_text}\n\nHere is the list of titles and artists:\n"
    for song in song_list:
        full_prompt += f"* {song.get('title', 'Unknown Title')} by {song.get('author', 'Unknown Artist')}\n"

    # Ollama API endpoint is usually just the base URL + /api/generate
    options = {
        "num_predict": 5000,
        "temperature": 0.9
    }

    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": True,
        "options": options
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        print(f"DEBUG AI (Ollama): Starting API call for model '{model_name}' at '{ollama_url}'.") # Debug print

        # --- First API Call ---
        response = requests.post(ollama_url, headers=headers, data=json.dumps(payload), stream=True, timeout=480) # Increased timeout to 120 seconds
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        full_raw_response_content = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        full_raw_response_content += chunk['response']
                    if chunk.get('done'):
                        break
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON line from stream: {line.decode('utf-8', errors='ignore')}")
                    continue

        # Ollama models often include thought blocks, extract text after common thought tags
        # Using a simple approach: find the last occurrence of common thought block enders
        thought_enders = ["</think>", "[/INST]", "[/THOUGHT]"] # Add other common patterns if needed
        extracted_text = full_raw_response_content.strip()
        for end_tag in thought_enders:
             if end_tag in extracted_text:
                 extracted_text = extracted_text.split(end_tag, 1)[-1].strip() # Take everything after the last tag
        cleaned_name = clean_playlist_name(extracted_text)
        return cleaned_name # Length check is done in the general function

    except requests.exceptions.RequestException as e:
        # Catch network-related errors, bad HTTP responses, etc.
        return f"Error calling Ollama API: {e}"
    except Exception as e:
        # Catch any other unexpected errors.
        return f"An unexpected error occurred: {e}"

# --- Gemini Specific Function ---
def get_gemini_playlist_name(gemini_api_key, model_name, prompt_template, feature1, feature2, feature3, song_list):
    """
    Calls the Google Gemini API to get a playlist name.

    Args:
        gemini_api_key (str): Your Google Gemini API key.
        model_name (str): The Gemini model to use (e.g., "gemini-1.5-flash-latest").
        prompt_template (str): The base template for the prompt.
        feature1 (str): The first primary category/feature.
        feature2 (str): The second primary category/feature.
        feature3 (str): The third primary category/feature.
        song_list (list): A list of dictionaries, where each dictionary contains 'title' and 'artist'.
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    if not gemini_api_key: # Allow any provided key, even if it's the placeholder
         return "Error: Gemini API key is missing or empty. Please provide a valid API key."

    try:
        if GEMINI_API_CALL_DELAY_SECONDS > 0:
            print(f"DEBUG AI (Gemini): Waiting for {GEMINI_API_CALL_DELAY_SECONDS}s before API call to respect rate limits.")
            time.sleep(GEMINI_API_CALL_DELAY_SECONDS)
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name)

        # Format the initial part of the prompt using the template and features
        prompt_text = prompt_template.format(
            feature1=feature1,
            feature2=feature2,
            feature3=feature3
        )

        # Construct the user message content by appending the song list
        full_prompt = f"{prompt_text}\n\nHere is the list of titles and artists:\n"
        for song in song_list:
            # Note: Using 'author' key as per tasks.py song list structure
            full_prompt += f"* {song.get('title', 'Unknown Title')} by {song.get('author', 'Unknown Artist')}\n"

        print(f"DEBUG AI (Gemini): Starting API call for model '{model_name}'.") # Debug print
        response = model.generate_content(full_prompt, request_options={'timeout': 480}) # Increased timeout

        # Extract text from the response
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            extracted_text = "".join(part.text for part in response.candidates[0].content.parts)
        else:
            print(f"DEBUG AI (Gemini): No content in response. Raw response: {response}")
            return "Error: Gemini returned no content."

        cleaned_name = clean_playlist_name(extracted_text)
        return cleaned_name # Length check is done in the general function

    except Exception as e:
        return f"Error calling Gemini API: {e}"

# --- General AI Naming Function ---
def get_ai_playlist_name(provider, ollama_url, ollama_model_name, gemini_api_key, gemini_model_name, prompt_template, feature1, feature2, feature3, song_list):
    """
    Selects and calls the appropriate AI model based on the provider.
    Applies length constraints after getting the name.
    """
    MIN_LENGTH = 15
    MAX_LENGTH = 40

    name = "AI Naming Skipped" # Default if provider is NONE or invalid

    if provider == "OLLAMA":
        name = get_ollama_playlist_name(ollama_url, ollama_model_name, prompt_template, feature1, feature2, feature3, song_list)
    elif provider == "GEMINI":
        name = get_gemini_playlist_name(gemini_api_key, gemini_model_name, prompt_template, feature1, feature2, feature3, song_list)
    # else: provider is NONE or invalid, name remains "AI Naming Skipped"

    # Apply length check and return final name or error
    # Only apply length check if a name was actually generated (not the skip message or an API error message)
    if name not in ["AI Naming Skipped"] and not name.startswith("Error"):
        cleaned_name = clean_playlist_name(name)
        if MIN_LENGTH <= len(cleaned_name) <= MAX_LENGTH:
            return cleaned_name
        else:
            # Return an error message indicating the length issue, but include the cleaned name for debugging
            return f"Error: AI generated name '{cleaned_name}' ({len(cleaned_name)} chars) outside {MIN_LENGTH}-{MAX_LENGTH} range."
    else:
        # Return the original skip message or API error message
        return name
