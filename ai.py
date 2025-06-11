# /home/guido/Music/AudioMuse-AI/ai.py

import requests
import json
import re
import ftfy # Import the ftfy library
import time # Import the time library
import unicodedata
import google.generativeai as genai # Import Gemini library
import os # Import os to potentially read GEMINI_API_CALL_DELAY_SECONDS

# creative_prompt_template is imported in tasks.py, so it should be defined here
creative_prompt_template = (
    "You are a highly creative music curator. Your SOLE task is to generate 1 concise (15-35 chars) playlist name.\n"
    "The name MUST be evocative, representative of the provided songs/features, and use real English words with ONLY standard ASCII (a-z, A-Z, 0-9, spaces, and - & ' ! . , ? ( ) [ ]).\n"
    "No special fonts or emojis.\n"
    "CRITICAL: The most important features for naming are the genres: '{feature1}', '{feature2}', '{feature3}'. The playlist name MUST incorporate at least one of these genres. Analyze the provided song list for its core vibe.\n"
    "Input Mood feat to consider: {additional_features_description}, use this mainly to say if something is relax, or for party or similar.\n"
    "Input Energy to consider: {energy_description} use this to say if the music has slow energy or high energy (0 min, 1 max).\n"
    "Mood and Energy have less importance than the genres.\n"
    "The playlist name should suggest an activity, mood, or context, similar to these:\n\n"
    "* GOOD EXAMPLES: 'Sunshine Pop Vibrations' (Concept: cheerful, energetic pop for sunny days)\n"
    "* GOOD EXAMPLES: 'Workout Power Hour Mix' (Concept: energetic rock/dance for intense workouts)\n"
    "* GOOD EXAMPLES: 'Relaxing Evening Melodies' (Concept: calming songs for a peaceful evening)\n\n"
    "* BAD EXAMPLES: For features 'rock', 'energetic', '80s' -> 'Midnight Pop Fever' (Inconsistent with features)\n"
    "* BAD EXAMPLES: 'Ambient Electronic Space - Electric Soundscapes - Emotional Waves' (Too long/descriptive)\n"
    "* BAD EXAMPLES: 'Blues Rock Fast Tracks' (Too direct/literal, not evocative enough)\n"
    "* BAD EXAMPLES: '𝑯𝒘𝒆 𝒂𝒓𝒐𝒏𝒊 𝒅𝒆𝒕𝒔' (Non-standard characters)\n\n"
    "CRITICAL: Your response MUST be ONLY the single playlist name. No explanations, no 'Playlist Name:', no numbering, no extra text or formatting whatsoever.\n"
)

def clean_playlist_name(name):
    if not isinstance(name, str):
        return ""
    # print(f"DEBUG CLEAN AI: Input name: '{name}'") # Print name before cleaning

    name = ftfy.fix_text(name)

    name = unicodedata.normalize('NFKC', name)
    # Stricter regex: only allows characters explicitly mentioned in the prompt.
    cleaned_name = re.sub(r'[^a-zA-Z0-9\s\-\&\'!\.\,\?\(\)\[\]]', '', name)
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
    return cleaned_name


# --- Ollama Specific Function ---
def get_ollama_playlist_name(ollama_url, model_name, full_prompt):
    """
    Calls a self-hosted Ollama instance to get a playlist name.
    This version handles streaming responses and extracts only the non-think part.

    Args:
        ollama_url (str): The URL of your Ollama instance (e.g., "http://192.168.3.15:11434/api/generate").
        model_name (str): The Ollama model to use (e.g., "deepseek-r1:1.5b").
        full_prompt (str): The complete prompt text to send to the model.
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Ollama API endpoint is usually just the base URL + /api/generate
    options = {
        "num_predict": 5000, # Max tokens to generate
        "temperature": 0.9
    }

    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "stream": True, # We handle streaming to get the full response
        "options": options
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        print(f"DEBUG AI (Ollama): Starting API call for model '{model_name}' at '{ollama_url}'.") # Debug print

        response = requests.post(ollama_url, headers=headers, data=json.dumps(payload), stream=True, timeout=960) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        full_raw_response_content = ""
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        full_raw_response_content += chunk['response']
                    if chunk.get('done'):
                        break # Stop processing when the 'done' signal is received
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
        # The final cleaning and length check is done in the general function
        return extracted_text

    except requests.exceptions.RequestException as e:
        # Catch network-related errors, bad HTTP responses, etc.
        return f"Error calling Ollama API: {e}"
    except Exception as e:
        # Catch any other unexpected errors.
        return f"An unexpected error occurred: {e}"

# --- Gemini Specific Function ---
def get_gemini_playlist_name(gemini_api_key, model_name, full_prompt):
    """
    Calls the Google Gemini API to get a playlist name.

    Args:
        gemini_api_key (str): Your Google Gemini API key.
        model_name (str): The Gemini model to use (e.g., "gemini-1.5-flash-latest").
        full_prompt (str): The complete prompt text to send to the model.
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Allow any provided key, even if it's the placeholder, but check if it's empty/default
    if not gemini_api_key or gemini_api_key == "YOUR-GEMINI-API-KEY-HERE":
         return "Error: Gemini API key is missing or empty. Please provide a valid API key."

    try:
        # Read delay from environment/config if needed, otherwise use the default
        gemini_call_delay = int(os.environ.get("GEMINI_API_CALL_DELAY_SECONDS", "7"))
        if gemini_call_delay > 0:
            print(f"DEBUG AI (Gemini): Waiting for {gemini_call_delay}s before API call to respect rate limits.")
            time.sleep(gemini_call_delay)

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name)

        print(f"DEBUG AI (Gemini): Starting API call for model '{model_name}'.") # Debug print
        response = model.generate_content(full_prompt, request_options={'timeout': 960}) # Increased timeout

        # Extract text from the response
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            extracted_text = "".join(part.text for part in response.candidates[0].content.parts)
        else:
            print(f"DEBUG AI (Gemini): No content in response. Raw response: {response}")
            return "Error: Gemini returned no content."

        # The final cleaning and length check is done in the general function
        return extracted_text

    except Exception as e:
        return f"Error calling Gemini API: {e}"

# --- General AI Naming Function ---
def get_ai_playlist_name(provider, ollama_url, ollama_model_name, gemini_api_key, gemini_model_name, prompt_template, feature1, feature2, feature3, song_list, other_feature_scores_dict):
    """
    Selects and calls the appropriate AI model based on the provider.
    Constructs the full prompt including new features.
    Applies length constraints after getting the name.
    """
    MIN_LENGTH = 15
    MAX_LENGTH = 40

    # --- Prepare the prompt ---
    # Add information about the new features to the prompt.
    # This is a conceptual example; you'll need to refine the prompt engineering.
    new_features_description = ""
    energy_description = "" # Initialize energy description
    if other_feature_scores_dict:
        # Extract energy score first, as it's handled separately
        # Check for 'energy_normalized' first, then fall back to 'energy'
        energy_score = other_feature_scores_dict.get('energy_normalized', other_feature_scores_dict.get('energy', 0.0))

        # Create energy description based on score (example thresholds)
        if energy_score < 0.3:
            energy_description = " It has low energy."
        elif energy_score > 0.7:
            energy_description = " It has high energy."
        # No description if medium energy (between 0.3 and 0.7)

        # Sort other features by score descending to highlight the most prominent ones.
        # Explicitly exclude both 'energy' and 'energy_normalized' from this list,
        # as energy is handled separately by 'energy_description'.
        sorted_other_features = sorted([
            (name, score) for name, score in other_feature_scores_dict.items()
            if name not in ['energy', 'energy_normalized'] # Exclude both potential energy keys
        ], key=lambda item: item[1], reverse=True)
        # Example threshold: include features with score > 0.6 (adjust as needed)
        prominent_features = [f"{name}" for name, score in sorted_other_features if score > 0.6] # Example threshold
        if prominent_features:
            new_features_description = " It is also notably " + ", ".join(prominent_features) + "."

    # Format the song list sample for the prompt
    formatted_song_list = "\n".join([f"- {song.get('title', 'Unknown Title')} by {song.get('author', 'Unknown Artist')}" for song in song_list[:5]]) # Limit sample size to 5

    # Construct the full prompt using the template and all features
    full_prompt = prompt_template.format(
        feature1=feature1,
        feature2=feature2,
        feature3=feature3,
        additional_features_description=new_features_description, # Insert the new features description
        energy_description=energy_description, # Insert the energy description
        song_list_sample=formatted_song_list
    )

    print(f"Sending prompt to AI ({provider}):\n{full_prompt}") # Log the prompt for debugging

    # --- Call the AI Model ---
    name = "AI Naming Skipped" # Default if provider is NONE or invalid

    if provider == "OLLAMA":
        name = get_ollama_playlist_name(ollama_url, ollama_model_name, full_prompt)
    elif provider == "GEMINI":
        name = get_gemini_playlist_name(gemini_api_key, gemini_model_name, full_prompt)
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