import requests
import json

# Strengthened prompt template for strict output
# Now includes good and bad examples, and explicit instruction for conciseness.
# Added length constraint and instruction to avoid special characters/fonts, allowing emojis.

creative_prompt_template = (
    "You are a highly creative music curator. Suggest ONLY 1 concise (15-35 chars), evocative, and representative playlist name.\n"
    "Name MUST be made of real English words using ONLY standard ASCII letters (a-z, A-Z), numbers (0-9), spaces, and these punctuation: - & ' ! . , ? ( ) [ ]. No special fonts or emojis.\n"
    "Characteristics: '{feature1}', '{feature2}', '{feature3}'. Analyze the lsit of provided songs for core vibe.\n\n"
    "* GOOD EXAMPLES: 'Sunshine Pop Vibrations' (23 chars) is good. (Concept: cheerful, energetic pop for sunny days)\n"
    "* GOOD EXAMPLES: 'Workout Power Hour Mix' (22 chars) is good. (Concept: energetic rock and dance tracks for intense workouts)\n"
    "* GOOD EXAMPLES: 'Relaxing Evening Melodies' (25 chars) is good. (Concept: calming and beautiful songs for a peaceful evening)\n\n"
    "* BAD EXAMPLES:  For featrue like 'rock', 'energetic', and '80s', a name like 'Midnight Pop Fever' (Wrong genre/era for 80s, rock, energetic)\n"
    "* BAD EXAMPLES: 'Ambient Electronic Space - Electric Soundscapes - Emotional Waves' (Too long, literal)\n"
    "* BAD EXAMPLES: 'Blues Rock Fast Tracks' (Too direct)\n"
    "* BAD EXAMPLES: '𝑯𝒘𝒆 𝒂𝒓𝒐𝒏𝒊 𝒅𝒆𝒕𝒔' (Non-standard chars)\n\n"
    "Your response MUST be ONLY the playlist name. No other text.\n\n"
)
import re
import ftfy # Import the ftfy library
import unicodedata

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
    MIN_LENGTH = 15
    MAX_LENGTH = 40

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
        print(f"DEBUG AI (ai.py): Starting first API call for model '{model_name}'.") # Debug print

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
        
        think_end_tag = "</think>" # Using the unescaped version
        # --- Extract and Clean First Response ---
        if think_end_tag in full_raw_response_content:
            start_index = full_raw_response_content.rfind(think_end_tag) + len(think_end_tag)
            extracted_text = full_raw_response_content[start_index:].strip()
        else:
            extracted_text = full_raw_response_content.strip()
            
        cleaned_name = clean_playlist_name(extracted_text)
        
        print(f"DEBUG AI (ai.py): First attempt cleaned: '{cleaned_name}' (Length: {len(cleaned_name)})") # Debug print

        # --- Check Length of the Response ---
        if MIN_LENGTH <= len(cleaned_name) <= MAX_LENGTH:
            print(f"DEBUG AI (ai.py): First attempt length OK. Returning.") # Debug print
            return cleaned_name
        elif cleaned_name.startswith("Error") or cleaned_name.startswith("An unexpected error"):
             print(f"DEBUG AI (ai.py): First attempt returned error. Returning.") # Debug print
             return cleaned_name
        else:
            print(f"Warning: AI name attempt '{cleaned_name}' ({len(cleaned_name)} chars) is outside {MIN_LENGTH}-{MAX_LENGTH} range.")
            return f"Error: AI generated name outside {MIN_LENGTH}-{MAX_LENGTH} character range."

    except requests.exceptions.RequestException as e:
        # Catch network-related errors, bad HTTP responses, etc.
        return f"Error calling Ollama API: {e}"
    except Exception as e:
        # Catch any other unexpected errors.
        return f"An unexpected error occurred: {e}"
