import requests
import json
import os # For potential future use, though API key is passed directly

# Strengthened prompt template for strict output
# Now includes good and bad examples, and explicit instruction for conciseness.
# Added length constraint and instruction to avoid special characters/fonts, allowing emojis.

creative_prompt_template = (
    "Adopt the persona of a highly creative and insightful music curator, known for crafting unique and perfectly fitting playlist titles.\n\n"
    "Your task is to capture the essence of a playlist through its name.\n"
    "Given the primary characteristics '{feature1}', '{feature2}', and '{feature3}', and the list of songs (appended below), suggest ONLY 1 playlist name.\n\n"
    "The playlist name must be:\n"
    "1. Evocative & Creative: It should spark interest, be memorable, and hint at the playlist's overall vibe or a unique concept.\n"
    "2. Clearly Representative: It MUST accurately reflect the core essence of '{feature1}', '{feature2}', and '{feature3}' and the general atmosphere of the songs.\n"
    "3. Concise & Catchy: Strictly between 15 and 35 characters long.\n"
    "4. English Words Only: The name MUST use only standard English words. No invented words or words from other languages.\n\n"

    "Core Naming Strategy - Internal Thought Process (before finalizing):\n"
    "A. Feature Synthesis: First, deeply consider '{feature1}', '{feature2}', and '{feature3}'. Don't just see them as tags. What is the *unifying idea, feeling, or image* that creatively combines their spirits? How can they be *thematically linked or synergized* rather than just listed? The name must embody this combined essence.\n"
    "B. Song List Inspiration: Next, briefly scan the provided song titles and artists. Look for subtle lyrical moods (e.g., hopeful, defiant, serene), recurring keywords or imagery (e.g., journey, light, city, dreams), or distinctive artist personas (e.g., storyteller, innovator, crooner). If these emergent qualities harmonize with your synthesized features, let them subtly inspire or refine the playlist name, adding depth and uniqueness.\n\n"

    "Output Constraints:\n"
    "* The generated name ABSOLUTELY MUST be between 15 and 35 characters long.\n"
    # Updated character constraint to emphasize English words and standard characters.
    "* The name MUST use only standard English words. Avoid special fonts, non-English characters, or unusual symbols. Stick to standard English letters, numbers, spaces, and common punctuation (like -, &, ', !). Emojis are acceptable if they are common and widely understood.\n\n"

    "GOOD EXAMPLES (demonstrating feature synthesis & style with three features):\n"
    # Adjusted examples for three features
    "* For features 'chillout', 'electronic', and 'night', a name like 'Midnight Ethereal Circuits' (26 chars) is good. (Concept: otherworldly tech for the night; thematic blend)\n"
    "* For features 'female vocalists', '00s', and 'empowering', a name like 'Y2K Diva Power Anthems' (24 chars) is good. (Concept: era-specific, strong female songs; thematic blend)\n"
    "* For features 'classic rock', '70s', and 'road trip', a name like 'Seventies Highway Rock' (23 chars) is good. (Concept: epic journey with 70s rock soundtrack; thematic blend)\n"
    "* For features 'Mellow', 'acoustic', and 'morning', a name like 'Acoustic Sunrise Serenity' (26 chars) is good. (Concept: gentle, morning-vibe acoustic music; thematic blend)\n"
    "* For features 'punk', '90s', and 'rebellious', a name like '90s Riot Spirit Anthems' (24 chars) is good. (Concept: defiant, energetic songs from the 90s; thematic blend)\n\n"

    "ADDITIONAL GOOD EXAMPLES (illustrating general creative style):\n"
    "Rock Classics Marathon, Acoustic Serenity, Workout Warriors, Road Trip Rhythms, Productivity Power Hour, Pre-Party Hype, Chillhop & Lofi Beats.\n\n"

    "BAD EXAMPLES (what to avoid):\n"
    "* If features are '80s', 'rock', and 'energetic', do NOT suggest 'Midnight Pop Fever'. (Reason: Wrong genre for 'rock', doesn't represent '80s' clearly, poor feature synthesis).\n"
    "* If features are 'ambient', 'electronic', and 'space', do NOT suggest 'Ambient Electronic Space - Electric Soundscapes - Emotional Waves'. (Reason: Too long, verbose, and too literal; lacks creative synthesis).\n"
    "* If features are 'blues', 'rock', and 'fast', do NOT suggest 'Blues Rock Fast Tracks'. (Reason: Too direct, lacks creative integration).\n"
    # Added user's specific bad example
    "* Do NOT suggest names like '𝑯𝒘𝒆 𝒂𝒓𝒐𝒏𝒊 𝒅𝒆𝒕𝒔'. (Reason: Uses non-standard characters/fonts and non-English words).\n\n"
    "Your response MUST be ONLY the playlist name. Do NOT include any introductory or concluding remarks, explanations, bullet points, bolding, or any other formatting. Just the name."
)

import re
import ftfy # Import the ftfy library

def clean_playlist_name(name):
    """Basic cleaning to remove potentially problematic characters while allowing standard text, common punctuation, and a range of emojis."""
    if not isinstance(name, str):
        return ""
    
    name = ftfy.fix_text(name) # Use ftfy to fix mojibake and normalize Unicode oddities

    if not isinstance(name, str):
        return ""
    # Remove control characters, zero-width spaces, and a broad range of non-standard unicode symbols/formatting.
    # This regex keeps alphanumeric, spaces, common punctuation, and a range of common emojis.
    # It's not exhaustive for all emojis but covers many. Adjust if needed.
    # Basic Latin, Latin-1 Supplement, General Punctuation, Spacing Modifier Letters, Combining Diacritical Marks,
    # Emoticons, Dingbats, Transport and Map Symbols, Supplemental Symbols and Pictographs, etc.
    # This is a simplified approach; a more robust solution might use a library or a stricter allow-list.
    cleaned_name = re.sub(r'[^\w\s\-\&\'!\.\,\?\(\)\[\]\u00C0-\u00FF\u2000-\u206F\u20A0-\u20CF\u2100-\u214F\u2190-\u21FF\u2300-\u23FF\u24C0-\u24FF\u25A0-\u25FF\u2600-\u26FF\u2700-\u27BF\u2B50\U0001F300-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0000200D]', '', name)
    return cleaned_name.strip()

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
