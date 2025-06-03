import requests
import json
import os # For potential future use, though API key is passed directly

# Strengthened prompt template for strict output
# Now includes good and bad examples, and explicit instruction for conciseness.
creative_prompt_template = (
    "These songs are selected to have similar genre, mood, bmp or other characteristics. "
    "Given the primary categories '{feature1} {feature2}', suggest only 1 concise, creative, and memorable playlist name. "
    "The generated name ABSOLUTELY MUST include both '{feature1}' and '{feature2}', but integrate them creatively, not just by directly re-using the tags. "
    "Keep the playlist name concise and not excessively long. "
    "The full category is '{category_name}' where the last feature is BPM"
    "GOOD EXAMPLE: For '80S Rock', a good name is 'Festive 80S Rock & Pop Mix'. "
    "GOOD EXAMPLE: For 'Ambient Electronic', a good name is 'Ambitive Electronic Experimental Fast'. "
    "BAD EXAMPLE: If categories are '80S Rock', do NOT suggest 'Midnight Pop Fever'. "
    "BAD EXAMPLE: If categories are 'Ambient Electronic', do NOT suggest 'Ambient Electronic - Electric Soundscapes - Ambient Artists, Tracks & Emotional Waves' (it's too long and verbose). "
    "BAD EXAMPLE: If categories are 'Blues Rock', do NOT suggest 'Blues Rock - Fast' (it's too direct and not creative enough). "
    "Your response MUST be ONLY the playlist name. Do NOT include any introductory or concluding remarks, explanations, bullet points, bolding, or any other formatting. Just the name.")

def get_ollama_playlist_name(ollama_url, model_name, prompt_template, feature1, feature2, category_name, song_list):
    """
    Calls a self-hosted Ollama instance to get a playlist name.
    This version handles streaming responses and extracts only the non-think part.

    Args:
        ollama_url (str): The URL of your Ollama instance (e.g., "http://192.168.3.15:11434/api/generate").
        model_name (str): The Ollama model to use (e.g., "deepseek-r1:1.5b").
        prompt_template (str): The base template for the prompt.
        feature1 (str): The first primary category/feature.
        feature2 (str): The second primary category/feature.
        category_name (str): The full original category name.
        song_list (list): A list of dictionaries, where each dictionary contains 'title' and 'artist'.
    Returns:
        str: The extracted playlist name from the model's response, or an error message.
    """
    # Format the initial part of the prompt using the template and features
    prompt_text = prompt_template.format(
        feature1=feature1,
        feature2=feature2,
        category_name=category_name
    )

    # Construct the user message content by appending the song list
    full_prompt = f"{prompt_text}\n\nHere is the list of titles and artists:\n"
    for song in song_list:
        full_prompt += f"* {song.get('title', 'Unknown Title')} by {song.get('author', 'Unknown Artist')}\n"

    # The final instruction for the model is already part of the creative_prompt_template
    # Re-iterate it for Ollama as per the example
    full_prompt += "\n**Your response MUST be ONLY the playlist name. Do NOT include any introductory or concluding remarks, explanations, bullet points, bolding, or any other formatting. Just the name.**"

    options = {
        "num_predict": 3000,
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
        response = requests.post(ollama_url, headers=headers, data=json.dumps(payload), stream=True, timeout=240) # Increased timeout to 120 seconds
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
        if think_end_tag in full_raw_response_content:
            start_index = full_raw_response_content.rfind(think_end_tag) + len(think_end_tag)
            extracted_text = full_raw_response_content[start_index:].strip()
        else:
            extracted_text = full_raw_response_content.strip()
        return extracted_text
    except requests.exceptions.RequestException as e:
        # Catch network-related errors, bad HTTP responses, etc.
        return f"Error calling Ollama API: {e}"
    except Exception as e:
        # Catch any other unexpected errors.
        return f"An unexpected error occurred: {e}"
