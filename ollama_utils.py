# ollama_utils.py
import requests
import json

def get_ollama_models(host="http://localhost:11434"):
    """Fetches available local models from Ollama."""
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]
    except requests.exceptions.RequestException:
        return []

def refine_text_stream(raw_text, system_prompt, model_name, host="http://localhost:11434"):
    """Streams the refined text generator from local Ollama."""
    url = f"{host}/api/generate"
    payload = {
        "model": model_name,
        "system": system_prompt,
        "prompt": raw_text,
        "stream": True
    }
    
    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        # Yield each chunk of text as it arrives
                        yield chunk["response"]
    except requests.exceptions.RequestException as e:
        yield f"\n[Ollama Connection Error: {e}. Ensure Ollama is running and the model is downloaded.]"