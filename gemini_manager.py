# gemini_manager.py
import os
import threading
from dotenv import load_dotenv
from google import genai
from google.genai import types
import requests
import json

# Load environment variables from .env file
load_dotenv()

class GeminiManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GeminiManager, cls).__new__(cls)
                # Initialize Google GenAI client
                # Assumes GEMINI_API_KEY is in environment
                cls._instance.client = genai.Client()
        return cls._instance

    def get_models(self):
        """Returns recommended Gemini models for dictation and embedding, plus local Ollama models."""
        models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-embedding-001"]
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                for m in data.get("models", []):
                    models.append(m["name"])
        except Exception as e:
            print(f"Could not fetch Ollama models: {e}")
        return models

    def is_gemini_model(self, model_name):
        return model_name.startswith("gemini")

    def get_embedding(self, text, model_name="gemini-embedding-001"):
        if not self.is_gemini_model(model_name):
            try:
                response = requests.post("http://localhost:11434/api/embeddings", json={
                    "model": model_name,
                    "prompt": text
                }, timeout=15)
                if response.status_code == 200:
                    return response.json().get("embedding", [])
                else:
                    print(f"Ollama embedding error: {response.text}")
                    return [0.0] * 384 # Fallback dimension representation
            except Exception as e:
                print(f"Ollama embedding exception: {e}")
                return [0.0] * 384

        try:
             response = self.client.models.embed_content(
                model=model_name,
                contents=text
            )
             return response.embeddings[0].values
        except Exception as e:
            print(f"Embedding error: {e}")
            # gemini-embedding-001 has dimension 3072
            return [0.0] * 3072

    def get_embeddings_batch(self, texts, model_name="gemini-embedding-001"):
        """Embeds a list of texts in a single batch call."""
        if not texts:
            return []
            
        if not self.is_gemini_model(model_name):
            # Ollama supports arrays in /api/embed
            try:
                response = requests.post("http://localhost:11434/api/embed", json={
                    "model": model_name,
                    "input": texts
                }, timeout=60)
                if response.status_code == 200:
                    return response.json().get("embeddings", [])
                else:
                    # Fallback to single embeddings
                    return [self.get_embedding(t, model_name) for t in texts]
            except Exception as e:
                print(f"Ollama batch exception: {e}")
                return [self.get_embedding(t, model_name) for t in texts]

        try:
            response = self.client.models.embed_content(
                model=model_name,
                contents=texts
            )
            return [emb.values for emb in response.embeddings]
        except Exception as e:
            print(f"Batch embedding error: {e}")
            # Ensure the output matches the input size with zero vectors
            return [[0.0] * 3072 for _ in range(len(texts))]

    def generate_stream(self, prompt, system_prompt, model_name="gemini-2.5-flash", temperature=1.0, max_tokens=None):
        if not self.is_gemini_model(model_name):
            try:
                options = {"temperature": temperature}
                if max_tokens: options["num_predict"] = max_tokens
                
                response = requests.post("http://localhost:11434/api/generate", json={
                    "model": model_name,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": True,
                    "options": options
                }, stream=True)
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                return
            except Exception as e:
                yield f"\n\nError generating response from Ollama: {e}"
                return

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens if max_tokens else None
            )
            response = self.client.models.generate_content_stream(
                model=model_name,
                contents=prompt,
                config=config
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"\n\nError generating response: {e}"

    def generate_sync(self, prompt, system_prompt, model_name="gemini-2.5-flash", temperature=1.0, max_tokens=None):
        if not self.is_gemini_model(model_name):
            try:
                options = {"temperature": temperature}
                if max_tokens: options["num_predict"] = max_tokens
                
                response = requests.post("http://localhost:11434/api/generate", json={
                    "model": model_name,
                    "prompt": prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": options
                })
                if response.status_code == 200:
                    return response.json().get("response", "")
                else:
                    return f"Error: {response.text}"
            except Exception as e:
                print(f"Ollama generation error: {e}")
                return f"Error: {e}"

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens if max_tokens else None
            )
            response = self.client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            print(f"Generation error: {e}")
            return f"Error: {e}"
