from dotenv import load_dotenv
load_dotenv()
from google import genai
import os
import json
client = genai.Client()

print("Available models:")
for m in client.models.list():
    if "embedContent" in m.supported_actions or "embedText" in m.supported_actions:
        print(f"Embedding model: {m.name}")
        
try:
    print("Trying single text embedding with text-embedding-004...")
    single = client.models.embed_content(model="text-embedding-004", contents="hello")
    print(f"Single success: {len(single.embeddings[0].values)}")
except Exception as e:
    print(f"Single err: {e}")

try:
    print("Trying batch text embedding with text-embedding-004...")
    batch = client.models.embed_content(model="text-embedding-004", contents=["hello", "world"])
    print(f"Batch success: {len(batch.embeddings)}")
except Exception as e:
    print(f"Batch err: {e}")
