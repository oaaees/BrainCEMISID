"""
LLM Engine module for interacting with the Google Gemini API or Local Ollama.
"""
import os
import requests
import json
from typing import Optional, List
from sentence_transformers import SentenceTransformer

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class LLMEngine:
    """Wrapper class for the LLM API (Gemini or local Ollama) to handle inference."""

    def __init__(self, use_local: bool = True, model_name: Optional[str] = None):
        """
        Initializes the LLM Engine.
        
        Args:
            use_local: If True, uses local Ollama and SentenceTransformers.
            model_name: The model to use. Default for local: 'gemma3:1b', for Gemini: 'gemini-2.0-flash'.
        """
        self.use_local = use_local
        
        if self.use_local:
            self.model_name = model_name or "gemma3:1b"
            self.ollama_url = "http://localhost:11434/api/generate"
            print(f"Initializing Local LLM Engine (Ollama: {self.model_name})...")
            # Local embedding model (runs on CPU/GPU offline)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model_name = model_name or "gemini-2.0-flash"
            print(f"Initializing Remote LLM Engine (Gemini: {self.model_name})...")
            self.api_key = os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not set. Please set it in your environment or pass it directly.")
            
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai package not found. Run 'pip install google-generativeai'.")
                
            genai.configure(api_key=self.api_key)
            self.remote_model = genai.GenerativeModel(self.model_name)

    def generate_response(self, prompt: str) -> str:
        """Generates a text response based on the provided prompt."""
        if self.use_local:
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
                response = requests.post(self.ollama_url, json=payload)
                response.raise_for_status()
                return response.json().get("response", "Error: No response from Ollama")
            except Exception as e:
                print(f"Error generating local response: {e}")
                return f"Error: {e}"
        else:
            try:
                response = self.remote_model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Error generating remote response: {e}")
                return f"Error: {e}"

    def generate_embedding(self, text: str) -> List[float]:
        """Generates an embedding vector for the provided text."""
        if self.use_local:
            try:
                # Generates a 384-dimensional vector locally
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            except Exception as e:
                print(f"Error generating local embedding: {e}")
                return []
        else:
            try:
                result = genai.embed_content(
                    model="models/gemini-embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            except Exception as e:
                print(f"Error generating remote embedding: {e}")
                return []
