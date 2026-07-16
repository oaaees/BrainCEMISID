"""
LLM Engine module for interacting with the Google Gemini API or Local Ollama.
"""
import os
import requests
import json
import time
from typing import Optional, List
from sentence_transformers import SentenceTransformer

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class LLMEngine:
    """Wrapper class for the LLM API (Gemini or local Ollama) to handle inference."""

    def __init__(self, use_local: bool = False, model_name: Optional[str] = None, remote_delay: float = 1.0):
        """
        Initializes the LLM Engine.
        
        Args:
            use_local: If True, uses local Ollama and SentenceTransformers.
            model_name: The model to use. Default for local: 'gemma3:1b', for Gemini: 'gemini-2.0-flash'.
            remote_delay: Seconds to wait after a remote call to avoid rate limits.
        """
        self.use_local = use_local
        self.remote_delay = remote_delay
        self._embedding_cache = {}
        
        if self.use_local:
            self.model_name = model_name or "gemma3:1b"
            self.ollama_url = "http://localhost:11434/api/generate"
            print(f"Initializing Local LLM Engine (Ollama: {self.model_name})...")
            # Local embedding model (runs on CPU/GPU offline)
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model_name = model_name or "gemma-3-4b-it"
            print(f"Initializing Remote LLM Engine (Gemini: {self.model_name})...")
            self.api_key = os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY is not set. Please set it in your environment or pass it directly.")
            
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai package not found. Run 'pip install google-generativeai'.")
                
            genai.configure(api_key=self.api_key)
            self.remote_model = genai.GenerativeModel(self.model_name)

    def generate_response(self, prompt: str, max_output_tokens: int = 300) -> str:
        """Generates a text response based on the provided prompt.
        
        Args:
            prompt: The text prompt to send to the LLM.
            max_output_tokens: Maximum number of tokens in the output. Prevents runaway repetition.
        """
        if self.use_local:
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_output_tokens
                    }
                }
                print(f"  [LLM] Requesting local response from {self.model_name}...")
                response = requests.post(self.ollama_url, json=payload)
                response.raise_for_status()
                print(f"  [LLM] Local response received.")
                return response.json().get("response", "Error: No response from Ollama")
            except Exception as e:
                print(f"Error generating local response: {e}")
                return f"Error: {e}"
        else:
            try:
                print(f"  [LLM] Requesting remote response from {self.model_name}...")
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=max_output_tokens
                )
                response = self.remote_model.generate_content(
                    prompt, 
                    generation_config=generation_config
                )
                print(f"  [LLM] Remote response received. Waiting {self.remote_delay}s...")
                time.sleep(self.remote_delay)
                return response.text
            except Exception as e:
                print(f"Error generating remote response: {e}")
                return f"Error: {e}"

    def generate_embedding(self, text: str) -> List[float]:
        """Generates an embedding vector for the provided text."""
        if text in self._embedding_cache:
            # print(f"  [Embedding] Cache hit for text ({len(text)} chars).") # Optional: uncomment if you want logging
            return self._embedding_cache[text]
            
        if self.use_local:
            try:
                # Generates a 384-dimensional vector locally
                print(f"  [Embedding] Encoding text locally ({len(text)} chars)...")
                embedding = self.embedding_model.encode(text)
                result = embedding.tolist()
                self._embedding_cache[text] = result
                return result
            except Exception as e:
                print(f"Error generating local embedding: {e}")
                return []
        else:
            try:
                print(f"  [Embedding] Requesting remote embedding ({len(text)} chars)...")
                result_api = genai.embed_content(
                    model="models/gemini-embedding-001", 
                    content=text,
                    task_type="retrieval_document"
                )
                print(f"  [Embedding] Remote embedding received. Waiting {self.remote_delay}s...")
                time.sleep(self.remote_delay)
                
                result = result_api['embedding']
                self._embedding_cache[text] = result
                return result
            except Exception as e:
                print(f"Error generating remote embedding: {e}")
                return []
