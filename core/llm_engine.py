"""
LLM Engine module for interacting with the Google Gemini API.
"""
import os
import google.generativeai as genai
from typing import Optional

class LLMEngine:
    """Wrapper class for the Gemini API to handle inference."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the LLM Engine.
        
        Args:
            api_key: The API key for Gemini. If None, it attempts to load from the environment.
            model_name: The Gemini model to use. default: 'gemini-2.5-flash'.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please set it in your environment or pass it directly.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_response(self, prompt: str) -> str:
        """
        Generates a response from the Gemini model based on the provided prompt.
        
        Args:
            prompt: The input prompt string.
            
        Returns:
            The generated text response.
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            # Handle potential API errors gracefully
            print(f"Error generating response from LLM: {e}")
            return f"Error: {e}"

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generates an embedding vector for the provided text using Gemini.
        
        Args:
            text: The text to embed.
            
        Returns:
            A list of floats representing the embedding vector.
        """
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []
