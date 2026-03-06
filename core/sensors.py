"""
Sensory module to extract qualitative sensory descriptions from raw input.
"""
from typing import Dict
from core.llm_engine import LLMEngine
import re
import json

class SensoryGate:
    """
    Parses raw textual input into a semantic sensory descriptor using an LLM.
    """
    def __init__(self, llm_engine: LLMEngine):
        self.llm_engine = llm_engine

    def extract_senses(self, text: str) -> Dict[str, str]:
        """
        Extracts sensory qualitative data from the input text.
        
        Args:
            text: Raw input text from the environment/user.
            
        Returns:
            A dictionary with the 5 senses as keys and qualitative descriptions as values.
        """
        prompt = (
            "You are the Sensory Gate of a cognitive architecture. Your task is to extract "
            "qualitative sensory information from the following text.\n\n"
            f"Input Text: \"{text}\"\n\n"
            "Return a strictly formatted JSON object with the following keys: "
            "'sight', 'hearing', 'smell', 'touch', 'taste'.\n"
            "If a sense is not present or implied in the text, use the value 'None'.\n"
            "Keep descriptions very short (1-3 words) and qualitative (e.g., 'dim blue lighting').\n"
            "Do not include any Markdown blocks, just the raw JSON object."
        )
        
        response = self.llm_engine.generate_response(prompt)
        
        try:
            # Clean up potential markdown blocks if the LLM includes them anyway
            clean_response = re.sub(r'```(?:json)?|```', '', response).strip()
            sensory_data = json.loads(clean_response)
            
            # Defensive check: ensure it's a dictionary
            if not isinstance(sensory_data, dict):
                raise ValueError("Sensory data is not a JSON object")
                
            # Ensure all keys exist and are formatted well
            final_data = {}
            for sense in ["sight", "hearing", "smell", "touch", "taste"]:
                val = sensory_data.get(sense, "None")
                final_data[sense] = str(val) if val else "None"
                
            return final_data
        
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Warning: Failed to parse sensory data: {e}. Raw response: {response}")
            return {
                "sight": "None", "hearing": "None", "smell": "None", 
                "touch": "None", "taste": "None"
            }
