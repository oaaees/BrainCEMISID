"""
Emotional Engine module. Simulates behavioral drift based on sensory stimuli.
"""
from typing import Dict, Tuple
from core.llm_engine import LLMEngine
import re
import json

class EmotionalState:
    """
    Maintains and shifts the current emotional state using a continuous mathematical model.
    Values for each emotion range from 0.0 to 1.0.
    """
    def __init__(self, llm_engine: LLMEngine, decay_rate: float = 0.05):
        self.llm_engine = llm_engine
        self.decay_rate = decay_rate
        
        # 5 Base Emotions
        self.emotions = {
            "Joy": 0.0,
            "Sadness": 0.0,
            "Anger": 0.0,
            "Fear": 0.0,
            "Surprise": 0.0
        }

    def shift_emotion(self, sensory_snapshot: Dict[str, str], recent_input: str) -> None:
        """
        Calculates the emotion delta based on sensory input, applies decay,
        and updates the internal emotional state.
        
        Args:
            sensory_snapshot: The qualitative sensory snapshot from SensoryGate.
            recent_input: The raw text input.
        """
        # 1. Apply natural decay to all emotions towards baseline 0.0
        for emotion in self.emotions:
            self.emotions[emotion] = max(0.0, self.emotions[emotion] - self.decay_rate)
            
        # Prepare the sensory state as a readable string
        senses = []
        for sense, description in sensory_snapshot.items():
            if description and description.lower() != "none":
                senses.append(f"{sense.capitalize()}: {description}")
        
        # If no sensory input, we just naturally decayed.
        if not senses:
            return
            
        senses_str = "\n".join(senses)
        
        # 2. Semantic Matching via LLM to find deltas
        prompt = (
            "You are the Emotional Mapping Table of a cognitive architecture. "
            "Analyze the following incoming sensory data and determine its emotional impact.\n\n"
            f"Sensory stimuli:\n{senses_str}\n\n"
            f"Contextual Input:\n\"{recent_input}\"\n\n"
            "Return a strictly formatted JSON object with the following keys: "
            "'Joy', 'Sadness', 'Anger', 'Fear', 'Surprise'.\n"
            "The values must be a float between 0.0 (no impact) and 0.5 (very strong impact), "
            "representing how much this sensory stimuli increases that specific emotion.\n"
            "Do not include any Markdown blocks, just the raw JSON object."
        )
        
        response = self.llm_engine.generate_response(prompt)
        
        try:
            # Clean up potential markdown blocks
            clean_response = re.sub(r'```(?:json)?|```', '', response).strip()
            deltas = json.loads(clean_response)
            
            # 3. Apply the Deltas (E_t = E_{t-1} + Delta)
            for emotion in self.emotions:
                if emotion in deltas:
                    delta_value = float(deltas[emotion])
                    # Clamp between 0.0 and 1.0
                    self.emotions[emotion] = min(1.0, self.emotions[emotion] + delta_value)
                    
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse emotional deltas. Raw response: {response}")

    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Returns the emotion with the highest current value."""
        dominant = max(self.emotions.items(), key=lambda x: x[1])
        # If all emotions are 0.0, return Neutral
        if dominant[1] == 0.0:
            return ("Neutral", 0.0)
        return dominant

    def get_personality_string(self) -> str:
        """
        Generates a string describing the agent's current behavioral state 
        to inject into the LLM system prompt.
        """
        active_emotions = {k: v for k, v in self.emotions.items() if v > 0.1}
        
        if not active_emotions:
            return "You are feeling completely neutral, calm, and analytical."
            
        dominant_emotion, intensity = self.get_dominant_emotion()
        
        intensity_str = "slightly"
        if intensity > 0.7:
            intensity_str = "highly"
        elif intensity > 0.4:
            intensity_str = "moderately"
            
        prompt = f"You are feeling {intensity_str} driven by {dominant_emotion}. "
        
        # Add nuance for secondary emotions
        secondary = [f"{k} ({v:.2f}/1.0)" for k, v in active_emotions.items() if k != dominant_emotion]
        if secondary:
            prompt += f"You also have underlying feelings of: {', '.join(secondary)}. "
            
        prompt += "Let these emotions subtly influence the tone, word choice, and perspective of your response without explicitly stating your emotions unless asked."
        return prompt
        
    def get_metadata_dict(self) -> Dict[str, str]:
        """Provides the emotional state formatted for ChromaDB metadata."""
        dominant_emotion, _ = self.get_dominant_emotion()
        metadata = {"dominant_emotion": dominant_emotion}
        for k, v in self.emotions.items():
             metadata[f"emotion_{k.lower()}"] = str(round(v, 2))
        return metadata
