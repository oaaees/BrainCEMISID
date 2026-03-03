"""
Planner module. Implements the COALA-inspired strategic decomposition.
"""
from typing import List, Dict, Any
from core.llm_engine import LLMEngine
import re
import json

class StrategicPlanner:
    """
    Transforms the agent from reactive to proactive by decomposing User Goals
    into an actionable sequence of steps, justified by the current cognitive state.
    """

    def __init__(self, llm_engine: LLMEngine):
        self.llm_engine = llm_engine

    def decompose_task(self, goal: str, cognitive_snapshot: str) -> Dict[str, Any]:
        """
        Analyzes a goal and current state to generate a sequence of reasoned steps.
        
        Args:
            goal: The overall user goal or objective.
            cognitive_snapshot: A string representing the current Senses + Emotions.
            
        Returns:
            A dictionary containing the 'thought' and the 'plan' (list of steps with reasons).
        """
        prompt = (
            "You are the Strategic Planner module of a cognitive architecture based on the COALA framework. "
            "Your task is to decompose the User Goal into a sequence of discrete sub-tasks (maximum 5 steps) "
            "to maintain low latency. Your plan must be heavily influenced by the provided Cognitive Snapshot.\n\n"
            f"Cognitive Snapshot (Senses + Emotions):\n{cognitive_snapshot}\n\n"
            f"User Goal: \"{goal}\"\n\n"
            "Return a strictly formatted JSON object with the following structure:\n"
            "{\n"
            "  \"thought\": \"A brief analysis of why this plan is needed given the emotional state.\",\n"
            "  \"plan\": [\n"
            "    {\"step\": \"Action description\", \"reason\": \"Why this step was chosen based on the emotional state (e.g. 'Current fear level is 0.8')\"}\n"
            "  ]\n"
            "}\n"
            "Do not include any Markdown blocks, just the raw JSON object."
        )
        
        response = self.llm_engine.generate_response(prompt)
        
        try:
            # Clean up potential markdown blocks
            clean_response = re.sub(r'```(?:json)?|```', '', response).strip()
            plan_data = json.loads(clean_response)
            
            # Ensure proper structure
            if "thought" not in plan_data or "plan" not in plan_data:
                raise ValueError("JSON missing required 'thought' or 'plan' keys.")
                
            # Enforce max 5 steps
            plan_data["plan"] = plan_data["plan"][:5]
            return plan_data
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse strategic plan. Raw response: {response}")
            # Fallback reactive plan
            return {
                "thought": "Failed to generate complex plan; falling back to reactive response.",
                "plan": [{"step": f"Process input: '{goal}' directly.", "reason": "Fallback mechanism triggered."}]
            }
