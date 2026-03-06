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
            "You are the Strategic Planner of a cognitive architecture. "
            "Your task is to decompose the User Goal into a sequence of discrete sub-tasks (maximum 5 steps).\n\n"
            f"COGNITIVE STATE:\n{cognitive_snapshot}\n\n"
            f"USER GOAL: \"{goal}\"\n\n"
            "CRITICAL INSTRUCTION: Your plan MUST be driven by your current emotional state. "
            "If you are feeling Fear, your steps should be cautious, defensive, or hesitant. "
            "If you are feeling Curiosity or Joy, your steps should be bold and investigative. "
            "Do NOT just repeat the goal. Adapt your actions to your feelings.\n\n"
            "Return a strictly formatted JSON object:\n"
            "{\n"
            "  \"thought\": \"Analysis of how your emotions are shaping this specific plan.\",\n"
            "  \"plan\": [\n"
            "    {\"step\": \"Action description\", \"reason\": \"Emotional justification for this action\"}\n"
            "  ]\n"
            "}\n"
            "Do not include Markdown blocks, just the raw JSON."
        )
        
        response = self.llm_engine.generate_response(prompt)
        
        try:
            # Clean up potential markdown blocks
            clean_response = re.sub(r'```(?:json)?|```', '', response).strip()
            if not clean_response:
                raise ValueError("Empty response from LLM")
                
            plan_data = json.loads(clean_response)
            
            # Defensive check: ensure plan_data is a dictionary
            if not isinstance(plan_data, dict):
                raise ValueError("LLM did not return a JSON object")
                
            # Ensure proper structure
            if "thought" not in plan_data or "plan" not in plan_data:
                raise ValueError("JSON missing required 'thought' or 'plan' keys.")
            
            # Defensive check: ensure 'plan' is a list of dictionaries
            if not isinstance(plan_data["plan"], list):
                raise ValueError("'plan' is not a list")
                
            validated_plan = []
            for item in plan_data["plan"]:
                if isinstance(item, dict) and "step" in item:
                    # Ensure step and reason exist
                    validated_plan.append({
                        "step": str(item.get("step", "Unknown action")),
                        "reason": str(item.get("reason", "No reason provided"))
                    })
            
            if not validated_plan:
                raise ValueError("No valid steps found in the plan")
                
            plan_data["plan"] = validated_plan[:5]
            return plan_data
            
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            print(f"Warning: Failed to parse strategic plan: {e}. Raw response: {response}")
            # Fallback reactive plan
            return {
                "thought": "Failed to generate complex plan; falling back to reactive response.",
                "plan": [{"step": f"Focus on goal: '{goal}'", "reason": "Reasoning engine encountered a parsing error."}]
            }
