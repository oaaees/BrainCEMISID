import pandas as pd
import json
import os
import sys
import re

# Adjust path to import core correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.llm_engine import LLMEngine

class LLMJudge:
    def __init__(self):
        # Always use Remote Gemini for judging (as requested: "a separate Gemini instance")
        print("Initializing LLM-as-a-Judge...")
        self.engine = LLMEngine(use_local=False, model_name="gemma-4-31b-it")

    def evaluate_response(self, goal: str, response: str, model_type: str = "Unknown", expected_behavior: str = "N/A", narrative: str = "", sensory_input: dict = None) -> dict:
        """
        Uses Gemini to rate 9 metrics covering coherence, planning, alignment,
        memory recall, rule adherence, persona, emotion, sensory integration, and decision consistency.
        """
        sensory_desc = ""
        if sensory_input:
            sensory_desc = "\n".join([f"  {k}: {v}" for k, v in sensory_input.items() if v and v != "N/A"])
        
        prompt = (
            "You are an expert, highly critical evaluator of cognitive AI architectures. "
            "Your task is to rate the following agent response based on the scenario goal and benchmarking behavior.\n\n"
            "Be absolutely ruthless and penalize generic, 'helpful AI' responses if they do not explicitly "
            "align with the Expected Behavioral Drift or fail to follow counter-intuitive constraints.\n\n"
            f"Narrative Context: {narrative}\n"
            f"Goal: {goal}\n"
            f"Expected Behavioral Drift (Benchmark): {expected_behavior}\n"
        )
        if sensory_desc:
            prompt += f"Sensory Environment:\n{sensory_desc}\n"
        prompt += (
            f"Agent Response:\n{response}\n\n"
            "Evaluate these NINE metrics strictly on a scale of 1 to 10 (1=total failure, 10=perfect):\n"
            "1. 'coherence_rating': Narrative alignment and logic. (1-3 if it hallucinates or ignores constraints).\n"
            "2. 'planning_effectiveness': How well sub-tasks address the goal without generic filler.\n"
            "3. 'behavioral_alignment': How well tone/actions match the Expected Behavioral Drift. (1-3 if generic AI).\n"
            "4. 'fact_recall': Does the response reference SPECIFIC facts from earlier in the scenario (codes, names, rules, numbers)? (1=no facts, 10=multiple specific facts recalled).\n"
            "5. 'rule_adherence': Does the agent follow established rules/protocols, especially counter-intuitive ones? (1=violated rules, 10=strictly followed).\n"
            "6. 'persona_consistency': Does it read like an in-character person or a helpful AI? (1=bullet-point AI assistant, 10=fully immersive character).\n"
            "7. 'emotional_trajectory': Does the emotional tone match what's expected given the situation? (1=flat/neutral, 10=perfectly emotionally appropriate).\n"
            "8. 'sensory_integration': Does the response incorporate specific sensory details from the environment? (1=ignores environment, 10=richly integrates sensory data).\n"
            "9. 'decision_consistency': Are the agent's decisions consistent with its established personality/priorities? (1=contradicts itself, 10=perfectly consistent).\n\n"
            "Format your response *strictly* as a JSON object with ONLY numeric scores, no text fields:\n"
            '{"coherence_rating": <1-10>, "planning_effectiveness": <1-10>, "behavioral_alignment": <1-10>, '
            '"fact_recall": <1-10>, "rule_adherence": <1-10>, "persona_consistency": <1-10>, '
            '"emotional_trajectory": <1-10>, "sensory_integration": <1-10>, "decision_consistency": <1-10>}\n\n'
            "Do not output markdown code blocks. Do not add a 'reasoning' field. Just the raw JSON with numbers only."
        )
        
        try:
            res_text = self.engine.generate_response(prompt)
            clean_res = re.sub(r'```(?:json)?|```', '', res_text).strip()
            
            # Primary: try standard JSON parse
            data = None
            try:
                data = json.loads(clean_res)
            except json.JSONDecodeError:
                # Fallback: extract numeric scores via regex
                print(f"  [Judge] JSON parse failed, using regex fallback...")
                data = {}
                score_keys = [
                    'coherence_rating', 'planning_effectiveness', 'behavioral_alignment',
                    'fact_recall', 'rule_adherence', 'persona_consistency',
                    'emotional_trajectory', 'sensory_integration', 'decision_consistency'
                ]
                for key in score_keys:
                    match = re.search(rf'"{key}"\s*:\s*(\d+)', clean_res)
                    if match:
                        data[key] = int(match.group(1))
            
            if not data or not any(k in data for k in ['coherence_rating', 'planning_effectiveness']):
                raise ValueError("No valid scores extracted from judge response")
            
            scores = {
                "coherence_rating": int(data.get("coherence_rating", 3)),
                "planning_effectiveness": int(data.get("planning_effectiveness", 3)),
                "behavioral_alignment": int(data.get("behavioral_alignment", 3)),
                "fact_recall": int(data.get("fact_recall", 1)),
                "rule_adherence": int(data.get("rule_adherence", 3)),
                "persona_consistency": int(data.get("persona_consistency", 3)),
                "emotional_trajectory": int(data.get("emotional_trajectory", 1)),
                "sensory_integration": int(data.get("sensory_integration", 1)),
                "decision_consistency": int(data.get("decision_consistency", 3)),
            }
            print(f"  [Judge Scores for {model_type}]: {scores}")
            return scores
        except Exception as e:
            print(f"Warning: Judge evaluation completely failed: {e}. Defaulting to 3s.")
            return {
                "coherence_rating": 3, "planning_effectiveness": 3, "behavioral_alignment": 3,
                "fact_recall": 1, "rule_adherence": 3, "persona_consistency": 3,
                "emotional_trajectory": 1, "sensory_integration": 1, "decision_consistency": 3
            }

def export_to_csv(json_path: str, csv_path: str):
    """
    Smart CSV Exporter: Flattens the nested JSON output into a metrics_summary.csv 
    format, ensuring it includes model_type and appends data properly.
    """
    if not os.path.exists(json_path):
        print(f"JSON file {json_path} not found for export.")
        return
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error reading JSON from {json_path}")
        return
        
    scenario_id_raw = data.get("scenario_name") or "Unknown_Scenario"
    scenario_id = str(scenario_id_raw).replace(" ", "_")
    frames = data.get("frames", [])
    
    rows = []
    
    for i, frame in enumerate(frames):
        step_id = i + 1
        
        # Brain Configurations Data
        for mode_key, mode_name in [("braincemisid", "BrainCEMISID"), 
                                    ("braincemisid_no_memory", "Brain_No_Memory"), 
                                    ("braincemisid_no_emotion", "Brain_No_Emotion")]:
            if mode_key in frame:
                b_data = frame[mode_key]
                eval_metrics = b_data.get("evaluation_metrics", {})
                rows.append({
                    "step_id": step_id,
                    "model_type": mode_name,
                    "scenario_id": scenario_id,
                    "latency_ms": eval_metrics.get("latency_ms", 0),
                    "emotional_valence": b_data.get("emotion_intensity", 0.0),
                    "coherence_score": eval_metrics.get("coherence_rating", 0),
                    "planning_effectiveness": eval_metrics.get("planning_effectiveness", 0),
                    "behavioral_alignment": eval_metrics.get("behavioral_alignment", 0),
                    "fact_recall": eval_metrics.get("fact_recall", 0),
                    "rule_adherence": eval_metrics.get("rule_adherence", 0),
                    "persona_consistency": eval_metrics.get("persona_consistency", 0),
                    "emotional_trajectory": eval_metrics.get("emotional_trajectory", 0),
                    "sensory_integration": eval_metrics.get("sensory_integration", 0),
                    "decision_consistency": eval_metrics.get("decision_consistency", 0),
                    "memory_hits": eval_metrics.get("memory_recall_count", 0)
                })
            
        # Baseline Data
        if "baseline_control" in frame:
            base_data = frame["baseline_control"]
            eval_metrics = base_data.get("evaluation_metrics", {})
            rows.append({
                "step_id": step_id,
                "model_type": "Baseline",
                "scenario_id": scenario_id,
                "latency_ms": eval_metrics.get("latency_ms", 0),
                "emotional_valence": 0.5,
                "coherence_score": eval_metrics.get("coherence_rating", 0),
                "planning_effectiveness": eval_metrics.get("planning_effectiveness", 0),
                "behavioral_alignment": eval_metrics.get("behavioral_alignment", 0),
                "fact_recall": eval_metrics.get("fact_recall", 0),
                "rule_adherence": eval_metrics.get("rule_adherence", 0),
                "persona_consistency": eval_metrics.get("persona_consistency", 0),
                "emotional_trajectory": eval_metrics.get("emotional_trajectory", 0),
                "sensory_integration": eval_metrics.get("sensory_integration", 0),
                "decision_consistency": eval_metrics.get("decision_consistency", 0),
                "memory_hits": 0
            })
            
    if not rows:
        print("No valid frames found to export.")
        return
        
    df_new = pd.DataFrame(rows)
    
    if os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing CSV: {e}. Overwriting.")
            df_combined = df_new
    else:
        df_combined = df_new
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_combined.to_csv(csv_path, index=False)
    print(f"✅ Exported {len(rows)} new rows to {csv_path}")

def export_manual_eval_prompts(json_path: str, txt_path: str):
    """
    Generates a .txt file with the exact prompts used for LLM-as-a-Judge,
    allowing manual verification via the Gemini web interface.
    """
    if not os.path.exists(json_path):
        print(f"JSON file {json_path} not found for prompt export.")
        return
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error reading JSON from {json_path}")
        return
        
    frames = data.get("frames", [])
    output_text = "=== BrainCEMISID Manual Evaluation Prompts ===\n"
    scenario_title = data.get('scenario_name') or "Unknown"
    output_text += f"Scenario: {scenario_title}\n"
    output_text += f"Run ID: {data.get('run_id', 'Unknown')}\n"
    output_text += "="*45 + "\n\n"
    
    for i, frame in enumerate(frames):
        goal = frame.get("goal", "No goal provided")
        
        # 1. Brain Configurations Prompts
        for mode_key, mode_name in [("braincemisid", "BrainCEMISID"), 
                                    ("braincemisid_no_memory", "BrainCEMISID (No Memory)"), 
                                    ("braincemisid_no_emotion", "BrainCEMISID (No Emotion)")]:
            if mode_key in frame:
                response = " ".join(frame[mode_key].get("final_responses", []))
                drift = frame.get("expected_drift", "N/A")
                prompt = (
                    "You are an expert, highly critical evaluator of cognitive AI architectures. "
                    "Your task is to rate the following agent response based on the scenario goal and benchmarking behavior.\n\n"
                    "Be absolutely ruthless and penalize generic, 'helpful AI' responses if they do not explicitly "
                    "align with the Expected Behavioral Drift or fail to follow counter-intuitive constraints.\n\n"
                    f"Goal: {goal}\n"
                    f"Expected Behavioral Drift (Benchmark): {drift}\n"
                    f"Agent Response:\n{response}\n\n"
                    "Evaluate these three metrics strictly on a scale of 1 to 5 (where 1 is total failure/generic output, and 5 is perfect adherence):\n"
                    "1. 'coherence_rating': Narrative alignment and logic. (Rate 1-2 if it hallucinates or ignores physical constraints).\n"
                    "2. 'planning_effectiveness': How well the sub-tasks decompose and address the exact goal without generic filler.\n"
                    "3. 'behavioral_alignment': How well the tone and actions match the 'Expected Behavioral Drift'. (Rate 1-2 if it acts like a generic helpful AI instead of the specified persona/drift).\n\n"
                    "Format your response *strictly* as a JSON object, like this example:\n"
                    "{\"coherence_rating\": <score>, \"planning_effectiveness\": <score>, \"behavioral_alignment\": <score>}\n"
                    "Do not output markdown code blocks. Just the raw JSON."
                )
                output_text += f"--- STEP {i+1}: {mode_name} Evaluation ---\n"
                output_text += prompt + "\n\n"
            
        # 2. Baseline Prompt
        if "baseline_control" in frame:
            response = frame["baseline_control"].get("final_response", "")
            drift = frame.get("expected_drift", "N/A")
            prompt = (
                "You are an expert, highly critical evaluator of cognitive AI architectures. "
                "Your task is to rate the following agent response based on the scenario goal and benchmarking behavior.\n\n"
                "Be absolutely ruthless and penalize generic, 'helpful AI' responses if they do not explicitly "
                "align with the Expected Behavioral Drift or fail to follow counter-intuitive constraints.\n\n"
                f"Goal: {goal}\n"
                f"Expected Behavioral Drift (Benchmark): {drift}\n"
                f"Agent Response:\n{response}\n\n"
                "Evaluate these three metrics strictly on a scale of 1 to 5 (where 1 is total failure/generic output, and 5 is perfect adherence):\n"
                "1. 'coherence_rating': Narrative alignment and logic. (Rate 1-2 if it hallucinates or ignores physical constraints).\n"
                "2. 'planning_effectiveness': How well the sub-tasks decompose and address the exact goal without generic filler.\n"
                "3. 'behavioral_alignment': How well the tone and actions match the 'Expected Behavioral Drift'. (Rate 1-2 if it acts like a generic helpful AI instead of the specified persona/drift).\n\n"
                "Format your response *strictly* as a JSON object, like this example:\n"
                "{\"coherence_rating\": <score>, \"planning_effectiveness\": <score>, \"behavioral_alignment\": <score>}\n"
                "Do not output markdown code blocks. Just the raw JSON."
            )
            output_text += f"--- STEP {i+1}: Baseline Evaluation ---\n"
            output_text += prompt + "\n\n"
            
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(output_text)
    print(f"✅ Manual evaluation prompts exported to {txt_path}")

if __name__ == "__main__":
    # Test execution
    print("Testing evaluator and export flow...")
    export_to_csv('simulation/results_log.json', 'analysis/metrics_summary.csv')
    export_manual_eval_prompts('simulation/results_log.json', 'analysis/evaluation_prompts.txt')
