"""
Simulation Runner.
Reads JSON scenarios, feeds them to the BrainCEMISID Orchestrator, 
and logs the results against a basic Control Group LLM.
"""
import json
import uuid
import sys
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Adjust path to import core correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.orchestrator import BrainCEmisidOrchestrator
from core.llm_engine import LLMEngine

def run_simulation(scenario_path: str, output_path: str):
    # Ensure variables are loaded first
    load_dotenv()
    """
    Runs a defined JSON scenario through BrainCEMISID and a control LLM,
    recording the outputs for later statistical comparison (t-Student).
    """
    
    # 1. Load Scenario
    try:
        with open(scenario_path, 'r', encoding='utf-8') as f:
            scenario_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Scenario file not found at {scenario_path}")
        sys.exit(1)
        
    print(f"--- Starting Scenario: {scenario_data.get('scenario', 'Unknown')} ---")
    
    # 2. Initialize Models
    print("Initializing BrainCEMISID Orchestrator...")
    run_uuid = str(uuid.uuid4())[:8]
    brain_orchestrator = BrainCEmisidOrchestrator(
        collection_name=f"sim_vault_{run_uuid}", 
        db_path=None # Ephemeral DB to not pollute production memory
    )
    
    print("Initializing Control Group LLM (Baseline)...")
    baseline_llm = LLMEngine() # Raw Gemini without Cognitive Architecture
    
    # 3. Execution Logs
    log_output = {
        "scenario_name": scenario_data.get('scenario'),
        "run_id": run_uuid,
        "timestamp": datetime.now().isoformat(),
        "frames": []
    }
    
    # Baseline Prompt Tracker (Control Group has simple conversational memory)
    baseline_history = ""
    
    # 4. Iterate through events
    for event in scenario_data.get("events", []):
        timestamp = event.get("timestamp", "T?")
        narrative = event.get("narrative", "")
        sensory_input = event.get("sensory_input", None)
        goal = event.get("goal", "")
        
        print(f"\n[{timestamp}] Processing Frame...")
        print(f"Narrative: {narrative}")
        print(f"Goal: {goal}")
        
        # --- BRAINCEMISID RUN ---
        print("\n  [BrainCEMISID Active]")
        brain_result = brain_orchestrator.process_frame(
            narrative=narrative, 
            goal=goal, 
            external_sensory_input=sensory_input
        )
        
        # --- BASELINE CONTROL GROUP RUN ---
        print("\n  [Baseline Control Active]")
        baseline_prompt = f"{baseline_history}\nNarrative: {narrative}\nGoal: {goal}\nResponse:"
        baseline_response = baseline_llm.generate_response(baseline_prompt)
        
        # Append for next baseline frame
        baseline_history += f"\nNarrative: {narrative}\nGoal: {goal}\nResponse: {baseline_response}\n"
        
        # --- LOG DATA ---
        frame_log = {
            "timestamp": timestamp,
            "narrative": narrative,
            "goal": goal,
            "braincemisid": {
                "dominant_emotion": brain_result["dominant_emotion"],
                "emotion_intensity": brain_result["emotion_intensity"],
                "strategic_plan": brain_result["plan"],
                "final_responses": [r["llm_output"] for r in brain_result["responses"]]
            },
            "baseline_control": {
                "final_response": baseline_response
            }
        }
        
        log_output["frames"].append(frame_log)
        
        # Terminal Summary
        print(f"\n--- Frame {timestamp} Summary ---")
        print(f"Brain Emotion: {brain_result['dominant_emotion']} ({brain_result['emotion_intensity']:.2f})")
        print(f"Brain Plan Steps: {len(brain_result['plan']['plan'])}")
        print("-" * 30)

    # 5. Save log
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(log_output, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Simulation complete. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrainCEMISID Scenario Runner")
    parser.add_argument('--scenario', type=str, default='simulation/emergency_sector_7.json', help='Path to JSON scenario')
    parser.add_argument('--output', type=str, default='simulation/results_log.json', help='Path to output JSON log')
    args = parser.parse_args()
    
    run_simulation(args.scenario, args.output)
