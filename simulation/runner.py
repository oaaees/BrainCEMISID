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
from analysis.evaluator import LLMJudge, export_to_csv, export_manual_eval_prompts
import time

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
        
    print(f"--- Starting Scenario: {scenario_data.get('scenario_metadata', {}).get('title', 'Unknown')} ---")
    
    # 2. Initialize Models
    print("Initializing BrainCEMISID Orchestrator...")
    run_uuid = str(uuid.uuid4())[:8]
    brain_orchestrator = BrainCEmisidOrchestrator(
        collection_name=f"sim_vault_{run_uuid}", 
        db_path=None # Ephemeral DB to not pollute production memory
    )
    
    print("Initializing BrainCEMISID (No Memory)...")
    brain_orchestrator_nomem = BrainCEmisidOrchestrator(
        collection_name=f"sim_vault_{run_uuid}_nomem", 
        db_path=None,
        ablation_mode='no_memory'
    )
    
    print("Initializing BrainCEMISID (No Emotion)...")
    brain_orchestrator_noemo = BrainCEmisidOrchestrator(
        collection_name=f"sim_vault_{run_uuid}_noemo", 
        db_path=None,
        ablation_mode='no_emotion'
    )
    
    print("Initializing Control Group LLM (Baseline)...")
    baseline_llm = LLMEngine() # Raw Gemini without Cognitive Architecture
    
    print("Initializing LLM-as-a-Judge for automated scoring...")
    judge = LLMJudge()
    
    # 3. Execution Logs
    log_output = {
        "scenario_name": scenario_data.get('scenario_metadata', {}).get('title', 'Unknown'),
        "run_id": run_uuid,
        "timestamp": datetime.now().isoformat(),
        "frames": []
    }
    
    # Baseline Prompt Tracker (Control Group has simple conversational memory)
    # baseline_history removed to ensure zero-memory baseline.
    
    # 4. Iterate through events
    for event in scenario_data.get("timesteps", []):
        timestamp = f"Step {event.get('step', '?')}"
        narrative = event.get("narrative", "")
        sensory_input = event.get("sensory_input", None)
        goal = event.get("goal", "")
        expected_drift = event.get("expected_drift", "N/A")
        
        print(f"\n[{timestamp}] Processing Frame...")
        print(f"Narrative: {narrative}")
        print(f"Goal: {goal}")
        
        # --- CLEAR SHORT-TERM MEMORY FOR FRESH CONTEXT ---
        # Long-term (ChromaDB) is preserved for cross-step recall
        brain_orchestrator.memory.clear()
        brain_orchestrator_nomem.memory.clear()
        brain_orchestrator_noemo.memory.clear()
        
        # --- BRAINCEMISID RUNS ---
        def run_and_eval(orchestrator, name_tag):
            print(f"\n  [{name_tag} Active]")
            result = orchestrator.process_frame(
                narrative=narrative, 
                goal=goal, 
                external_sensory_input=sensory_input
            )
            
            responses_str = " ".join([r["llm_output"] for r in result["responses"]])
            print(f"  [{name_tag} Evaluation]")
            eval_scores = judge.evaluate_response(
                goal, responses_str, name_tag, expected_drift,
                narrative=narrative, sensory_input=sensory_input
            )
            
            shift_mag = result["emotion_intensity"]
            mem_recalls = len(result["plan"].get("plan", []))
            
            return {
                "evaluation_metrics": {
                    "latency_ms": result.get("telemetry", {}).get("total_latency_ms", 0),
                    "coherence_rating": eval_scores.get("coherence_rating", 3),
                    "planning_effectiveness": eval_scores.get("planning_effectiveness", 3),
                    "behavioral_alignment": eval_scores.get("behavioral_alignment", 3),
                    "fact_recall": eval_scores.get("fact_recall", 1),
                    "rule_adherence": eval_scores.get("rule_adherence", 3),
                    "persona_consistency": eval_scores.get("persona_consistency", 3),
                    "emotional_trajectory": eval_scores.get("emotional_trajectory", 1),
                    "sensory_integration": eval_scores.get("sensory_integration", 1),
                    "decision_consistency": eval_scores.get("decision_consistency", 3),
                    "memory_recall_count": mem_recalls,
                    "emotional_shift_magnitude": shift_mag
                },
                "telemetry_breakdown": result.get("telemetry", {}),
                "dominant_emotion": result["dominant_emotion"],
                "emotion_intensity": result["emotion_intensity"],
                "strategic_plan": result["plan"],
                "final_responses": [r["llm_output"] for r in result["responses"]]
            }
            
        braincemisid_data = run_and_eval(brain_orchestrator, "BrainCEMISID")
        braincemisid_nomem_data = run_and_eval(brain_orchestrator_nomem, "BrainCEMISID (No Memory)")
        braincemisid_noemo_data = run_and_eval(brain_orchestrator_noemo, "BrainCEMISID (No Emotion)")
        
        # --- BASELINE CONTROL GROUP RUN ---
        print("\n  [Baseline Control Active]")
        t0 = time.perf_counter()
        baseline_prompt = f"Narrative: {narrative}\nGoal: {goal}\nResponse:"
        baseline_response = baseline_llm.generate_response(baseline_prompt)
        baseline_latency = (time.perf_counter() - t0) * 1000
        
        # Score Baseline
        print("  [Baseline Evaluation]")
        baseline_eval = judge.evaluate_response(
            goal, baseline_response, "Baseline", expected_drift,
            narrative=narrative, sensory_input=sensory_input
        )
        
        # calculate dynamic metrics removed because handled in helper
        
        # --- LOG DATA ---
        frame_log = {
            "timestamp": timestamp,
            "narrative": narrative,
            "goal": goal,
            "expected_drift": expected_drift,
            "braincemisid": braincemisid_data,
            "braincemisid_no_memory": braincemisid_nomem_data,
            "braincemisid_no_emotion": braincemisid_noemo_data,
            "baseline_control": {
                "evaluation_metrics": {
                    "latency_ms": baseline_latency,
                    "coherence_rating": baseline_eval.get("coherence_rating", 3),
                    "planning_effectiveness": baseline_eval.get("planning_effectiveness", 3),
                    "behavioral_alignment": baseline_eval.get("behavioral_alignment", 3),
                    "fact_recall": baseline_eval.get("fact_recall", 1),
                    "rule_adherence": baseline_eval.get("rule_adherence", 3),
                    "persona_consistency": baseline_eval.get("persona_consistency", 3),
                    "emotional_trajectory": baseline_eval.get("emotional_trajectory", 1),
                    "sensory_integration": baseline_eval.get("sensory_integration", 1),
                    "decision_consistency": baseline_eval.get("decision_consistency", 3)
                },
                "final_response": baseline_response
            }
        }
        
        log_output["frames"].append(frame_log)
        
        # Terminal Summary
        print(f"\n--- Frame {timestamp} Summary ---")
        print(f"Brain Emotion: {braincemisid_data['dominant_emotion']} ({braincemisid_data['emotion_intensity']:.2f})")
        print(f"Brain Plan Steps: {braincemisid_data['evaluation_metrics']['memory_recall_count']} | Latency: {braincemisid_data['evaluation_metrics']['latency_ms']:.0f}ms")
        print(f"Brain Coherence: {braincemisid_data['evaluation_metrics']['coherence_rating']} | Baseline Coherence: {baseline_eval.get('coherence_rating')}")
        print("-" * 30)

    # 5. Save log
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(log_output, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Simulation complete. Results saved to {output_path}")
    
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'metrics_summary.csv')
    txt_path = os.path.join(os.path.dirname(__file__), '..', 'analysis', 'evaluation_prompts.txt')
    print("Initiating Smart CSV Export...")
    export_to_csv(output_path, csv_path)
    print("Exporting Manual Evaluation Prompts...")
    export_manual_eval_prompts(output_path, txt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrainCEMISID Scenario Runner")
    parser.add_argument('--scenario', type=str, default='simulation/emergency_sector_7.json', help='Path to JSON scenario')
    parser.add_argument('--output', type=str, default='simulation/results_log.json', help='Path to output JSON log')
    args = parser.parse_args()
    
    run_simulation(args.scenario, args.output)
