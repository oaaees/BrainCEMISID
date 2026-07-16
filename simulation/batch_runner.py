import os
import subprocess
import sys

def run_all_simulations():
    """
    Finds all .json files in the simulation folder and runs them through runner.py
    """
    sim_dir = "simulation"
    runner_script = os.path.join(sim_dir, "runner.py")
    
    # Get all .json files except results_log.json
    scenarios = [f for f in os.listdir(sim_dir) if f.endswith('.json') and f != 'results_log.json']
    
    if not scenarios:
        print("No scenarios found in simulation/ folder.")
        return

    print(f"Found {len(scenarios)} scenarios: {', '.join(scenarios)}")
    
    # Use the current python executable (from venv)
    python_exe = sys.executable

    for scenario in scenarios:
        scenario_path = os.path.join(sim_dir, scenario)
        print(f"\n" + "="*50)
        print(f"RUNNING SCENARIO: {scenario}")
        print("="*50)
        
        try:
            # Run the command and wait for it to finish
            subprocess.run([python_exe, runner_script, "--scenario", scenario_path], check=True)
            print(f"\n✅ Finished: {scenario}")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error running {scenario}: {e}")
            # Continue to next scenario or stop? Let's continue.
            continue
            
    print("\n" + "="*50)
    print("ALL SIMULATIONS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    run_all_simulations()
