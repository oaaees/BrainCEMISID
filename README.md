# BrainCEMISID: Synthetic Cognitive Architecture

A research project for a Systems Engineering thesis implementing a COALA-inspired cognitive architecture.

## 🚀 Quick Start (Local Mode)

1. **Environment**:
   ```powershell
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Ollama**:
   Ensure Ollama is running and run `ollama pull gemma3:1b`.

3. **Simulation**:
   ```powershell
   python simulation/runner.py
   ```

4. **Statistical Analysis**:
   ```powershell
   python analysis/stats_engine.py
   ```

## 📂 Project Structure
- `core/`: Cognitive modules (Sensors, Emotions, Memory, Planner).
- `simulation/`: Scenario definitions and simulation runner.
- `analysis/`: Experimental data, stats engine, and visualization plots.
- `tests/`: Unit testing suite.
 
