"""
Entry point for the BrainCEMISID architecture.
Orchestrates data flow: Input -> Memory -> LLM -> Output
"""
import os
import sys
from dotenv import load_dotenv

from core.llm_engine import LLMEngine
from core.memory import Memory
from core.planner import StrategicPlanner
from core.sensors import SensoryGate
from core.emotions import EmotionalState

def main():
    # Load environment variables from .env file (if it exists)
    load_dotenv()

    print("Initializing BrainCEMISID...")

    try:
        # Dependency Injection: Instantiating modules
        # The LLM engine will pick up GEMINI_API_KEY from the environment
        llm_engine = LLMEngine()
        # Inject the embedding function into Memory for ChromaDB
        memory = Memory(embedding_fn=llm_engine.generate_embedding)
        planner = StrategicPlanner(llm_engine=llm_engine)
        sensory_gate = SensoryGate(llm_engine=llm_engine)
        emotions = EmotionalState(llm_engine=llm_engine)
        
    except ValueError as e:
        print(f"Initialization Error: {e}")
        sys.exit(1)

    print("BrainCEMISID is ready. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down BrainCEMISID...")
                break
            
            if not user_input.strip():
                continue

            # 1. Sensory processing
            sensory_snapshot = sensory_gate.extract_senses(user_input)
            
            # 2. Emotional Shift
            emotions.shift_emotion(sensory_snapshot, user_input)
            dominant_emotion, intensity = emotions.get_dominant_emotion()
            personality_string = emotions.get_personality_string()
            
            print(f"[Internal State] Emotion: {dominant_emotion} ({intensity:.2f}/1.0) | Senses: {sum(1 for v in sensory_snapshot.values() if v.lower() != 'none')}/5")
            
            # Prepare cognitive snapshot string for the planner
            cognitive_snapshot = f"Emotion: {dominant_emotion} ({intensity:.2f}/1.0)\n"
            active_senses = [f"{k.capitalize()}: {v}" for k, v in sensory_snapshot.items() if v.lower() != 'none']
            if active_senses:
                cognitive_snapshot += f"Active Senses: {', '.join(active_senses)}"
            else:
                cognitive_snapshot += "Active Senses: None"
                
            # 3. Planning (Task Decomposition)
            plan_data = planner.decompose_task(user_input, cognitive_snapshot)
            print(f"\n[Strategic Plan Thought] {plan_data['thought']}")
            
            # 4. Execution Loop
            for i, step_info in enumerate(plan_data['plan']):
                step_action = step_info['step']
                step_reason = step_info['reason']
                
                print(f"\n  ➤ Step {i+1}: {step_action} (Reason: {step_reason})")
                
                # Context Retrieval PER STEP
                long_term_context = memory.retrieve_relevant_context(step_action, top_k=3)
                
                prompt = memory.build_prompt(
                    new_input=f"Execute Phase {i+1}: {step_action}", 
                    long_term_context=long_term_context, 
                    current_emotion=personality_string, 
                    sensory_snapshot=sensory_snapshot
                )
                
                # LLM Inference
                response = llm_engine.generate_response(prompt)
                print(f"  🤖 Agent: {response}")
                
                # Memory Flow: Store interaction per step
                memory.add_interaction(role="user", content=f"Phase {i+1}: {step_action}")
                memory.add_interaction(role="agent", content=response)
                
                # Vectorize and save to long-term memory with metadata
                user_metadata = {
                    "role": "user",
                    "plan_step": str(i+1),
                    **emotions.get_metadata_dict(),
                    **{f"sense_{k}": v for k, v in sensory_snapshot.items() if v.lower() != 'none'}
                }
                memory.store_memory(f"User executing step {i+1} against goal '{user_input}': {step_action}", metadata=user_metadata)
                
                agent_metadata = {
                    "role": "agent",
                    "plan_step": str(i+1),
                    **emotions.get_metadata_dict()
                }
                memory.store_memory(f"Agent response to step {i+1} ({step_action}): {response}", metadata=agent_metadata)
            
            print("\n--- Plan Execution Complete ---\n")

        except KeyboardInterrupt:
            print("\nShutting down BrainCEMISID...")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
