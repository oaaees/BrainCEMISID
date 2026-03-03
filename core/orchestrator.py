"""
Core Orchestrator module. 
Encapsulates all modules (Sensors, Emotions, Memory, Planner, LLM) into a single cohesive entity 
for simple interactions and targeted simulation running.
"""
from typing import Dict, Any, List
from core.llm_engine import LLMEngine
from core.memory import Memory
from core.planner import StrategicPlanner
from core.sensors import SensoryGate
from core.emotions import EmotionalState

class BrainCEmisidOrchestrator:
    """
    The central coordinator of the BrainCEMISID cognitive architecture.
    """
    def __init__(self, collection_name: str = "brain_memory", db_path: str = "./chroma_db", decay_rate: float = 0.05):
        """
        Initializes and links all cognitive modules.
        """
        self.llm_engine = LLMEngine()
        
        # Injected dependencies
        self.memory = Memory(
            collection_name=collection_name, 
            db_path=db_path, 
            embedding_fn=self.llm_engine.generate_embedding
        )
        self.sensory_gate = SensoryGate(llm_engine=self.llm_engine)
        self.emotions = EmotionalState(llm_engine=self.llm_engine, decay_rate=decay_rate)
        self.planner = StrategicPlanner(llm_engine=self.llm_engine)

    def process_frame(self, narrative: str, goal: str, external_sensory_input: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Processes a single "frame" or "turn" of interaction.
        This represents the complete COALA cognitive cycle.
        
        Args:
            narrative: The contextual text or event happening in this frame.
            goal: The agent's immediate objective.
            external_sensory_input: Optional pre-parsed sensory data. If None, the SensoryGate will extract it from the narrative.
            
        Returns:
            A dictionary containing the full state, plan, and final generated responses.
        """
        # 1. Sensory processing
        if external_sensory_input:
            sensory_snapshot = external_sensory_input
        else:
            sensory_snapshot = self.sensory_gate.extract_senses(narrative)
            
        # 2. Emotional Shift
        self.emotions.shift_emotion(sensory_snapshot, narrative)
        dominant_emotion, intensity = self.emotions.get_dominant_emotion()
        personality_string = self.emotions.get_personality_string()
        
        # Prepare cognitive snapshot string for the planner
        cognitive_snapshot = f"Emotion: {dominant_emotion} ({intensity:.2f}/1.0)\n"
        active_senses = [f"{k.capitalize()}: {v}" for k, v in sensory_snapshot.items() if v.lower() != 'none']
        if active_senses:
            cognitive_snapshot += f"Active Senses: {', '.join(active_senses)}"
        else:
            cognitive_snapshot += "Active Senses: None"
            
        # 3. Strategic Planning
        plan_data = self.planner.decompose_task(goal, cognitive_snapshot)
        
        # 4. Execution Loop & Final Accumulation
        responses = []
        for i, step_info in enumerate(plan_data['plan']):
            step_action = step_info['step']
            
            # Context Retrieval
            long_term_context = self.memory.retrieve_relevant_context(step_action, top_k=3)
            
            prompt = self.memory.build_prompt(
                new_input=f"Current Narrative: {narrative}\nTarget Goal: {goal}\nExecute Phase {i+1}: {step_action}", 
                long_term_context=long_term_context, 
                current_emotion=personality_string, 
                sensory_snapshot=sensory_snapshot
            )
            
            # Application
            response = self.llm_engine.generate_response(prompt)
            responses.append({"step": step_action, "llm_output": response})
            
            # Memory Flow Logging
            self.memory.add_interaction(role="user", content=f"Phase {i+1}: {step_action}")
            self.memory.add_interaction(role="agent", content=response)
            
            # Vectorize
            user_metadata = {
                "role": "user",
                "plan_step": str(i+1),
                **self.emotions.get_metadata_dict(),
                **{f"sense_{k}": v for k, v in sensory_snapshot.items() if v.lower() != 'none'}
            }
            self.memory.store_memory(f"User executing step {i+1} against goal '{goal}': {step_action}", metadata=user_metadata)
            
            agent_metadata = {
                "role": "agent",
                "plan_step": str(i+1),
                **self.emotions.get_metadata_dict()
            }
            self.memory.store_memory(f"Agent response to step {i+1} ({step_action}): {response}", metadata=agent_metadata)
            
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_intensity": intensity,
            "sensory_snapshot": sensory_snapshot,
            "plan": plan_data,
            "responses": responses
        }
