"""
Core Orchestrator module. 
Encapsulates all modules (Sensors, Emotions, Memory, Planner, LLM) into a single cohesive entity 
for simple interactions and targeted simulation running.
"""
import time
import re
import json
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
    def __init__(self, collection_name: str = "brain_memory", db_path: str = "./chroma_db", decay_rate: float = 0.15, ablation_mode: str = None):
        """
        Initializes and links all cognitive modules.
        """
        self.llm_engine = LLMEngine()
        self.frame_counter = 0  # Track which frame we're on for memory tagging
        self.ablation_mode = ablation_mode
        
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
        Uses a SINGLE-PASS execution model: the planner decomposes the goal,
        but the plan steps become instructions within ONE consolidated LLM call
        (instead of multiple calls that cause repetition).
        
        Args:
            narrative: The contextual text or event happening in this frame.
            goal: The agent's immediate objective.
            external_sensory_input: Optional pre-parsed sensory data. If None, the SensoryGate will extract it from the narrative.
            
        Returns:
            A dictionary containing the full state, plan, and final generated responses.
        """
        telemetry = {}
        self.frame_counter += 1
        current_frame = self.frame_counter
        
        # 1. Sensory processing
        t0 = time.perf_counter()
        if external_sensory_input:
            sensory_snapshot = external_sensory_input
        else:
            sensory_snapshot = self.sensory_gate.extract_senses(narrative)
        telemetry['sensory_latency_ms'] = (time.perf_counter() - t0) * 1000
            
        # 2. Emotional Shift
        t0 = time.perf_counter()
        if self.ablation_mode == 'no_emotion':
            dominant_emotion, intensity = "Neutral", 0.0
            personality_string = ""
        else:
            self.emotions.shift_emotion(sensory_snapshot, narrative)
            dominant_emotion, intensity = self.emotions.get_dominant_emotion()
            personality_string = self.emotions.get_personality_string()
        telemetry['emotion_latency_ms'] = (time.perf_counter() - t0) * 1000
        
        # Prepare cognitive snapshot string for the planner
        cognitive_snapshot = f"Emotion: {dominant_emotion} ({intensity:.2f}/1.0)\n"
        active_senses = [f"{k.capitalize()}: {v}" for k, v in sensory_snapshot.items() if v.lower() != 'none']
        if active_senses:
            cognitive_snapshot += f"Active Senses: {', '.join(active_senses)}"
        else:
            cognitive_snapshot += "Active Senses: None"
            
        # 3. Strategic Planning
        t0 = time.perf_counter()
        plan_data = self.planner.decompose_task(goal, cognitive_snapshot)
        telemetry['planner_latency_ms'] = (time.perf_counter() - t0) * 1000
        
        # 4. SINGLE-PASS Execution (consolidated LLM call)
        t0 = time.perf_counter()
        num_steps = len(plan_data['plan'])
        print(f"  [Orchestrator] Executing {num_steps} plan steps in single pass...")
        
        # Build plan steps as structured instructions
        plan_instructions = "\n".join([
            f"  Phase {i+1}: {step_info['step']} (Reason: {step_info['reason']})"
            for i, step_info in enumerate(plan_data['plan'])
        ])
        
        # Retrieve long-term context ONCE using the goal
        print(f"  [Memory] Retrieving context for goal...")
        if self.ablation_mode == 'no_memory':
            long_term_context = ""
            key_facts = []
        else:
            long_term_context = self.memory.retrieve_relevant_context(goal, top_k=3)
            key_facts = self.memory.retrieve_key_facts(goal, top_k=3)
        
        # Build ONE consolidated prompt
        consolidated_input = (
            f"You are an office worker living through a stressful Monday morning. "
            f"You must respond IN CHARACTER as this person — not as a helpful AI.\n\n"
            f"WHAT IS HAPPENING RIGHT NOW (Frame {current_frame}):\n{narrative}\n\n"
            f"YOUR OBJECTIVE: {goal}\n\n"
            f"YOUR INTERNAL PLAN:\n{plan_instructions}\n\n"
            f"CRITICAL RULES:\n"
            f"- Respond naturally as the character. If you need to speak to someone, write the actual dialogue.\n"
            f"- Reference ONLY details from 'WHAT IS HAPPENING RIGHT NOW' above.\n"
            f"- IGNORE any details from your memories that contradict or do not appear in the current narrative.\n"
            f"- Do NOT mention blinking lights, sirens, or sensory details unless they appear in the CURRENT narrative.\n"
            f"- Do NOT describe what you 'will do' — actually do it."
        )
        
        prompt = self.memory.build_prompt(
            new_input=consolidated_input, 
            long_term_context=long_term_context,
            key_facts=key_facts,
            current_emotion=personality_string, 
            sensory_snapshot=sensory_snapshot
        )
        
        # ONE LLM call instead of N calls
        response = self.llm_engine.generate_response(prompt)
        
        # Store as a single consolidated interaction
        responses = [{"step": "consolidated_execution", "llm_output": response}]
        
        # Memory Flow Logging (one entry, not N)
        if self.ablation_mode != 'no_memory':
            print(f"  [Memory] Storing consolidated interaction...")
            self.memory.add_interaction(role="user", content=f"Goal: {goal} | Plan: {plan_instructions}")
            self.memory.add_interaction(role="agent", content=response)
        else:
            print(f"  [Memory] Ablation: Wiping short-term interaction history.")
            self.memory.clear()
        
        # Vectorize consolidated interaction with frame timestamps
        user_metadata = {
            "role": "user",
            "frame": str(current_frame),
            "plan_steps": str(num_steps),
            **self.emotions.get_metadata_dict(),
            **{f"sense_{k}": v for k, v in sensory_snapshot.items() if v.lower() != 'none'}
        }
        self.memory.store_memory(
            f"[Frame {current_frame}] Goal '{goal}' with plan: {plan_instructions}", 
            metadata=user_metadata
        )
        
        agent_metadata = {
            "role": "agent",
            "frame": str(current_frame),
            "plan_steps": str(num_steps),
            **self.emotions.get_metadata_dict()
        }
        self.memory.store_memory(
            f"[Frame {current_frame}] Agent response to goal '{goal}': {response}", 
            metadata=agent_metadata
        )
        
        # Extract and store key facts from the narrative for future recall
        if self.ablation_mode != 'no_memory':
            self._extract_key_facts(narrative, goal)
        
        print(f"  [Orchestrator] Single-pass execution complete.")
        telemetry['execution_latency_ms'] = (time.perf_counter() - t0) * 1000
        telemetry['total_latency_ms'] = sum(telemetry.values())
        
        return {
            "dominant_emotion": dominant_emotion,
            "emotion_intensity": intensity,
            "sensory_snapshot": sensory_snapshot,
            "plan": plan_data,
            "responses": responses,
            "telemetry": telemetry
        }

    def _extract_key_facts(self, narrative: str, goal: str):
        """Extracts critical facts from the narrative and stores them in LTM via ChromaDB."""
        prompt = (
            "Extract key facts from this text that might be important to remember later. "
            "Return ONLY a JSON array of short fact strings (max 5 facts). "
            "Focus on: names, numbers, codes, rules, deadlines, decisions.\n\n"
            f"Text: {narrative}\n"
            f"Goal: {goal}\n\n"
            'Example: ["Server passcode is 882-X9", "Do not sync if temp > 28C"]\n'
            "Return just the JSON array, no markdown."
        )
        try:
            response = self.llm_engine.generate_response(prompt)
            clean = re.sub(r'```(?:json)?|```', '', response).strip()
            facts = json.loads(clean)
            if isinstance(facts, list):
                for fact in facts[:3]:
                    self.memory.store_key_fact(str(fact))
        except Exception as e:
            print(f"  [KeyFacts] Extraction failed (non-critical): {e}")
