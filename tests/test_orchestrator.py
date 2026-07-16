"""
Tests for the BrainCEMISID Orchestrator with Single-Pass Execution.
Uses mock LLM to verify the orchestrator calls generate_response exactly ONCE
and produces the expected output structure.
"""
import unittest
from unittest.mock import MagicMock, patch, call
import uuid

from core.orchestrator import BrainCEmisidOrchestrator


class TestOrchestratorSinglePass(unittest.TestCase):
    """Verifies the single-pass execution model eliminates multi-call repetition."""
    
    def setUp(self):
        """Set up orchestrator with fully mocked LLM engine."""
        # We patch LLMEngine at the module level so the orchestrator uses our mock
        self.mock_llm = MagicMock()
        
        # Default mock responses for the various LLM calls
        # 1. Sensory extraction (JSON)
        self.mock_llm.generate_response.side_effect = self._mock_responses()
        self.mock_llm.generate_embedding.return_value = [0.1] * 384  # Fake 384-dim vector
        
    def _mock_responses(self):
        """Generator for sequential mock LLM responses."""
        responses = [
            # 1. Emotional delta calculation
            '{"Joy": 0.0, "Sadness": 0.1, "Anger": 0.0, "Fear": 0.3, "Surprise": 0.1}',
            # 2. Planner decomposition
            '{"thought": "Fear drives caution", "plan": [{"step": "Scan environment", "reason": "Safety first"}, {"step": "Take action", "reason": "Proceed carefully"}]}',
            # 3. SINGLE consolidated execution response
            'I carefully scanned the environment. The area appears safe. I am now proceeding with caution to take action.'
        ]
        for r in responses:
            yield r
    
    def test_single_llm_call_for_execution(self):
        """
        Verifies that process_frame calls generate_response exactly 3 times:
        1x for emotions, 1x for planning, 1x for execution (NOT 1x per step).
        """
        with patch('core.orchestrator.LLMEngine', return_value=self.mock_llm):
            orchestrator = BrainCEmisidOrchestrator(
                collection_name=f"test_{uuid.uuid4().hex}", 
                db_path=None
            )
        
        result = orchestrator.process_frame(
            narrative="A dark corridor stretches ahead.",
            goal="Explore the corridor safely",
            external_sensory_input={
                "sight": "darkness",
                "hearing": "silence",
                "smell": "None",
                "touch": "cold air",
                "taste": "None"
            }
        )
        
        # Should be exactly 3 calls: emotion + planner + ONE execution
        self.assertEqual(
            self.mock_llm.generate_response.call_count, 3,
            f"Expected 3 LLM calls (emotion, planner, execution), got {self.mock_llm.generate_response.call_count}"
        )
        print("✅ Single-pass execution: exactly 3 LLM calls (emotion, planner, 1x execution)")
    
    def test_output_structure(self):
        """Verifies the output dictionary has all expected keys."""
        with patch('core.orchestrator.LLMEngine', return_value=self.mock_llm):
            orchestrator = BrainCEmisidOrchestrator(
                collection_name=f"test_{uuid.uuid4().hex}", 
                db_path=None
            )
        
        result = orchestrator.process_frame(
            narrative="Test narrative",
            goal="Test goal",
            external_sensory_input={
                "sight": "light", "hearing": "None", "smell": "None",
                "touch": "None", "taste": "None"
            }
        )
        
        expected_keys = ["dominant_emotion", "emotion_intensity", "sensory_snapshot", 
                         "plan", "responses", "telemetry"]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key '{key}' in output")
        
        # Responses should be a list with exactly 1 consolidated entry
        self.assertEqual(len(result["responses"]), 1, 
                         "Should have exactly 1 consolidated response, not N")
        self.assertEqual(result["responses"][0]["step"], "consolidated_execution")
        
        print("✅ Output structure verified: all keys present, 1 consolidated response")
    
    def test_memory_stores_two_entries_per_frame(self):
        """Verifies memory stores exactly 2 entries per frame (1 user + 1 agent)."""
        with patch('core.orchestrator.LLMEngine', return_value=self.mock_llm):
            orchestrator = BrainCEmisidOrchestrator(
                collection_name=f"test_{uuid.uuid4().hex}", 
                db_path=None
            )
        
        result = orchestrator.process_frame(
            narrative="Test",
            goal="Test goal",
            external_sensory_input={
                "sight": "None", "hearing": "None", "smell": "None",
                "touch": "None", "taste": "None"
            }
        )
        
        # Short-term memory should have exactly 2 entries
        self.assertEqual(len(orchestrator.memory.history), 2, 
                         f"Expected 2 history entries, got {len(orchestrator.memory.history)}")
        self.assertEqual(orchestrator.memory.history[0]["role"], "user")
        self.assertEqual(orchestrator.memory.history[1]["role"], "agent")
        
        # Long-term memory (ChromaDB) should have exactly 2 entries
        self.assertEqual(orchestrator.memory.collection.count(), 2,
                         f"Expected 2 ChromaDB entries, got {orchestrator.memory.collection.count()}")
        
        print("✅ Memory: exactly 2 entries stored per frame (1 user, 1 agent)")


class TestMemoryKeyFacts(unittest.TestCase):
    """Tests for the new key fact storage and retrieval."""
    
    def setUp(self):
        from core.memory import Memory
        self.memory = Memory(collection_name=f"test_kf_{uuid.uuid4().hex}", db_path=None)
    
    def test_store_and_retrieve_key_fact(self):
        """Verifies that key facts can be stored and retrieved with priority filtering."""
        self.memory.store_key_fact("Temperature below 5C means do NOT use Aux Generator")
        self.memory.store_memory("Some normal memory about the weather", {"role": "agent"})
        
        # Retrieve key facts only
        key_facts = self.memory.retrieve_key_facts("temperature generator", top_k=5)
        
        self.assertIn("Temperature below 5C", key_facts)
        self.assertIn("⚠", key_facts)  # Priority marker
        self.assertNotIn("normal memory", key_facts)  # Should not include non-critical
        
        print("✅ Key fact stored and retrieved with priority filtering")
    
    def test_empty_key_facts(self):
        """Verifies empty return when no key facts exist."""
        result = self.memory.retrieve_key_facts("anything")
        self.assertEqual(result, "")
        print("✅ Empty key facts handled correctly")


if __name__ == "__main__":
    unittest.main()
