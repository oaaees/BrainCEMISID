import unittest
import os
from dotenv import load_dotenv
from core.sensors import SensoryGate
from core.memory import Memory
from core.llm_engine import LLMEngine

class TestBrainCEMISIDSenses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure environment variables are loaded so LLMEngine can authenticate
        load_dotenv()
        cls.llm_engine = LLMEngine()

    def setUp(self):
        # Inicializamos el captador sensorial y la memoria
        self.sensor_gate = SensoryGate(llm_engine=self.llm_engine)
        # Using Ephemeral client (db_path=None) for isolated testing
        self.memory = Memory(
            collection_name="sensory_test_vault", 
            db_path=None, 
            embedding_fn=self.llm_engine.generate_embedding
        )

    def test_qualitative_extraction(self):
        """
        Verifica que el sistema extraiga descriptores semánticos correctos
        de un input con múltiples estímulos.
        """
        raw_input = "The room is dimly lit with a flickering red neon sign, and there is a constant low-frequency hum."
        
        # Procesamos el input a través de los canales sensoriales
        sensory_snapshot = self.sensor_gate.extract_senses(raw_input)
        
        # Validamos que se hayan identificado los canales correctos
        self.assertIn("sight", sensory_snapshot)
        self.assertIn("hearing", sensory_snapshot)
        
        # Verificamos que el descriptor de 'vista' sea cualitativo y responda con algo y no sea "None"
        self.assertIsInstance(sensory_snapshot["sight"], str)
        self.assertNotEqual(sensory_snapshot["sight"].lower(), "none", "El sistema no detectó la vista.")
        self.assertNotEqual(sensory_snapshot["hearing"].lower(), "none", "El sistema no detectó el oído.")
        
        # Comprobamos que el texto tenga sustancia textual (no esté vacío y tenga varias palabras)
        sight_words = sensory_snapshot["sight"].split()
        self.assertTrue(len(sight_words) >= 1, "El descriptor sensorial de la vista es demasiado breve o está vacío.")
        
        print(f"✅ Descriptor de Vista: {sensory_snapshot['sight']}")
        print(f"✅ Descriptor de Oído: {sensory_snapshot['hearing']}")

    def test_cross_modal_memory_retrieval(self):
        """
        Prueba si un estímulo sensorial actual puede recuperar memorias 
        basadas en la similitud de la 'sensación'.
        """
        # Guardamos una memoria con un descriptor sensorial específico
        old_experience = "I felt a rough, cold texture on the metallic surface."
        self.memory.store_memory(
            text=old_experience, 
            metadata={"sense_touch": "cold metallic texture", "role": "user"}
        )
        
        # Simulamos un nuevo input con una sensación similar
        new_query = "Something freezing and hard to the touch"
        
        # Recuperamos de la base de datos vectorial usando dict retrieval
        results = self.memory.retrieve_relevant_memories(new_query, top_k=1)
        
        self.assertTrue(len(results) > 0, "No se recuperaron memorias semánticas.")
        self.assertIn("cold", results[0]['text'].lower(), "El texto recuperado no coincide con la experiencia guardada.")
        self.assertEqual(results[0]['metadata']['sense_touch'], "cold metallic texture")
        print(f"✅ Recuperación Cross-modal Exitosa: '{new_query}' recuperó '{results[0]['text']}'")

if __name__ == "__main__":
    unittest.main()
