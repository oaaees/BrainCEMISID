import unittest
from core.memory import Memory

class TestBrainCEMISIDMemory(unittest.TestCase):
    def setUp(self):
        # Inicializa una instancia de memoria para pruebas usando un cliente efímero 
        # (db_path=None) para no contaminar la base de datos principal local.
        self.memory = Memory(collection_name="test_cognitive_vault", db_path=None)
        
    def test_sensory_emotional_retrieval(self):
        """
        Verifica que un recuerdo guardado con un sentido y emoción específicos
        pueda ser recuperado mediante una consulta semántica.
        """
        # 1. Definimos una experiencia multisensorial
        sensory_input = "The smell of fresh coffee in the morning at the ULA campus."
        metadata = {
            "sense": "smell",
            "emotion": "nostalgia",
            "location": "Mérida"
        }
        
        # 2. Guardamos en la arquitectura
        self.memory.store_memory(sensory_input, metadata)
        
        # 3. Consultamos por algo relacionado (no idéntico)
        query = "A pleasant aroma during university hours"
        results = self.memory.retrieve_relevant_memories(query, top_k=1)
        
        # 4. Validaciones (Assertions)
        self.assertTrue(len(results) > 0, "El sistema de recuperación no devolvió nada.")
        self.assertIn("coffee", results[0]['text'].lower(), "La recuperación semántica falló.")
        self.assertEqual(results[0]['metadata']['emotion'], "nostalgia", "El estado emocional se perdió.")
        print(f"✅ Prueba de Memoria Sensorial: PASADA. Recuperado: {results[0]['text']}")

    def test_emotional_drift_filtering(self):
        """
        Verifica si podemos filtrar recuerdos basados exclusivamente en la emoción,
        clave para la deriva conductual del proyecto.
        """
        self.memory.store_memory("I feel very tired after the exam", {"emotion": "stress"})
        self.memory.store_memory("I won a scholarship today!", {"emotion": "joy"})
        
        # Recuperamos solo lo relacionado con 'joy'
        joyful_memories = self.memory.retrieve_by_emotion("joy")
        
        self.assertTrue(any("scholarship" in m['text'] for m in joyful_memories))
        self.assertFalse(any("tired" in m['text'] for m in joyful_memories))
        print("✅ Prueba de Filtro Emocional: PASADA.")

if __name__ == "__main__":
    unittest.main()
