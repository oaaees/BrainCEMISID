import unittest
from core.memory import Memory

import uuid

class TestBrainCEMISIDMemory(unittest.TestCase):
    def setUp(self):
        # Inicializa una instancia de memoria para pruebas usando un cliente efímero 
        # (db_path=None) para no contaminar la base de datos principal local.
        # Usa un nombre de colección único para aislar las pruebas.
        self.memory = Memory(collection_name=f"test_vault_{uuid.uuid4().hex}", db_path=None)
        
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

    def test_short_term_memory_limits(self):
        """
        Verifica que add_interaction respeta el límite de max_history
        y elimina las entradas más antiguas correspondientes.
        """
        self.memory.max_history = 3
        
        self.memory.add_interaction("user", "Hello 1")
        self.memory.add_interaction("agent", "Hi 1")
        self.memory.add_interaction("user", "Hello 2")
        self.memory.add_interaction("agent", "Hi 2")
        
        # Debe haber solo 3 interacciones
        self.assertEqual(len(self.memory.history), 3)
        # La más antigua 'Hello 1' (índice 0 cuando era 4, luego se borró)
        self.assertEqual(self.memory.history[0]["content"], "Hi 1")
        self.assertEqual(self.memory.history[-1]["content"], "Hi 2")
        print("✅ Prueba de Límite de Memoria a Corto Plazo: PASADA.")

    def test_get_context_and_clear(self):
        """
        Verifica el formato del contexto devuelto y que la función clear
        limpie la memoria a corto plazo.
        """
        self.memory.add_interaction("user", "How are you?")
        self.memory.add_interaction("assistant", "I am fine.")
        
        context = self.memory.get_context()
        self.assertIn("User: How are you?", context)
        self.assertIn("Assistant: I am fine.", context)
        
        self.memory.clear()
        self.assertEqual(len(self.memory.history), 0)
        self.assertEqual(self.memory.get_context(), "No previous context.")
        print("✅ Prueba de Obtención de Contexto y Limpieza: PASADA.")

    def test_empty_retrieval(self):
        """
        Verifica que la recuperación maneje correctamente una colección vacía.
        """
        results = self.memory.retrieve_relevant_memories("Test query", top_k=2)
        self.assertEqual(len(results), 0)
        
        context = self.memory.retrieve_relevant_context("Test query")
        self.assertEqual(context, "")
        print("✅ Prueba de Recuperación Vacía: PASADA.")

    def test_build_prompt(self):
        """
        Verifica que la construcción del prompt fusione correctamente todas las entradas.
        """
        self.memory.add_interaction("user", "Who was Simon Bolivar?")
        long_term = "- Simon Bolivar was a military and political leader."
        sensory = {"sight": "A historical text", "smell": "None"}
        
        prompt = self.memory.build_prompt(
            new_input="Tell me more about him.",
            long_term_context=long_term,
            current_emotion="Curiosity",
            sensory_snapshot=sensory
        )
        
        self.assertIn("Dominant Emotion: Curiosity", prompt)
        self.assertIn("Sight: A historical text", prompt)
        self.assertNotIn("Smell", prompt)
        self.assertIn("User: Who was Simon Bolivar?", prompt)
        self.assertIn("- Simon Bolivar was a military and political leader.", prompt)
        self.assertIn("User: Tell me more about him.", prompt)
        print("✅ Prueba de Construcción de Prompt: PASADA.")

if __name__ == "__main__":
    unittest.main()
