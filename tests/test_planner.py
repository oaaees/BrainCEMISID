import unittest
import os
from dotenv import load_dotenv
from core.llm_engine import LLMEngine
from core.planner import StrategicPlanner

class TestBrainCEMISIDPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load environment variables for the Gemini Engine
        load_dotenv()
        cls.llm_engine = LLMEngine()

    def setUp(self):
        # Inicializamos el planificador estratégico con el LLM Engine inyectado
        self.planner = StrategicPlanner(llm_engine=self.llm_engine)

    def test_plan_decomposition(self):
        """
        Verifica que el agente descomponga una meta en un formato 
        estructurado (JSON) con pasos y razones.
        """
        goal = "Explore the dark basement to find a power generator."
        cognitive_snapshot = (
            "Emotion: Fear (0.80/1.0)\n"
            "Active Senses: Sight: total darkness, Hearing: metallic scraping"
        )
        
        plan_data = self.planner.decompose_task(goal, cognitive_snapshot)
        
        # Validaciones de estructura principal COALA
        self.assertIn("thought", plan_data, "El plan carece del paso de pensamiento ('thought').")
        self.assertIn("plan", plan_data, "El plan carece de los pasos de acción ('plan').")
        
        plan = plan_data["plan"]
        self.assertIsInstance(plan, list, "Los pasos del plan deben ser una lista.")
        self.assertTrue(0 < len(plan) <= 5, "El planificador no generó pasos, o generó más del máximo de 5 permitidos.")
        
        # Validaciones de contenido neuro-simbólico por paso
        first_step = plan[0]
        self.assertIn("step", first_step)
        self.assertIn("reason", first_step)
        
        print(f"\n✅ Pensamiento COALA: {plan_data['thought']}")
        print(f"✅ Primer Paso Generado (Miedo Alto): {first_step['step']}")
        print(f"   Motivo: {first_step['reason']}")

    def test_emotional_influence_on_planning(self):
        """
        Verifica la deriva conductual: ¿Cambia el plan si el agente 
        pasa de tener miedo a estar curioso?
        """
        goal = "Approach the mysterious glowing object."
        
        # Escenario A: Miedo alto
        fear_snapshot = (
            "Emotion: Fear (0.90/1.0)\n"
            "Active Senses: None"
        )
        plan_data_fear = self.planner.decompose_task(goal, fear_snapshot)
        
        # Escenario B: Curiosidad alta
        curiosity_snapshot = (
            "Emotion: Curiosity (0.90/1.0)\n"
            "Active Senses: None"
        )
        plan_data_curiosity = self.planner.decompose_task(goal, curiosity_snapshot)
        
        # Las acciones iniciales (o la forma de abordarlas) deberían ser distintas
        step_fear = plan_data_fear["plan"][0]['step']
        step_curiosity = plan_data_curiosity["plan"][0]['step']
        
        self.assertNotEqual(step_fear, step_curiosity, 
                         "El plan no muestra deriva conductual basada en emociones.")
        
        print("\n✅ Prueba de Deriva Conductual Pasada exitosamente:")
        print(f"   Paso Inicial (Miedo): {step_fear}")
        print(f"   Paso Inicial (Curiosidad): {step_curiosity}")

if __name__ == "__main__":
    unittest.main()
