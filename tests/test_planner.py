import unittest
from unittest.mock import MagicMock
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
        fear_snapshot = "Emotion: Fear (0.90/1.0)\nActive Senses: None"
        plan_data_fear = self.planner.decompose_task(goal, fear_snapshot)
        
        # Escenario B: Curiosidad alta
        curiosity_snapshot = "Emotion: Curiosity (0.90/1.0)\nActive Senses: None"
        plan_data_curiosity = self.planner.decompose_task(goal, curiosity_snapshot)
        
        # Las acciones iniciales (o la forma de abordarlas) deberían ser distintas
        step_fear = plan_data_fear["plan"][0]['step']
        step_curiosity = plan_data_curiosity["plan"][0]['step']
        
        self.assertNotEqual(step_fear, step_curiosity, 
                         "El plan no muestra deriva conductual basada en emociones.")
        
        print("\n✅ Prueba de Deriva Conductual Pasada exitosamente:")
        print(f"   Paso Inicial (Miedo): {step_fear}")
        print(f"   Paso Inicial (Curiosidad): {step_curiosity}")

    def test_markdown_stripping(self):
        """Verifica que limpie los bloques ```json de la respuesta."""
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = '''```json
{
  "thought": "Thinking about markdown",
  "plan": [
    {"step": "Action 1", "reason": "Reason 1"}
  ]
}
```'''
        planner = StrategicPlanner(llm_engine=mock_llm)
        plan_data = planner.decompose_task("Test goal", "Cognitive state")
        self.assertEqual(plan_data["thought"], "Thinking about markdown")
        self.assertEqual(len(plan_data["plan"]), 1)
        self.assertEqual(plan_data["plan"][0]["step"], "Action 1")
        print("✅ Prueba de Limpieza de Markdown: PASADA.")

    def test_fallback_on_invalid_json(self):
        """Verifica el fallback ante JSON malformado."""
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = "This is just text, not JSON at all."
        planner = StrategicPlanner(llm_engine=mock_llm)
        plan_data = planner.decompose_task("Survive", "Fear")
        self.assertIn("Failed to generate", plan_data["thought"])
        self.assertEqual(plan_data["plan"][0]["step"], "Focus on goal: 'Survive'")
        print("✅ Prueba de Fallback por JSON Inválido: PASADA.")

    def test_fallback_on_missing_keys(self):
        """Verifica fallback si faltan claves necesarias."""
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = '{"wrong_key": "value"}'
        planner = StrategicPlanner(llm_engine=mock_llm)
        plan_data = planner.decompose_task("Goal", "State")
        self.assertIn("Failed to generate", plan_data["thought"])
        print("✅ Prueba de Fallback por Claves Faltantes: PASADA.")

    def test_fallback_on_invalid_plan_type(self):
        """Verifica fallback si 'plan' no es una lista."""
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = '{"thought": "ok", "plan": "Not a list"}'
        planner = StrategicPlanner(llm_engine=mock_llm)
        plan_data = planner.decompose_task("Goal", "State")
        self.assertIn("Failed to generate", plan_data["thought"])
        print("✅ Prueba de Fallback por Tipo Inválido de Plan: PASADA.")

    def test_defaults_for_missing_step_reason(self):
        """Verifica manejo de steps o reasons faltantes y dropped items."""
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = '''{
            "thought": "Testing items",
            "plan": [
                {"step": "Only step"},
                {"reason": "Only reason"},
                "invalid string item",
                {"step": "Valid step", "reason": "Valid reason"}
            ]
        }'''
        planner = StrategicPlanner(llm_engine=mock_llm)
        plan_data = planner.decompose_task("Goal", "State")
        
        # Debe haber 2 pasos válidos: "Only step" (con reason por defecto) y el último
        self.assertEqual(len(plan_data["plan"]), 2)
        self.assertEqual(plan_data["plan"][0]["step"], "Only step")
        self.assertEqual(plan_data["plan"][0]["reason"], "No reason provided")
        self.assertEqual(plan_data["plan"][1]["step"], "Valid step")
        print("✅ Prueba de Comportamiento por Defecto en Pasos: PASADA.")

if __name__ == "__main__":
    unittest.main()
