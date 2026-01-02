from ..toxicity_predictor import ToxicityPredictor
from transformers import pipeline
from typing import Dict, Any

class FrenchToxicityClassifierPredictor(ToxicityPredictor):
    """
    Toxicity Predictor implementation for EIStakovskii/french_toxicity_classifier_plus_v2.
    Uses the Hugging Face pipeline, retrieves the score, and allows for threshold tuning.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.model_name = config.get("model_name", "french_toxicity_classifier_plus_v2")
        self.model_id = config.get("model_id", "EIStakovskii/french_toxicity_classifier_plus_v2")

    def initialise_predictor(self):
        """Load the model via the Hugging Face pipeline."""
        self.model = pipeline(
            "text-classification", 
            model=self.model_id,
        )
        print(f"{self.model_name} model loaded successfully via Hugging Face pipeline.")

    def predict(self, text: str) -> int:
        return self.model(text)[0]["label"] == "toxic"
