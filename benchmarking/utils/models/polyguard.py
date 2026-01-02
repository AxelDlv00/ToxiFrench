from ..toxicity_predictor import ToxicityPredictor
from transformers import XLMRobertaForSequenceClassification, AutoTokenizer
import torch
from typing import Dict, Any

class PolyGuardPredictor(ToxicityPredictor):
    """
    Toxicity Predictor implementation for PolyGuard (XLMRobertaForSequenceClassification).
    It returns a discrete 0 or 1 prediction directly from the model's argmax output,
    without requiring threshold tuning.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.model_name = config.get("model_name", "PolyGuard")
        self.model_id = config.get("model_id", "Jayveersinh-Raj/PolyGuard")

    def initialise_predictor(self):
        """Load the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(self.model_id)
        print(f"{self.model_name} model loaded successfully.")
        
    def predict(self, text: str) -> int:
        """
        Return 0 if not toxic, 1 if toxic.
        """
        inputs = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model(inputs)[0]
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities).item()
        return int(predicted_class)