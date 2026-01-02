from ..toxicity_predictor import ToxicityPredictor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Any

class DistilBertPredictor(ToxicityPredictor):
    """
    Toxicity Predictor for citizenlab/distilbert-base-multilingual-cased-toxicity.
    It uses a predefined threshold to convert the continuous toxicity score 
    into a discrete binary prediction (0 or 1).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        
        self.model_name = config.get("model_name", "distilbert-base-multilingual-cased-toxicity")
        self.model_id = config.get("model_id", "citizenlab/" + self.model_name)

    def initialise_predictor(self):
        """Load the model and tokenizer from Hugging Face."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        print(f"{self.model_name} model loaded successfully.")
        
    def predict(self, text: str) -> int:
        """
        Return 0 if not toxic, 1 if toxic based on the predefined threshold.
        """
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        score = torch.sigmoid(logits[0][0]).item()

        return 1 if score > 0.5 else 0