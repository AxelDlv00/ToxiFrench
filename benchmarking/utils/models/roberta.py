from ..toxicity_predictor import ToxicityPredictor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ToxicBertPredictor(ToxicityPredictor):
    """
    Toxicity Predictor implementation for unitary/multilingual-toxic-xlm-roberta.
    It uses a direct sequence classification approach and tunes the threshold.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.model_name = config.get("model_name", "multilingual-toxic-xlm-roberta")
        self.model_id = f"unitary/{self.model_name}"

    def initialise_predictor(self):
        """Load the model/tokenizer and then tune the threshold."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)

        print(f"{self.model_name} model loaded successfully.")

    def predict_score(self, text: str) -> float:
        """
        Return the continuous probability (0.0 to 1.0) of the text being toxic.
        This uses the model's logits and the sigmoid function.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        score = torch.sigmoid(logits[0][0]).item()
        return score
    
    def predict(self, text: str) -> int:
        """
        Return 0 if not toxic, 1 if toxic, using the probability score and the 
        optimized best_threshold. This fulfills the ABC requirement.
        """
        score = self.predict_score(text)
        return 1 if score > 0.5 else 0