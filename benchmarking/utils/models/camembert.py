from ..toxicity_predictor import ToxicityPredictor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Union

class CamemBertPredictor(ToxicityPredictor):
    """
    Toxicity Predictor for CamemBERT-based models (e.g., AgentPublic/camembert-base-toxic-fr-user-prompts).
    This model outputs three classes (Non-Toxic, Sensible, Toxic). 
    The 'Sensible' and 'Toxic' classes are combined to return a binary '1' (Toxic) prediction.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model_id = config.get(
            "model_id", 
            "AgentPublic/camembert-base-toxic-fr-user-prompts"
        )

    def initialise_predictor(self):
        """Load the CamemBERT model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device)
            print(f"CamemBERT model ({self.model_id}) loaded successfully on {self.device}.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CamemBERT predictor: {e}")

    def _get_raw_scores(self, text: str) -> Dict[str, float]:
        """
        Helper to get the raw probability scores for the three classes.
        """

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)
        
        non_toxic_prob = probs[0][0].item()
        toxic_prob = 1 - non_toxic_prob

        # We assume that even if labels are changed the convention is 0 = Non-Toxic and the other are merged
        return {
            'Non-Toxic': non_toxic_prob,
            'Toxic': toxic_prob
        }

    def predict(self, text: str) -> int:
        """
        Returns 0 if non-toxic, 1 if toxic (Sensible/Toxic/... combined).
        """
        scores = self._get_raw_scores(text)

        prob_toxic_or_sensible = scores.get("Toxic", 0.0)
        prob_non_toxic = scores.get("Non-Toxic", 0.0)
        
        if prob_toxic_or_sensible > prob_non_toxic:
            return 1 
        else:
            return 0 