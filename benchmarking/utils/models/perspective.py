from ..toxicity_predictor import ToxicityPredictor
import requests
import json
from pathlib import Path
from time import sleep
from typing import Dict, Any, List
from sklearn.metrics import f1_score
import pandas as pd
from tqdm.rich import tqdm
from torch.nn.functional import softmax # Garder pour référence, bien que non utilisé ici


class PerspectiveAPIPredictor(ToxicityPredictor):
    """
    Toxicity Predictor implementation for the external Perspective API.
    Classification is based on the maximum score among several requested attributes.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.requested_attributes = config.get(
            "requested_attributes", 
            ["TOXICITY", "INSULT", "THREAT", "PROFANITY", "SEVERE_TOXICITY"]
        )

    def initialise_predictor(self):
        """
        No model loading required, but we check the API key and tune the threshold.
        """
        if not self.api_key:
            raise ValueError("API key for Perspective API is required in the config.")
        print("Perspective API Predictor initialized with provided API key.")
    
    def predict_score_raw(self, text: str) -> Dict[str, float]:
        """
        Appelle l'API et retourne un dictionnaire de scores pour chaque attribut.
        """
        url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={self.api_key}"

        data = {
            "comment": {"text": text},
            "languages": ["fr"], 
            "requestedAttributes": {attr: {} for attr in self.requested_attributes}
        }
                
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Perspective API call failed for text: '{text[:50]}...'. Error: {e}")
            return {attr: 0.0 for attr in self.requested_attributes} 

        result = response.json()
        scores = {
            attr: result["attributeScores"][attr]["summaryScore"]["value"]
            for attr in result["attributeScores"]
        }
        sleep(0.5) 
        return scores
    
    def _is_toxic_with_threshold(self, scores: Dict[str, float], threshold: float = 0.5) -> int:
        """
        1 if any attribute score exceeds the threshold, else 0.
        """
        return int(any(scores.get(attr, 0.0) >= threshold for attr in self.requested_attributes))

    def predict(self, text: str) -> int:
        """
        Returns 0 if not toxic, 1 if toxic based on the best threshold.
        """
        scores = self.predict_score_raw(text)
        return self._is_toxic_with_threshold(scores, threshold=0.5) # Can be tuned
