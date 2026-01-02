from ..toxicity_predictor import ToxicityPredictor
from openai import OpenAI
from pathlib import Path
from time import sleep
from typing import Dict, Any

class OpenAIModerationPredictor(ToxicityPredictor):
    """
    Toxicity Predictor implementation for the external OpenAI Moderation API.
    The prediction is based on the model's internal 'flagged' status.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.client = None

    def initialise_predictor(self):
        """
        Initialize the OpenAI client by reading the API key.
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is missing from the configuration.")
        
        try:
            self.client = OpenAI(api_key=self.api_key)
            print(f"OpenAI client initialized.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client or read API key: {e}")

    def predict(self, text: str) -> int:
        """
        Returns 0 if not toxic, 1 if toxic based on the 'flagged' status.
        """
        try:
            response = self.client.moderations.create(
                model=self.model_name,
                input=[{"type": "text", "text": text}]
            )
            result = response.results[0]
            sleep(0.5) 
            return int(result.flagged)

        except Exception as e:
            print(f"Prediction failed for text: '{text[:50]}...'. Returning 0.")
            return 0