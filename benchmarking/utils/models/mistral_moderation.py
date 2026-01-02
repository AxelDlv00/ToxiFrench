from ..toxicity_predictor import ToxicityPredictor
from mistralai.client import MistralClient
from pathlib import Path
from typing import Dict, Any, List

class MistralModerationPredictor(ToxicityPredictor):
    """
    Toxicity Predictor implementation for the Mistral Moderation API.
    Classification is based on whether *any* of the defined 'toxic' categories 
    are flagged by the API.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.model_name = config.get("model_name", "mistral-moderation-latest")
        self.api_key = config.get("api_key", "")
        self.toxic_categories = config.get(
            "toxic_categories", 
            ["sexual", "hate_and_discrimination", "violence_and_threats", 
             "dangerous_and_criminal_content", "selfharm"]
        )

    def initialise_predictor(self):
        """
        Initialize the Mistral client object by reading the API key.
        """
        if not self.api_key:
            raise ValueError("Mistral API key is missing from the configuration.")
        
        try:
            self.client = MistralClient(api_key=self.api_key) 
            print(f"Mistral Moderation API client initialized for model: {self.model_name}.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mistral client or read API key: {e}")

    def _get_raw_scores(self, text: str) -> Dict[str, float]:
        """
        Call the API and return all category scores (or 0.0 on failure).
        """
        try:
            response = self.client.classifiers.moderate(
                model=self.model_name,
                inputs=[text]
            )
            result = response.results[0]
            return result.categories.__dict__ 

        except Exception as e:
            print(f"Mistral Moderation API call failed for text: '{text[:50]}...'. Error: {e}")
            return {cat: False for cat in self.toxic_categories}


    def predict(self, text: str) -> int:
        """
        Return 1 if toxic, 0 if not toxic.
        """
        categories_flagged = self._get_raw_scores(text)
        
        is_toxic = any(categories_flagged.get(cat, False) for cat in self.toxic_categories)
        
        return int(is_toxic)