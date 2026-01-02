from ..toxicity_predictor import ToxicityPredictor
from mistralai import Mistral
import random
from time import sleep
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

class MistralAPIPredictor(ToxicityPredictor):
    """
    Toxicity Predictor for Mistral (v1.0+) with Rate Limit handling.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.model_name = config.get("model_name", "mistral-small-latest") 
        self.api_key = config.get("api_key", "")
        self.system_prompt = config.get("system_prompt", "")
        self.prompt_template = config.get("prompt_template", "") 
        self.language = config.get("language", "fr")
        
        if self.language == "fr":
            self.expected_responses = {0 : "non-toxique", 1: "toxique"}
            self.question = "Ce message est-il toxique ?"
        else:
            self.expected_responses = {0 : "non-toxic", 1: "toxic"}
            self.question = "Is this message toxic?"

        self.max_retries = config.get("max_retries", 10) # Increased default retries
        self.delay_base = config.get("delay_base", 2.0)
        # Force lower concurrency if not specified to avoid 429s
        self.max_workers = config.get("parallel_requests", 2) 

    def initialise_predictor(self):
        if not self.api_key:
            raise ValueError("Mistral API key is missing.")
        try:
            self.client = Mistral(api_key=self.api_key) 
            print(f"Mistral API client initialized for model: {self.model_name}.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Mistral client: {e}")

    def generate_messages(self, text: str) -> List[Dict[str, str]]:
        input_text = f"{self.prompt_template} « {text} »\n{self.question}\n"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]

    def predict(self, text: str) -> int:
        messages = self.generate_messages(text)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.complete(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=50,
                    temperature=0.0
                )
                
                full_response_text = response.choices[0].message.content.strip().lower()
                
                if self.expected_responses[0] in full_response_text:
                    return 0
                elif self.expected_responses[1] in full_response_text:
                    return 1
                
                print(f"Mistral: Ambiguous response: '{full_response_text}'.")
                return None

            except Exception as e:
                error_str = str(e)
                # Specific handling for Rate Limits (429)
                if "429" in error_str or "Rate limit" in error_str:
                    # Wait much longer for rate limits (5 to 10 seconds + exponential)
                    wait = 5 + (2 ** attempt) + random.uniform(0, 3)
                    print(f"Mistral 429 Hit (Attempt {attempt+1}/{self.max_retries}). Cooling down for {wait:.2f}s...")
                else:
                    # Standard backoff for other errors
                    wait = (self.delay_base ** attempt) + random.uniform(0, 1)
                    print(f"Mistral Error (Attempt {attempt+1}): {e}. Retrying in {wait:.2f}s...")
                
                sleep(wait)
        
        print(f"Mistral: Failed to predict after {self.max_retries} attempts.")
        return None

    def predict_batch(self, texts: List[str]) -> List[int]:
        if not self.client:
            self.initialise_predictor()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.predict, texts))
        return results