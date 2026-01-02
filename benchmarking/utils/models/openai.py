from ..toxicity_predictor import ToxicityPredictor
from openai import OpenAI
from pathlib import Path
from time import sleep
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

class GPTPredictor(ToxicityPredictor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.model_name = config.get("model_name", "gpt-4o-mini")
        self.api_key = config.get("api_key", None)
        self.system_prompt = config.get("system_prompt", "")
        self.prompt_template = config.get("prompt_template", "") 
        self.language = config.get("language", "fr")
        
        if self.language == "fr":
            self.expected_responses = {0 : "non-toxique", 1: "toxique"}
            self.question = "Ce message est-il toxique ?"
        else:
            self.expected_responses = {0 : "non-toxic", 1: "toxic"}
            self.question = "Is this message toxic?"
            
        self.max_retries = config.get("max_retries", 5)
        self.delay_base = config.get("delay_base", 0.5)
        self.max_workers = config.get("parallel_requests", 10)

    def initialise_predictor(self):
        if not self.api_key:
            raise ValueError("OpenAI API key is missing.")
        try:
            self.client = OpenAI(api_key=self.api_key)
            print(f"OpenAI client initialized for model: {self.model_name} with api key : {self.api_key[:8]}****{self.api_key[-4:]}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")

    def generate_messages(self, text: str) -> List[Dict[str, str]]:
        input_text = f"{self.prompt_template} « {text} »\n{self.question}\n"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
    
    def predict(self, text: str) -> int:
        """Méthode unitaire avec retry logic"""
        messages = self.generate_messages(text)
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=10 
                )                
                result = response.choices[0].message.content.strip().lower()
                if self.expected_responses[0] in result:
                    return 0
                elif self.expected_responses[1] in result:
                    return 1
                else:
                    return result
            except Exception as e:
                sleep(self.delay_base * (2 ** attempt))
                print(f"DEBUG ERROR: {type(e).__name__}: {e}")
                continue 
        return "Error"

    def predict_batch(self, texts: List[str]) -> List[int]:
        """
        Surcharge de la méthode batch pour utiliser le parallélisme réseau.
        """
        if not self.client:
            self.initialise_predictor()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.predict, texts))
            
        return results