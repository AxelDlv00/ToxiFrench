from ..toxicity_predictor import ToxicityPredictor
from openai import OpenAI
import random
from time import sleep
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

class DeepseekPredictor(ToxicityPredictor):
    """
    Toxicity Predictor implementation for Deepseek-Chat via its OpenAI-compatible API.
    Updated to support batched processing via threaded parallelism.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.model_name = config.get("model_name", "deepseek-chat") 
        self.api_key = config.get("api_key", None)
        self.base_url = config.get("base_url", "https://api.deepseek.com")
    
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
        self.delay_base = config.get("delay_base", 2.0)
        # Load the parallel_requests config to define worker count
        self.max_workers = config.get("parallel_requests", 10)

    def initialise_predictor(self):
        """Initialise the Deepseek API client."""
        if not self.api_key:
            raise ValueError("Deepseek API key is missing from the configuration.")
        
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) 
            print(f"Deepseek client initialized for model: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Deepseek client: {e}")

    def generate_messages(self, text: str) -> List[Dict[str, str]]:
        input_text = f"{self.prompt_template} « {text} »\n{self.question}\n"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
    
    def predict(self, text: str) -> int:
        """
        Atomic prediction method with optimized retry logic.
        """
        messages = self.generate_messages(text)
        
        for attempt in range(self.max_retries):
            try:
                # API Call
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=False,
                    temperature=0,  # Ensure deterministic output
                    max_tokens=10
                )
                
                result = response.choices[0].message.content.strip().lower()

                # Parsing logic
                if self.expected_responses[0] in result:
                    return 0
                elif self.expected_responses[1] in result:
                    return 1
                
                # If ambiguous response, logging it but retrying might not help unless temperature > 0
                print(f"Deepseek: Ambiguous response: '{result}'. Defaulting to None.")
                return None
                
            except Exception as e:
                # Exponential backoff only on error
                wait = (self.delay_base ** attempt) + random.uniform(0, 1)
                print(f"Deepseek Error (Attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {wait:.2f}s...")
                sleep(wait) 
                
        print(f"Deepseek: Failed to predict after {self.max_retries} attempts.")
        return None

    def predict_batch(self, texts: List[str]) -> List[int]:
        """
        Processes a list of texts in parallel using ThreadPoolExecutor.
        This matches the 'run_predictions_batched' requirement.
        """
        if not self.client:
            self.initialise_predictor()

        # Executor handles the concurrency, map preserves the order of results
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.predict, texts))
            
        return results