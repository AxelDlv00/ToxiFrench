from ..toxicity_predictor import ToxicityPredictor
from google import genai
from google.api_core import exceptions as genai_errors
from pathlib import Path
from time import sleep
import random
from typing import Dict, Any, List

class GeminiPredictor(ToxicityPredictor):
    """
    Toxicity Predictor implementation for Gemini 2.5 Flash via the Google GenAI API.
    Uses instruction following, text parsing, and built-in retries with backoff.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = None
        self.model_name = config.get("model_name", "gemini-2.5-flash") # or other models
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
        self.delay_base = config.get("delay_base", 2) 

    def initialise_predictor(self):
        """Initialise the Gemini API client."""
        if not self.api_key:
            raise ValueError("Gemini API key is missing from the configuration.")
        
        try:
            self.client = genai.Client(api_key=self.api_key) 
            print(f"Gemini API client initialized for model: {self.model_name}.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client or read API key: {e}")

    def generate_content_prompt(self, text: str) -> str:
        """
        Builds the content prompt by combining system and user prompts.
        """
        input_text = self.system_prompt + "\n" + self.prompt_template + f"« {text} »" + f"\n {self.question}\n"
        return input_text
    
    def predict(self, text: str) -> int:
        """
        Generate the classification.
        """
        input_content = self.generate_content_prompt(text)
        
        for attempt in range(self.max_retries):
            
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=input_content
                )
                
                result = response.text.strip().lower()

                if self.expected_responses[0] in result:
                    return 0
                elif self.expected_responses[1] in result:
                    return 1
                else:
                    print(f"Unexpected response format: '{result}'. Retrying...")
                    return None
                
            except genai_errors.ServiceUnavailable as e:
                wait_time = self.delay_base ** attempt + random.uniform(0, 1)
                if attempt < self.max_retries - 1:
                    sleep(wait_time)
                    continue
                else:
                    return None

            except Exception as e:
                # Handle other errors
                wait_time = self.delay_base ** attempt + random.uniform(0, 1)
                if attempt < self.max_retries - 1:
                    sleep(wait_time)
                    continue
                else:
                    return None
        
        return None