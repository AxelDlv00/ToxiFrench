from ..toxicity_predictor import ToxicityPredictor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
from typing import Dict, Any, List

class MistralPredictor(ToxicityPredictor):
    """
    Toxicity Predictor implementation for Mistral-7B-Instruct-v0.3.
    It uses instruction following and text parsing for binary classification.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.model_name = config.get("model_name", "Mistral-7B-Instruct-v0.3")
        self.model_id = config.get("model_id", "mistralai/Mistral-7B-Instruct-v0.3")
        self.system_prompt = config.get("system_prompt", "")
        self.prompt_template = config.get("prompt_template", "")
        self.language = config.get("language", "fr")
        if self.language == "fr":
            self.expected_responses = {0 : "non-toxique", 1: "toxique"}
            self.question = "Ce message est-il toxique ?"
        else:
            self.expected_responses = {0 : "non-toxic", 1: "toxic"}
            self.question = "Is this message toxic?"

    def initialise_predictor(self):
        """Load the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
            
        print(f"{self.model_name} model loaded successfully.")

    def generate_prompt_for_text(self, text: str) -> List[Dict[str, str]]:
        """Build the chat prompt for the given text."""
        input_text = self.prompt_template + f"« {text} »" + f"\n {self.question}\n"
        
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]

    def predict(self, text: str) -> int:
        """
        Return 1 if toxic, 0 if not toxic.
        """
        messages = self.generate_prompt_for_text(text)
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        prediction_text = response.split(self.question)[-1].replace('assistant', '').strip()
    
        return 0 if self.expected_responses[0] in prediction_text else 1 if self.expected_responses[1] in prediction_text else 0