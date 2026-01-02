from ..toxicity_predictor import ToxicityPredictor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Any

class Qwen25Predictor(ToxicityPredictor):
    """
    Toxicity Predictor for Qwen-2.5-3B-Instruct using Few-Shot prompting.
    Returns a discrete 0 or 1 prediction based on parsing the textual output.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.model_name = config.get("model_name", "Qwen2.5-3B-Instruct")
        self.model_id = f'Qwen/{self.model_name}'
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
        """Load the model/tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        print(f"{self.model_name} model loaded successfully.")

    def generate_prompt_for_text(self, text: str) -> List[Dict[str, str]]:
        """Generate the chat prompt for the given text."""
        input_text = self.prompt_template + f"« {text} »" + f"\n {self.question}\n"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]

    def predict(self, text: str) -> int:
        """
        Return 0 if not toxic, 1 if toxic.
        """
        messages = self.generate_prompt_for_text(text)
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=50) 

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)        
        prediction_text = response.split(self.question)[-1].replace('assistant', '').strip()

        return 0 if self.expected_responses[0] in prediction_text.lower() else 1