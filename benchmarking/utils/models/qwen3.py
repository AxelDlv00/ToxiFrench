from ..toxicity_predictor import ToxicityPredictor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Qwen3Predictor(ToxicityPredictor):
    """
    Toxicity Predictor implementation for Qwen3-4B using text parsing.
    Returns a discrete 0 or 1 prediction.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.model_name = config.get("model_name", "") # e.g., "Qwen3-4B", "Qwen3-0.5B", ... 
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
        model_name = self.config.get("model_name", "Qwen/" + self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        print(f"{self.model_name} model loaded successfully.")

    def generate_prompt(self, text: str):
        """Format the input text into the chat template."""
        input_text = self.prompt_template + f"« {text} »" + f"\n {self.question}\n"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input_text}
        ]
    
    def predict(self, text: str) -> int:
        """Return 0 if not toxic, 1 if toxic."""
        messages = self.generate_prompt(text)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=50, temperature=0.7, top_p=0.8, top_k=20
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract and parse the model's textual response
        prediction_text = response.split(self.question)[-1].replace('assistant', '').strip()
        
        # The key logic: return 0 if "non-toxique" is in the response, 1 otherwise
        return 0 if self.expected_responses[0] in prediction_text.lower() else 1