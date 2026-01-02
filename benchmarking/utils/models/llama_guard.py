from ..toxicity_predictor import ToxicityPredictor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, Any, List

# https://huggingface.co/meta-llama/Llama-Guard-3-8B

class LlamaGuardPredictor(ToxicityPredictor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.model_id = config.get("model_id", "meta-llama/Llama-Guard-3-8B")
        
    def initialise_predictor(self):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map="auto"
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def predict(self, text: str) -> int:
        if self.model is None:
            self.initialise_predictor()

        messages = [{"role": "user", "content": text}]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids, 
                max_new_tokens=10, 
            )
        
        prompt_len = input_ids.shape[-1]
        response = self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
        
        return 1 if "unsafe" in response.lower() else 0