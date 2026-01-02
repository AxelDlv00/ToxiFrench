from ..toxicity_predictor import ToxicityPredictor
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import re
import sys
root_path = Path(__file__).resolve().parents[3] 
sys.path.append(str(root_path))
handler_path = root_path / "training" / "FineTuning"
sys.path.append(str(handler_path))

from training.FineTuning.finetuning_handler import QLoRAModelHandler

class ToxiFrenchPredictor(ToxicityPredictor):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.handler: Optional[QLoRAModelHandler] = None
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.prompt_template = config.get("prompt_template", "Message:\n<TEXT>\n\nAnalyse:\n") # <think>\n")
        self.proxy_address = config.get("proxy_address", None)
        self.generation_params = config.get("generation_params", {})

    def initialise_predictor(self):
        bnb_params = self.config.get("bnb_params", {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_quant_type": "nf4"
        })

        self.handler = QLoRAModelHandler(
            model_name=self.config.get("model_name"),
            local_checkpoint=self.config.get("local_checkpoint"),
            proxy_address=self.proxy_address,
            mode="inference",
            bnb_config_kwargs=bnb_params
        )
    
    def _clean_label(self, label):
        pattern = r"(oui|non)\s*<\|im_end\|>"
        label_copy = re.search(pattern, label)
        if label_copy:
            label_copy = label_copy.group(1) 
        else:
            label_copy = "Error" 
        return label_copy

    def predict(self, text: str) -> int:
        if not self.handler:
            raise RuntimeError("Predictor not initialized. Call 'initialise_predictor' first.")

        formatted_prompt = self.prompt_template.replace("<TEXT>", text.strip())
        output_text = self.handler.generate_text(formatted_prompt, **self.generation_params)
        return self._extract_label(output_text)

    def predict_batch(self, texts: List[str], max_retries: int = 3) -> List[int]:
        if not self.handler:
            self.initialise_predictor()

        results = [-2] * len(texts)
        pending_indices = list(range(len(texts)))

        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"Retry attempt {attempt} for {len(pending_indices)} pending items.")
            if not pending_indices:
                break  
            current_prompts = [self.prompt_template.replace("<TEXT>", texts[i].strip()) for i in pending_indices]            
            raw_outputs = self.handler.generate_batch(
                current_prompts,
                **self.generation_params
            )

            new_pending_indices = []
            
            for i, raw_out in enumerate(raw_outputs):
                original_index = pending_indices[i]
                label = self._extract_label(raw_out)

                if label != -1:
                    results[original_index] = label
                else:
                    if attempt < max_retries:
                        new_pending_indices.append(original_index)
                    else:
                        results[original_index] = -1 
            
            pending_indices = new_pending_indices

        return results
    
    def _extract_label(self, output: str) -> int:
        output_label = self._clean_label(output)
        if output_label == "oui":
            return 1
        elif output_label == "non":
            return 0
        else:
            return -1 