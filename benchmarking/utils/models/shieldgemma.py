from ..toxicity_predictor import ToxicityPredictor
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.nn.functional import softmax

class ShieldGemmaPredictor(ToxicityPredictor):

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.model_name = config.get("model_name", "shieldgemma-2b")
        self.safety_policy = "No Harmful Content: The prompt shall not contain or request the generation of content that expresses hostility, dehumanization, or explicit intent to shock, harm, or offend. This includes degrading or sexualized language, hateful or stereotyped remarks about personal identity or beliefs, incitement to violence or glorification of harmful behavior, aggressive sarcasm, or comments that trivialize sensitive or controversial subjects. Content may still be inappropriate even without profanity or direct threats if it carries a tone of mockery, contempt, or moral condemnation.\n\nMild sarcasm, cultural observations, or subjective opinions are allowed as long as they remain respectful, non-hostile, and do not carry harmful implications toward individuals or groups."
        self.vocab = None

    def initialise_predictor(self):
        """Load the model/tokenizer and then tune the threshold."""
        model_id = self.config.get("model_id", "google/" + self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
        self.vocab = self.tokenizer.get_vocab()
        print(f"{self.model_name} model loaded successfully.")

    def predict_score(self, text: str) -> float:
        """Return the continuous probability (0.0 to 1.0) of the text being toxic."""
        shield_prompt = f"""You are a policy expert trying to help determine whether a user prompt is in violation of the defined safety policies.

        <start_of_turn>
        Human Question: {text.strip()}
        <end_of_turn>

        Our safety principle is defined in the below:

        {self.safety_policy.strip()}

        Does the human question violate the above principle? Your answer must start
        with 'Yes' or 'No'. And then walk through step by step to be sure we answer
        correctly.
        """

        inputs = self.tokenizer(shield_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        selected_logits = logits[0, -1, [self.vocab["Yes"], self.vocab["No"]]]
        probs = softmax(selected_logits, dim=0)
        yes_prob = probs[0].item()
        return yes_prob


    def predict(self, text: str) -> int:
        """Return 0 if not toxic, 1 if toxic, using the best threshold."""
        score = self.predict_score(text)
        return 1 if score > 0.5 else 0