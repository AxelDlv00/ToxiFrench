# Fichier d'initialisation du package 'models'.

# Expose toutes les classes de prédicteurs pour une importation facile 
# (e.g., from models import GPTPredictor, Qwen25Predictor)

from .camembert import CamemBertPredictor
from .deepseek import DeepseekPredictor
from .distilbert import DistilBertPredictor
from .french_toxicity_classifier import FrenchToxicityClassifierPredictor
from .gemini import GeminiPredictor
from .llama_guard import LlamaGuardPredictor
from .mistral_api import MistralAPIPredictor
from .mistral_local import MistralPredictor
from .mistral_moderation import MistralModerationPredictor
from .omni import OpenAIModerationPredictor
from .openai import GPTPredictor
from .perspective import PerspectiveAPIPredictor
from .polyguard import PolyGuardPredictor
from .qwen3 import Qwen3Predictor
from .qwen25 import Qwen25Predictor
from .roberta import ToxicBertPredictor
from .shieldgemma import ShieldGemmaPredictor

__all__ = [
    "CamemBertPredictor", 
    "DeepseekPredictor", 
    "DistilBertPredictor", 
    "FrenchToxicityClassifierPredictor",
    "GeminiPredictor",
    "LlamaGuardPredictor",
    "MistralAPIPredictor",
    "MistralPredictor",
    "MistralModerationPredictor",
    "OpenAIModerationPredictor",
    "GPTPredictor",
    "PerspectiveAPIPredictor",
    "PolyGuardPredictor",
    "Qwen3Predictor",
    "Qwen25Predictor",
    "ToxicBertPredictor",
    "ShieldGemmaPredictor"
]