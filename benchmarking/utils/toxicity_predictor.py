# ╔════════════════════════════════════════════════════════╗
# ║    Abstract base class for all toxicity predictors.    ║
# ╚════════════════════════════════════════════════════════╝

# +--------------------------------------------------------+
# |   It allows to consistently define the interface for   |
# |  different toxicity prediction implementations (from   |
# |         local models to API-based predictors).         |
# +--------------------------------------------------------+

from abc import ABC, abstractmethod

class ToxicityPredictor(ABC):
    """
    Abstract class for Toxicity Predictors.
    """
    
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def initialise_predictor(self):
        """Load the model/tokenizer, initialize the API client, etc."""
        pass
    
    @abstractmethod
    def predict(self, text: str) -> int:
        """Return 0 if not toxic, 1 if toxic."""
        pass