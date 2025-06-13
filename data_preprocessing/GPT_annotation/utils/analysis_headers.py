from dataclasses import dataclass
from typing import Dict
from pathlib import Path
import os

# === GLOBAL VARIABLES ===
ROOT = Path("../..") / "data" / "headers_prompts"

# === UTILS ===
def safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing prompt file: {path}")

# === PromptType Class ===

@dataclass
class PromptType:
    name: str
    system_prompt: str
    start_prompt: str
    end_prompt: str

    def create_prompt(self, fields: Dict[str, str]) -> str:
        """Build prompt with fields like {'message': '...', 'tone': '...'}"""
        prompt = self.start_prompt
        for key, value in fields.items():
            prompt += f"**{key}** :\n« {value} »\n\n"
        return prompt + self.end_prompt

# === PROMPT TYPES ===

explication_prompt = PromptType(
    name="Explication",
    system_prompt=safe_read(ROOT / "explication.txt"),
    start_prompt="Voici les informations à analyser :\n\n",
    end_prompt="Ta tâche est de reformuler, résumer et expliquer ce message de manière claire et neutre.\n\n"
)

ton_prompt = PromptType(
    name="Ton",
    system_prompt=safe_read(ROOT / "ton.txt"),
    start_prompt="Voici les informations à analyser :\n\n",
    end_prompt="Ta tâche est d’identifier de façon nuancée les tons présents dans ce message.\n\n"
)

intention_prompt = PromptType(
    name="Intention",
    system_prompt=safe_read(ROOT / "intention.txt"),
    start_prompt="Voici les informations à analyser :\n\n",
    end_prompt="Ta tâche est d’identifier les intentions principales du message de manière nuancée.\n\n"
)

categorie_prompt = PromptType(
    name="Catégorie",
    system_prompt=safe_read(ROOT / "categorie.txt"),
    start_prompt="Voici les informations à analyser :\n\n",
    end_prompt="Ta tâche est d’identifier les catégories correspondant au message de manière nuancée.\n\n"
)

taxonomie_prompt = PromptType(
    name="Labels selon la taxonomie",
    system_prompt=safe_read(ROOT / "taxonomie.txt"),
    start_prompt="Voici les informations à analyser :\n\n",
    end_prompt="Ta tâche est d’identifier les labels appropriés selon la taxonomie donnée.\n\n"
)

toxicite_score_prompt = PromptType(
    name="Score de toxicité",
    system_prompt=safe_read(ROOT / "bareme.txt"),
    start_prompt="Voici les informations à analyser :\n\n",
    end_prompt="Ta tâche est d’attribuer une note de toxicité en tenant compte de toutes ces analyses et en suivant le barème donné.\n\n"
)

certitude_prompt = PromptType(
    name="Certitude",
    system_prompt=safe_read(ROOT / "certitude.txt"),
    start_prompt="Voici les informations à analyser :\n\n",
    end_prompt="Ta tâche est d’évaluer la certitude du score de toxicité.\n\n"
)

conclusion_prompt = PromptType(
    name="Conclusion",
    system_prompt=safe_read(ROOT / "conclusion.txt"),
    start_prompt="Voici les informations à analyser :\n\n",
    end_prompt="Ta tâche est conclure sur la toxicité du contenu.\n\n"
)

# === DICTIONARY OF PROMPTS ===

PROMPTS_HEADERS = {
    "explication": explication_prompt,
    "ton": ton_prompt,
    "intention": intention_prompt,
    "categorie": categorie_prompt,
    "labels": taxonomie_prompt,
    "toxicite_score": toxicite_score_prompt,
    # "certitude": certitude_prompt,
    "conclusion": conclusion_prompt,
}