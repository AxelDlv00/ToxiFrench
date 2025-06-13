import pandas as pd
from typing import Iterable, List, Tuple, Union

class ThoughtBuilder:
    def __init__(self, 
                 arguments: dict[str, str],
                 open_tag: str = "<think>", 
                 close_tag: str = "</think>"):
        self.arguments = arguments
        self.open_tag = open_tag
        self.close_tag = close_tag

    def build(self, row) -> str:
        if "literal_conclusion_annotator" in self.arguments:
            args_str = "\n\n".join(f'{v}\n' for k, v in self.arguments.items())
        else:
            f_none = lambda x: x if x else "None"
            args_str = "\n\n".join(f'{v}\n{f_none(row[k])}' for k, v in self.arguments.items())
        return f"{self.open_tag}\n{args_str}\n{self.close_tag}"
    
thought_explication = ThoughtBuilder(
    arguments={"explication": "Explication :"},
    # arguments={"explication": "<cot_explication>"},
    open_tag="<think>",
    close_tag="</think>"
)
thought_ton = ThoughtBuilder(
    arguments={"ton": "Tons :"},
    # arguments={"ton": "<cot_ton>"},
    open_tag="<think>",
    close_tag="</think>"
)
thought_intention = ThoughtBuilder(
    arguments={"intention": "Intentions :"},
    # arguments={"intention": "<cot_intention>"},
    open_tag="<think>",
    close_tag="</think>"
)
thought_categorie = ThoughtBuilder(
    arguments={"categorie_list": "Catégorie(s) de toxicité implicite :", "categorie_justification": "Justification :"},
    # arguments={"categorie_list": "<cot_categorie_list>", "categorie_justification": "<cot_categorie_justification>"},
    open_tag="<think>",
    close_tag="</think>"
)
thought_labels = ThoughtBuilder(
    arguments={"labels_list": "Labels :", "labels_justification": "Justification :"},
    # arguments={"labels_list": "<cot_labels_list>", "labels_justification": "<cot_labels_justification>"},
    open_tag="<think>",
    close_tag="</think>"
)
thought_toxicite = ThoughtBuilder(
    arguments={"note_string": "Score de toxicité :", "note_justification": "Justification :"},
    # arguments={"note_string": "<cot_toxicite_note>\nScore de toxicité :", "note_justification": "<cot_toxicite_justification>"},
    open_tag="<think>",
    close_tag="</think>"
)
thought_conclusion = ThoughtBuilder(
    arguments={"literal_conclusion_annotator": "En conclusion, ce message est-il toxique ?"},
    open_tag="",
    close_tag=""
)

class CoTColumnBuilder:
    """
    Add a <think> … </think> chain-of-thought column to a dataframe.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        new_col: str = "cot_text",
        column_map: List[ThoughtBuilder] = [
            thought_explication,
            thought_ton,
            thought_intention,
            thought_categorie,
            thought_labels,
            thought_toxicite,
            thought_conclusion,
        ]
    ):
        self.df = df
        self.column_map = column_map
        self.new_col = new_col

    # ------------------------------------------------------------------ #
    def add_cot_column(
        self,
    ) -> pd.DataFrame:

        cot_list: List[str] = []
        for _, row in self.df.iterrows():
            cot_text = '\n'.join(
                builder.build(row) 
                for builder in self.column_map
            )
            cot_list.append(cot_text)

        self.df[self.new_col] = cot_list
        return self.df