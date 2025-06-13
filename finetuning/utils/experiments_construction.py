"""experiments_construction.py
A utility class for generating multiple dataset variants for finetuning experiments
on toxicity detection with or without Chain‑of‑Thought (CoT) annotations.

Author: Axel Delaval (Naela)

The class supports the full matrix of configurations described below.
Configuration key (4–5 chars):
    <r|o><e|d><a|b>[<s|m|l>]
        r / o       -> random order | ordered (curriculum)
        e / d       -> equal class proportion | dataset (original) proportion
        a / b       -> with CoT | without CoT columns
        s / m / l   -> sample size : s = 200, m = 1000, l = all rows (~50000)
Examples
--------
    cfg = "rea"   # random, equal proportion, with CoT
    cfg = "rdbm"  # random, dataset proportion, without CoT, medium (1000 rows)

Usage
-----
>>> builder = FinetuneDatasetBuilder(df,
                                     text_col="content",
                                     label_col="annotator's conclusion",
                                     cot_col="cot")
>>> small_balanced = builder.get_split("rea")  # DataFrame
>>> splits = builder.build_all()                # dict[str, DataFrame]
>>> builder.save_splits(Path("./data/splits"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["FinetuneDatasetBuilder"]


# ---------------------------------------------------------------------------
# Helper types & constants
# ---------------------------------------------------------------------------
_Order = Literal["r", "o"]          # random | ordered
_Proportion = Literal["e", "d"]     # equal | different (original)
_CoT = Literal["a", "b"]              # with-cot | without-cot
_SizeCode = Literal["s", "m", "l"]   # small | medium | large

SIZE_MAP: dict[_SizeCode, Optional[int]] = {"s": 200, "m": 1000, "l": None}

# ---------------------------------------------------------------------------
class FinetuneDatasetBuilder:
    """Factory for dataset variants needed in the finetuning sweep.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset containing at least *text* and *label* columns,
        optionally a *cot* column for chain-of-thought explanations.
    text_col : str, default "content"
        Name of the text column.
    label_col : str, default "annotator's conclusion"
        Name of the binary label column (0 = non-toxic, 1 = toxic).
    cot_col : Optional[str], default "cot"
        Name of the CoT column. If *None*, CoT handling is disabled.
    random_state : int, default 42
        Global random seed for reproducibility.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        text_col: str = "content",
        label_col: str = "literal_conclusion_annotator",
        cot_col: Optional[str] = "cot_text",
        random_state: int = 42,
    ) -> None:
        self.df: pd.DataFrame = df.copy()
        self.text_col = text_col
        self.label_col = label_col
        self.cot_col = cot_col
        self.rng = np.random.default_rng(random_state)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def get_split(self, order, prop, cot, size) -> pd.DataFrame:
        """Return a single dataset split according to *code*.

        Parameters
        ----------
        code : str
            Combination key as documented in the module docstring.
        """
        df = self._subset_columns(cot)
        df = self._apply_proportion(df, prop)
        df = self._apply_sample(df, prop, size)
        df = self._apply_order(df, order)
        return df.reset_index(drop=True)
    
    def build_all(self) -> Dict[str, pd.DataFrame]:
        """Generate and return *all* combinations as a dict keyed by code."""
        combos: List[Tuple[str, str, str, Optional[str]]] = [
            (order, prop, cot, size)
            for order in ("r", "o")  # random | ordered
            for prop in ("e", "d")  # equal | different
            for cot in ("a", "b")  # with CoT | without CoT
            for size in ("s", "m", "l")  # small | medium | large
        ]

        all_splits: Dict[str, pd.DataFrame] = {}
        for (order, prop, cot, size) in combos:
            code = order + prop + cot + size
            all_splits[code] = self.get_split(order, prop, cot, size)
        return all_splits

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _subset_columns(self, cot: _CoT, keep=['msg_id', "conclusion", "note"]) -> pd.DataFrame:
        """Select relevant columns based on the CoT type."""
        cols = keep + [self.text_col, self.label_col, self.cot_col]
        df = self.df[cols].copy()
        if cot == "b":
            df[self.cot_col] = "Ce message est-il toxique ?"
        return df

    def _apply_proportion(self, df: pd.DataFrame, proportion: _Proportion) -> pd.DataFrame:
        """Balance the dataset according to the specified proportion."""
        if proportion == "e":
            counts = df[self.label_col].value_counts()
            n_min = counts.min()
            dfs = []
            for label, group in df.groupby(self.label_col):
                dfs.append(group.sample(n=n_min, random_state=self.rng.integers(0, 2**32)))
            df_balanced = pd.concat(dfs, ignore_index=True)
            return df_balanced
        return df  # 'd' keeps original distribution

    def _apply_sample(
        self,
        df: pd.DataFrame,
        proportion: _Proportion,
        size_code: Optional[_SizeCode],
    ) -> pd.DataFrame:
        """Sample the DataFrame according to the specified size and proportion."""
        if proportion == "e":
            # Then we want equal proportions of toxic and non-toxic 
            n_rows = SIZE_MAP.get(size_code, None)
            if n_rows is not None:
                n_rows //= 2
                df_toxic = df[df[self.label_col] == 'oui'].sample(n=n_rows, random_state=self.rng.integers(0, 2**32))
                df_non_toxic = df[df[self.label_col] == 'non'].sample(n=n_rows, random_state=self.rng.integers(0, 2**32))
                return pd.concat([df_toxic, df_non_toxic], ignore_index=True)
            return df.sample(frac=1.0, random_state=self.rng.integers(0, 2**32))
        elif proportion == "d":
            # Original distribution, just sample based on size_code
            n_rows = SIZE_MAP.get(size_code, None)
            if n_rows is not None and n_rows < len(df):
                return df.sample(n=n_rows, random_state=self.rng.integers(0, 2**32))
            return df.sample(frac=1.0, random_state=self.rng.integers(0, 2**32))
        else: # Invalid proportion code
            raise ValueError(f"Invalid proportion code: {proportion}")
        
    def _apply_order(self, df: pd.DataFrame, order: _Order) -> pd.DataFrame:
        """Order the DataFrame according to the specified order type."""
        if order == "r":
            return df.sample(frac=1.0, random_state=self.rng.integers(0, 2**32))
        else:  # If we use curriculum learning, we want measure the complexity by the agreement between humans and GPT
            df = df.sample(frac=1.0, random_state=self.rng.integers(0, 2**32))
            df['agreement'] = (df["literal_conclusion_annotator"].apply(lambda x: 1 if x == "oui" else 0) == df['conclusion'])
            agreement_per_note = df.groupby('note')['agreement'].mean().reset_index()
            df['note_order'] = df['note'].map(agreement_per_note.set_index('note')['agreement'])
            df.sort_values(by='note_order', ascending=False, inplace=True)
            return df
