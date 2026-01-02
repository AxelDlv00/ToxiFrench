# ╔════════════════════════════════════════════════════════╗
# ║                   Clean the dataset                    ║
# ╚════════════════════════════════════════════════════════╝

# /********************************************************\
# * This script cleans the anonymized forum dataset by    *
# * filtering out messages based on content length and    *
# * removing messages containing sensitive information    *
# * such as URLs, emails, and phone numbers.              *
# \********************************************************/

import pandas as pd
from pathlib import Path
from typing import List, Pattern

import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm.auto import tqdm

# ╭────────────────────────────────────────────────────────╮
# │                    Global variables                    │
# ╰────────────────────────────────────────────────────────╯

# Define directory and file paths
ROOT_DIR                    = Path("..")  
DATA_DIR                    = ROOT_DIR / "data"
PATH_ANONYM                 = DATA_DIR / "ForumData" / "anonymous_forum.csv"
PATH_OUT                    = DATA_DIR / "ForumData" / "anonymous_forum_filtered.csv"

# Content length filtering parameters (in words)
MIN_LENGTH: int             = 5
MAX_LENGTH: int             = 25
    
# Patterns for identifying sensitive content (URLs, emails, etc.)
PATTERNS_TO_MASK: List[str] = [
    r"https?://\S+",                # URLs starting with http/https
    r"www\.\S+",                    # URLs starting with www
    r"\b\S+@\S+\.\S+\b",            # Emails
    r"\b\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{3,4}\b",  # Phone numbers (rough heuristic)
    r"@\w+",                        # @usernames (e.g., Twitter, Instagram)
    r"u/\w+",                       # Reddit-style usernames
    r"\b\d{1,3}(\.\d{1,3}){3}\b",   # IP addresses
]

# ╭────────────────────────────────────────────────────────╮
# │                      Core Functions                    │
# ╰────────────────────────────────────────────────────────╯

def load_dataset(path: Path) -> pd.DataFrame:
    """Loads the dataset and prints initial information."""
    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path, sep=",", encoding="utf-8")
    print(f"Number of rows before cleaning: {df.shape[0]}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def filter_by_content_length(df: pd.DataFrame, min_len: int, max_len: int) -> pd.DataFrame:
    """
    Filters rows based on content length (number of words) and plots the distribution.
    """
    print(f"\n--- Filtering by Content Length (Min={min_len}, Max={max_len}) ---")
    
    tqdm.pandas(desc="Computing content length")
    df["content_length"] = df["content"].progress_apply(lambda x: len(str(x).split()))

    rows_before = df.shape[0]
    df_filtered = df[(df["content_length"] >= min_len) & (df["content_length"] <= max_len)].reset_index(drop=True)
    rows_after = df_filtered.shape[0]

    print(f"Removed {rows_before - rows_after} rows (too short or too long content).")
    print(f"Remaining rows after length filtering: {rows_after}")

    df_filtered = df_filtered.drop(columns=["content_length"])
    
    return df_filtered

def mask_and_filter_sensitive_content(df: pd.DataFrame, patterns: List[str]) -> pd.DataFrame:
    """
    Filters out rows whose content matches one of the provided sensitive patterns (URLs, emails, etc.).
    """
    print("\n--- Filtering Sensitive Content (URLs, Emails, etc.) ---")
    
    # Combine patterns into a single efficient regular expression
    combined_pattern = re.compile("|".join(patterns), flags=re.IGNORECASE)
    
    # Flag rows that contain any of the sensitive patterns
    tqdm.pandas(desc="Checking for sensitive content")
    df["non_anonymized_flag"] = df["content"].progress_apply(
        lambda x: bool(combined_pattern.search(str(x))) if pd.notna(x) else False
    )

    rows_before = df.shape[0]
    non_anon_rows = df["non_anonymized_flag"].sum()
    
    print(f"Found {non_anon_rows} rows with non-anonymized sensitive patterns.")
    
    # Keep only rows where the flag is False (i.e., content is clean)
    df_filtered = df[~df["non_anonymized_flag"]].reset_index(drop=True)
    rows_after = df_filtered.shape[0]
    
    print(f"Removed {rows_before - rows_after} rows containing sensitive content.")
    
    # Drop helper column
    df_filtered = df_filtered.drop(columns=["non_anonymized_flag"])
    
    return df_filtered

def save_dataset(df: pd.DataFrame, path: Path):
    """Saves the final cleaned dataset."""
    df.to_csv(path, sep=",", encoding="utf-8", index=False)
    print(f"\nFinal cleaned dataset saved to {path}. Remaining rows: {df.shape[0]}")

def main():
    """Executes the full dataset cleaning pipeline."""

    df = load_dataset(PATH_ANONYM)
    df_length_filtered = filter_by_content_length(df, MIN_LENGTH, MAX_LENGTH)
    save_dataset(df_length_filtered, PATH_OUT)

    df_to_mask = load_dataset(PATH_OUT)
    df_final = mask_and_filter_sensitive_content(df_to_mask, PATTERNS_TO_MASK)
    save_dataset(df_final, PATH_OUT)
    
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║                   Dataset Cleaning Complete              ║")
    print("╚════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    # Initialize tqdm progress bar for pandas operations
    tqdm.pandas(desc="Processing")
    
    main()