# ╔════════════════════════════════════════════════════════╗
# ║              Anonymization of the dataset              ║
# ╚════════════════════════════════════════════════════════╝

# /********************************************************\
# *This script performs full anonymization of the raw forum*
# *                        dataset.                        *
# * The goal is to ensure that no identifiable information *
# *remains in the dataset, while preserving metadata useful*
# *                     for analysis.                      *
# *              It should be run only once.               *
# \********************************************************/

# ╭────────────────────────────────────────────────────────╮
# │                       Libraries                        │
# ╰────────────────────────────────────────────────────────╯

import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from spellchecker import SpellChecker
from tqdm import tqdm
from flashtext import KeywordProcessor

# ╭────────────────────────────────────────────────────────╮
# │                    Global variables                    │
# ╰────────────────────────────────────────────────────────╯

ROOT_DIR                    = Path("..")
PATH_SECRETS                = ROOT_DIR / ".secrets"
PATH_DATA                   = ROOT_DIR / "data"

PATH_DATASET                = PATH_SECRETS / "raw_forum.csv"
PATH_USERS_MAP              = PATH_SECRETS / "mapping_users.csv"
PATH_TOPICS_MAP             = PATH_SECRETS / "mapping_topics.csv"
PATH_MSG_ID_MAP             = PATH_SECRETS / "mapping_msg_id.csv"
PATH_SUPPLEMENTARY_WORDS    = PATH_SECRETS / "french_dict_supplementary.txt"
PATH_ANONYM                 = PATH_DATA    / "ForumData" / "anonymous_forum.csv"
# ╭────────────────────────────────────────────────────────╮
# │                         Utils                          │
# ╰────────────────────────────────────────────────────────╯

def sha_alias(text: str, n: int, prefix: str) -> str:
    """Returns `prefix` + first `n` hex chars of SHA-256(text)."""
    return prefix + hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]

def initialize_spell_checker() -> Tuple[SpellChecker, Set[str]]:
    """Initialize the SpellChecker and load the French dictionary."""
    print("Initializing French dictionary...")
    
    spell = SpellChecker(language='fr')
    french_words = set(spell.word_frequency.words()) 
    
    # Add supplementary words
    if PATH_SUPPLEMENTARY_WORDS.exists():
        with open(PATH_SUPPLEMENTARY_WORDS, "r", encoding="utf-8") as f:
            supplementary_words = set(f.read().splitlines())
        french_words.update(supplementary_words)
    
    print(f"Number of French words considered: {len(french_words)}")
    return spell, french_words

def is_real_username(name: str, french_words: Set[str]) -> bool:
    """Heuristic : Treats as real username if not a French word."""
    return isinstance(name, str) and name.lower() not in french_words

# ╭────────────────────────────────────────────────────────╮
# │                        Main Code                       │   
# ╰────────────────────────────────────────────────────────╯

def load_and_inspect_data() -> pd.DataFrame:
    """Loads the raw dataset, performs basic inspection, and shuffles."""
    print(f"Loading dataset from {PATH_DATASET}...")
    df = pd.read_csv(PATH_DATASET, sep=",", encoding="utf-8", low_memory=False)
    
    # Shuffle for a full anonymization sample and reset index
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Convert date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Display metadata
    print("\n--- Data Inspection ---")
    print(f"Number of rows : {df.shape[0]}")
    print(f"Number of columns : {df.shape[1]}")
    print(f"Columns : {df.columns.tolist()}")
    print(f"Unique users : {df['user'].nunique()}")
    print(f"Unique topics : {df['topic'].nunique()}")
    print(f"Deleted messages : {df['deleted'].sum()}")
    print(f"`msg_id` is unique : {df['msg_id'].is_unique}")
    print("-------------------------\n")
    
    return df

def build_and_save_mappings(df: pd.DataFrame, french_words: Set[str]) -> Tuple[Dict, Dict, Dict, List[str]]:
    """Builds and saves the mapping tables (user, topic, msg_id)."""
    
    # --- User Mapping ---
    print("Building user mapping...")
    users = df["user"].dropna().unique()
    users_real = []
    mapping_users = {}
    for u in tqdm(users, desc="Users"):
        if is_real_username(u, french_words):
            # Map real usernames to anon_user_
            mapping_users[u] = sha_alias(u, 10, "anon_user_")
            users_real.append(u)
        else:
            # Map French words to frenchword_ (keeping metadata, but protecting identity)
            mapping_users[u] = sha_alias(u, 10, "frenchword_")
    
    print(f"Number of 'real' usernames identified for content scrubbing: {len(users_real)}")

    # --- Topic Mapping ---
    print("Building topic mapping...")
    topics = df["topic"].dropna().unique()
    mapping_topics = {
        t: sha_alias(t, 8, "anon_topic_") for t in tqdm(topics, desc="Topics")
    }

    # --- Message ID Mapping ---
    print("Building message ID mapping...")
    msg_ids = df["msg_id"].dropna().unique()
    mapping_msg_ids = {
        m: sha_alias(str(m), 12, "anon_msg_") for m in tqdm(msg_ids, desc="Msg IDs")
    }
    
    # --- Saving Mappings ---
    print("Saving mapping tables...")
    pd.DataFrame.from_dict(mapping_users, orient="index", columns=["anonymous_user"])\
      .rename_axis("original_user").to_csv(PATH_USERS_MAP, encoding="utf-8")
    
    pd.DataFrame.from_dict(mapping_topics, orient="index", columns=["anonymous_topic"])\
      .rename_axis("original_topic").to_csv(PATH_TOPICS_MAP, encoding="utf-8")
      
    pd.DataFrame.from_dict(mapping_msg_ids, orient="index", columns=["anonymous_msg_id"])\
      .rename_axis("original_msg_id").to_csv(PATH_MSG_ID_MAP, encoding="utf-8")
    
    return mapping_users, mapping_topics, mapping_msg_ids, users_real

def apply_basic_anonymization(df: pd.DataFrame, mappings: Tuple[Dict, Dict, Dict]) -> pd.DataFrame:
    """Applies mappings and performs basic column anonymization/transformation."""
    mapping_users, mapping_topics, mapping_msg_ids = mappings
    
    print("Applying mappings to columns...")
    df["user"]   = df["user"].map(mapping_users)
    df["topic"]  = df["topic"].map(mapping_topics)
    df["msg_id"] = df["msg_id"].map(mapping_msg_ids)

    print("Dropping and transforming sensitive/unnecessary columns...")
    
    # Extract only the hour of the day
    df["hour"] = df["date"].dt.hour.astype("Int8")
    
    # Drop URL columns and the full date/time column
    df = df.drop(columns=["topic_url", "profile_url", "date"], errors='ignore')
    
    return df

def clean_content_from_usernames(df: pd.DataFrame, users_real: List[str]) -> pd.DataFrame:
    """Detects and removes messages containing 'real' usernames in the content."""
    
    print("Content scrubbing: searching for real usernames in messages...")
    
    # Build the KeywordProcessor using only the list of 'real' usernames
    kp = KeywordProcessor()
    for user in tqdm(users_real, desc="Building Keyword Processor"):
        kp.add_keyword(user)

    rows_before = df.shape[0]

    def contains_real_username(text):
        if isinstance(text, str):
            # Use KeywordProcessor for efficient keyword extraction
            return bool(kp.extract_keywords(text))
        return False

    # Filter out rows that DO contain a real username
    df = df[~df["content"].apply(contains_real_username)].reset_index(drop=True)

    rows_after = df.shape[0]
    removed_count = rows_before - rows_after
    print(f"Messages removed containing real usernames: {removed_count}")
    
    if removed_count > 0:
        print(f"Proportion of messages removed: {removed_count/rows_before:.2%}")
        
    return df

def main():
    """Main function to execute the complete anonymization process."""
    _ , french_words = initialize_spell_checker()
    df = load_and_inspect_data()
    mappings = build_and_save_mappings(df, french_words)
    mapping_users, mapping_topics, mapping_msg_ids, users_real = mappings
    df = apply_basic_anonymization(df, (mapping_users, mapping_topics, mapping_msg_ids))
    df = clean_content_from_usernames(df, users_real)

    print(f"\nSaving anonymized dataset to {PATH_ANONYM}...")
    df.to_csv(PATH_ANONYM, sep=",", encoding="utf-8", index=False)
    
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║             Anonymization Process Complete               ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"Final size of the anonymized dataset : {df.shape}")


if __name__ == "__main__":
    main()