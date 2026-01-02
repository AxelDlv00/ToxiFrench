# ╔════════════════════════════════════════════════════════╗
# ║           Prioritize data using weak signals           ║
# ╚════════════════════════════════════════════════════════╝

# /********************************************************\
# * This script prioritizes messages in the cleaned forum   *
# * dataset by calculating a complex weight to boost 'weak  *
# * signals'. Weak signals are defined as messages posted   *
# * during off-peak hours (e.g., early morning) and those   *
# * from banned users or deleted messages. The goal is to   *
# * reorder the dataset to highlight these potentially      *
# * important messages for further analysis.                *
# \********************************************************/

import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict

# ╭────────────────────────────────────────────────────────╮
# │                    Global variables                    │
# ╰────────────────────────────────────────────────────────╯

# Define directory and file paths
ROOT_DIR: Path              = Path("..") 
DATA_DIR: Path              = ROOT_DIR / "data"
# Input path for the filtered dataset
PATH_FILTERED_DATASET: Path = DATA_DIR / "ForumData" / "anonymous_forum_filtered.csv"
# Output path for the prioritized dataset
PATH_PRIORITIZED_DATASET: Path = DATA_DIR / "ForumData" / "anonymous_forum_prioritized.csv"

# Parameters for Weak Signal Prioritization
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# > The following parameters are purely empirical, as the  >
# >   weak prioritization is only to sort between the 2M   >
# >                       messages.                        >
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
MU: float                   = 6.0    # Mean (center) of the Gaussian distribution (e.g., 6 AM)
SIGMA: float                = 2.5    # Standard deviation (width) of the Gaussian distribution
LAMBDA_BANNED: float        = 10.0   # Weight multiplier for messages from banned users
LAMBDA_DELETED: float       = 3.0    # Weight multiplier for deleted messages

# ╭────────────────────────────────────────────────────────╮
# │                      Core Functions                    │
# ╰────────────────────────────────────────────────────────╯

def load_dataset(path: Path) -> pd.DataFrame:
    """Loads the filtered dataset."""
    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path, sep=",", encoding="utf-8")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def calculate_priority_weights(df: pd.DataFrame, mu: float, sigma: float, lambda_banned: float, lambda_deleted: float) -> pd.DataFrame:
    """
    Calculates a complex priority weight to boost 'weak signals':
    1. Gaussian weight based on the hour (e.g., boosting off-peak hours).
    2. Normalization by the message volume per hour.
    3. Boosting based on user status (banned) and message status (deleted).
    """
    print("\n--- Calculating Priority Weights ---")

    # Compute Gaussian weight based on the 'hour' column
    # The weight is highest when 'hour' is close to MU (6 AM)
    print(f"Applying Gaussian weighting (mu={mu}, sigma={sigma})...")
    df["gauss_weight"] = np.exp(-((df["hour"] - mu) ** 2) / (2 * sigma ** 2))
    # 

    # Normalize by the number of messages per hour
    print("Normalizing by message volume per hour...")
    hour_counts: Dict[int, int] = df["hour"].value_counts().to_dict()
    df["hour_count"] = df["hour"].map(hour_counts)
    
    # Hour-normalized weight: boosts signals in hours with low message counts
    # The minimum count is set to 1 to avoid division by zero
    df["hour_norm_weight"] = df["gauss_weight"] / df["hour_count"].clip(lower=1)

    # Apply Boosts for Weak Signals (Banned/Deleted)
    print(f"Applying boosts (Banned: x{1 + lambda_banned}, Deleted: x{1 + lambda_deleted})...")
    
    # Priority weight: combines time preference (Gaussian + Volume) and content status (Banned + Deleted)
    df["priority_weight"] = (
        df["hour_norm_weight"] * (1 + lambda_banned * df["banned"]) * (1 + lambda_deleted * df["deleted"])
    )
    
    print(f"Weights calculated. Min Weight: {df['priority_weight'].min():.4f}, Max Weight: {df['priority_weight'].max():.4f}")
    
    return df

def prioritize_and_save(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """
    Samples the entire dataset using the priority weights and saves the result 
    as a single, ordered file.
    """
    print("\n--- Prioritizing Dataset (Sampling by Weight) ---")
    
    # Use weights for sampling: ensures that high-priority messages are selected
    # with a probability proportional to their weight.
    # Since frac=1.0, we resample the entire dataset, effectively shuffling it
    # according to the weights.
    df_prioritized = df.sample(
        frac=1.0,
        random_state=42,
        weights=df["priority_weight"],
        replace=False # Ensure all rows are used exactly once in the new order
    ).reset_index(drop=True)

    print(f"Dataset successfully prioritized. New order established: {df_prioritized.shape[0]} rows.")
    
    # Drop helper columns
    df_final = df_prioritized.drop(
        columns=["gauss_weight", "hour_count", "hour_norm_weight", "priority_weight"],
        errors="ignore"
    )

    # Save the single, ordered file
    df_final.to_csv(path, index=False, encoding="utf-8")
    print(f"Prioritized dataset saved to {path}.")
    
    return df_final

def main():
    """Executes the weak signal prioritization pipeline."""
    
    df = load_dataset(PATH_FILTERED_DATASET)
    df = calculate_priority_weights(df, MU, SIGMA, LAMBDA_BANNED, LAMBDA_DELETED)
    prioritize_and_save(df, PATH_PRIORITIZED_DATASET)
    
    print("\n╔════════════════════════════════════════════════════════╗")
    print("║            Weak Signal Prioritization Complete           ║")
    print("╚════════════════════════════════════════════════════════╝")

if __name__ == "__main__":
    main()