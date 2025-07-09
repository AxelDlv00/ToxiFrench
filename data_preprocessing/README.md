# `data_preprocessing/` folder

This folder contains all scripts, notebooks, and utilities used for preparing, cleaning, anonymizing, and annotating the datasets for the **ToxiFrench** project.

## Folder structure

- `anonymization/`: Notebooks and scripts for full anonymization of the raw forum dataset, including user, topic, and message ID mapping.
- `cleaning/`: Notebooks for filtering, cleaning, and preprocessing the raw data (removing irrelevant, too short/long, or noisy messages).
- `GPT_annotation/`: Scripts and notebooks for automatic annotation using GPT-based APIs, including batch processing and quality checks (the cleanest version is in [data](../data/cleaned_annotation/) in the `parquet` format).
- `weak_signals/`: Notebooks for extracting and preparing disjoint subsets from the main dataset by trying to ordering with a decreasing toxicity using weak signals (bans, deletions, ...)

## Usage

- Use the notebooks in each subfolder to reproduce the preprocessing pipeline: from raw data anonymization to cleaning, annotation, and subset extraction.
- Mapping files and confidential resources are stored in the `data/confidential/` directory and are not versioned.
- Each notebook is self-documented and contains step-by-step instructions.

---

For more details on the overall methodology and data organization, see the project's [main README](../README.md).