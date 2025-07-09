# `data/` folder

This folder contains all the main datasets and resources associated with the **ToxiFrench** project. It includes raw, filtered, and annotated datasets, as well as mapping files and external benchmarks.

## Folder structure

- `anonymous_forum.csv`: (not versioned) Raw anonymized dataset extracted from the forum.
- `anonymous_forum_filtered.csv`: (not versioned) Filtered version of the dataset, with irrelevant, too big, too small, or noisy messages removed.
- `confidential/`: Confidential files (not versioned), including raw data, user/topic mappings, and API keys.
- `headers_prompts/`: Prompt files used for automatic annotation  by `GPT api`.
- `subsets_Di/`: Disjoint subsets extracted from the main dataset (weakly) ordered by toxicity (using signals such as the banned status).
- `subsets_Di_annotated/`: Disjoint subsets from `subsets_Di/` with GPT annotations (or `NaN`)

--- 

For more details on methodology and overall organization, see the project's [main README](../README.md).