# `data/` folder

This folder contains all the main datasets and resources associated with the **ToxiFrench** project. It includes raw, filtered, and annotated datasets, as well as mapping files and external benchmarks.

## Folder structure

- `anonymous_forum.csv`: Raw anonymized dataset extracted from the forum.
- `anonymous_forum_filtered.csv`: Filtered version of the dataset, with irrelevant, too big, too small, or noisy messages removed.
- `cleaned_annotation/`: Well formatted annotations by `GPT api` (with some LLMs agreement e.g. `mistral`, `qwen`).
- `confidential/`: Confidential files (not versioned), including raw data, user/topic mappings, and API keys.
- `english_benchmarks/`: External benchmarks, notably the Jigsaw dataset for English toxicity detection.
- `headers_prompts/`: Prompt files used for automatic annotation  by `GPT api`.
- `subsets_Di/`: Disjoint subsets extracted from the main dataset (weakly) ordered by toxicity (using signals such as the banned status).
- `subsets_Di_annotated/`: Disjoint subsets from `subsets_Di/` with GPT annotations (or `NaN`)

--- 

For more details on methodology and overall organization, see the project's [main README](../README.md).