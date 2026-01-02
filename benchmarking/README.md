# Benchmarking scrips

This directory contains the core Python scripts and configuration files necessary to run the toxicity prediction evaluation pipeline.

## Key Files

| File | Description |
| :--- | :--- |
| `benchmark.py` | The **main execution script**. It handles: loading configurations, reading the benchmark data, running predictions in parallel using the specified predictor class, and calculating/saving performance metrics (Precision, Recall, F1, AUC, etc.). |
| `config.json` | **Configuration file** for all experiments. Each entry defines a specific test run, including: the `predictor` class to use, the `model_name`, the benchmark file (`.csv`), the API keys, and the prompt templates (`system_prompt`, `prompt_template`). |

## Subdirectories

| Directory | Content |
| :--- | :--- |
| `prompts/` | Stores the text files containing the reusable prompts (system instructions and templates) in French and English, often used for API-based LLMs and local models (e.g., `system_prompt_french.txt`, `detailed_toxicity_english.txt`). |
| `utils/` | Contains the base class and specific implementation files for all toxicity predictors. |
| `utils/toxicity_predictor.py` | Defines the abstract base class (`ToxicityPredictor`) that all prediction implementations must inherit from, ensuring a consistent `predict(self, text)` interface. |
| `utils/models/` | Houses the concrete predictor implementations for various models (OpenAI, Gemini, Mistral, Hugging Face local models, etc.). |

## Usage

To run an experiment, ensure your environment variables are set and specify the name of the experiment defined in `config.json`:

```bash
# Example: Run the Gemini 2.5 Flash French test
python benchmark.py --experiment "Gemini-2.5-flash-Detailed-TFR"
```