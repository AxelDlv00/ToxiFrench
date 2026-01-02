## Forum Data Processing & Annotation Pipeline

### Directory Structure

```bash
├── CoT_annotate.py
├── anonymize_data.py
├── clean_data.py
├── prioritize_weak_signals.py
└── utils
    ├── analysis_headers.py
    └── openai_api_handling.py
```

### Script Functions

| Step | Script | Summary of Function |
| :--- | :--- | :--- |
| **1** | `anonymize_data.py` | **Anonymization:** Removes PII, replaces identifiers (user, topic, message ID) with SHA-256 hashes, and removes URL columns. |
| **2** | `clean_data.py` | **Cleaning:** Filters messages by content length (word count) and removes rows containing residual sensitive patterns like URLs or emails. |
| **3** | `prioritize_weak_signals.py` | **Prioritization:** Calculates a priority weight (boosting rare signals like off-peak messages, banned users, and deleted content) and reorders the dataset. |
| **4** | `CoT_annotate.py` | **GPT Annotation:** Launches the prioritized file for large-scale labeling using the OpenAI GPT Batch API, employing Chain-of-Thought (CoT) prompting. |

### Utilities

| Utility | Summary of Function |
| :--- | :--- |
| `utils/analysis_headers.py` | Configuration file defining the prompts and instructions for each annotation task (`--step`). |
| `utils/openai_api_handling.py` | Core class for managing the connection, batch creation, upload, status monitoring, and result parsing for the OpenAI API. |