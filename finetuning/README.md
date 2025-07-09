---
license: mit
task_categories:
- text-classification
language:
- fr
tags:
- text-classification
- toxicity
- hate-speech
- content-moderation
- chain-of-thought
- curriculum-learning
- nlp
- french-dataset
- classification
pretty_name: ToxiFrench
datasets:
- Naela00/ToxiFrenchFinetuning
base_model:
- Qwen/Qwen3-4B
---
# ToxiFrench: Benchmarking and Investigating SLMs and CoT Finetuning for French Toxicity Detection

<!-- Badges/Tags -->
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deployed-brightgreen?style=flat-square&logo=github)](https://axeldlv00.github.io/ToxiFrench/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?style=flat-square&logo=github)](https://github.com/AxelDlv00/ToxiFrench)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-blue?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/Naela00/ToxiFrenchFinetuning)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](./LICENSE)

**Author:** Axel Delaval

**Affiliations:** École Polytechnique & Shanghai Jiao Tong University (SJTU)

**Email:** [axel.delaval@gmail.com](mailto:axel.delaval@gmail.com)

---

> ⚠️ **Content Warning**
> This project and the associated dataset contain examples of text that may be considered offensive, toxic, or otherwise disturbing. The content is presented for research purposes only.

---

## Table of Contents
- [Abstract](#abstract)
- [Key Contributions](#key-contributions)
- [How to use ?](#how-to-use)
    - [Notations](#notations)
    - [Example Usage](#example-usage)
- [License](#license)
- [Citation](#citation)

## Abstract

Despite significant progress in English toxicity detection, performance drastically degrades in other languages like French, a gap stemming from disparities in training corpora and the culturally nuanced nature of toxicity. This paper addresses this critical gap with three key contributions. First, we introduce ToxiFrench, a new public benchmark dataset for French toxicity detection, comprising 53,622 entries. This dataset was constructed using a novel annotation strategy that required manual labeling for only 10% of the data, minimizing effort and error. Second, we conducted a comprehensive evaluation of toxicity detection models. Our findings reveal that while Large Language Models (LLMs) often achieve high performance, Small Language Models (SLMs) can demonstrate greater robustness to bias, better cross-language consistency, and superior generalization to novel forms of toxicity. Third, to identify optimal transfer-learning methods, we conducted a systematic comparison of In-Context Learning (ICL), Supervised Fine-tuning (SFT), and Chain-of-Thought (CoT) reasoning using `Qwen3-4B` and analyzed the impact of data imbalance. We propose a novel approach for CoT fine-tuning that employs a dynamic weighted loss function, significantly boosting performance by ensuring the model's reasoning is faithful to its final conclusion.

---

## Key Contributions

* **Dataset and benchmark:** Introduction of ToxiFrench, a new public benchmark dataset for French toxicity detection (53,622 entries).
* **Evaluation state-of-the-art detectors:** Extensive evaluation of LLMs (`GPT-4o`, `DeepSeek`, `Gemini`, `Mistral`, ...), SLMs (`Qwen`, `Gemma`, `Mistral`, ...), Transformers (`CamemBERT`, `DistilBERT`, ...), and moderation APIs (`Perspective API`, `OpenAI moderation`, `Mistral moderation`, ...), showing that **SLMs outperform LLMs** in robustness to bias, cross-language consistency, and generalization to novel toxicity forms.
* **Transfer learning strategies:** Systematic comparison of ICL, SFT, and CoT reasoning.
* **Model development:** Development of a **state-of-the-art 4B SLM** for French toxicity detection that outperforms several powerful LLMs based on the `Qwen3-4B` model.
* **CoT fine-tuning:** Introduction of a *novel* approach for CoT fine-tuning that employs a **dynamic weighted loss function**, significantly boosting performance by ensuring the model's reasoning is *faithful* to its final conclusion.

---

## How to use ?

This repository contains the **ToxiFrench** model, a **French language model** fine-tuned for **toxic comment classification**. It is based on the [**Qwen/Qwen3-4B**](https://huggingface.co/Qwen/Qwen3-4B) architecture and is designed to detect and classify toxic comments in French text.

We performed a series of experiments to evaluate the model's performance under different fine-tuning configurations, focusing on the impact of **data selection strategies** and **Chain-of-Thought (CoT)** annotations.

We used QLORA adapters, make sure to specify `adapter_name` when loading the model, otherwise the base model, without any fine-tuning, will be loaded.

### Notations

For conciseness, we use a three-letter notation to describe the different configurations of the fine-tuning experiments. Each experiment follows a naming scheme like: **(<strong style="color: #d9534f;">r</strong>/<strong style="color: #428bca;">o</strong>)(<strong style="color: #d9534f;">e</strong>/<strong style="color: #428bca;">d</strong>)(<strong style="color: #d9534f;">c</strong>/<strong style="color: #428bca;">b</strong>)**  
Where: 

<table style="width:100%; border-collapse: collapse;">
  <thead>
    <tr>
      <th style="text-align:left; padding: 8px; border-bottom: 2px solid black;">Parameter</th>
      <th style="text-align:left; padding: 8px; border-bottom: 2px solid black;">Code</th>
      <th style="text-align:left; padding: 8px; border-bottom: 2px solid black;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Data Order</strong></td>
      <td style="padding: 8px; color: #d9534f;">[r]</td>
      <td style="padding: 8px;">Training data is presented in a <strong style="color: #d9534f;">random</strong> order.</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #ddd; color: #428bca;">[o]</td>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">Data is <strong style="color: #428bca;">ordered</strong> (Curriculum Learning).</td>
    </tr>
    <tr>
      <td rowspan="2" style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Class Balance</strong></td>
      <td style="padding: 8px; color: #d9534f;">[e]</td>
      <td style="padding: 8px;">Training set has an <strong style="color: #d9534f;">equal</strong> (balanced) number of toxic and non-toxic samples.</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #ddd; color: #428bca;">[d]</td>
      <td style="padding: 8px; border-bottom: 1px solid #ddd;">Training set uses a <strong style="color: #428bca;">different</strong> (imbalanced) class distribution.</td>
    </tr>
    <tr>
      <td rowspan="2" style="padding: 8px;"><strong>Training Target</strong></td>
      <td style="padding: 8px; color: #d9534f;">[c]</td>
      <td style="padding: 8px;">Finetuning on the complete <strong style="color: #d9534f;">Chain-of-Thought</strong> annotation.</td>
    </tr>
    <tr>
      <td style="padding: 8px; color: #428bca;">[b]</td>
      <td style="padding: 8px;">Finetuning on the final <strong style="color: #428bca;">binary</strong> label only (direct classification).</td>
    </tr>
  </tbody>
</table>

> e.g. `rec` is the model trained on an oversampled dataset for balance, with batches in an arbitrary order (`r`), and with CoT reasoning (`c`).

### Example Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Choose which adapter to load
target_adapter_name = "rec" # Among the following six configurations : "odc", "oeb", "oec", "rdc", "reb", "rec"

# Load the base model
base_model_name = "Qwen/Qwen3-4B"
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the specific adapter by name from the repository
adapter_repo_id = "Naela00/ToxiFrench"
model = PeftModel.from_pretrained(
    model,
    adapter_repo_id,
    adapter_name=target_adapter_name # Precise which experiment to load
)

print(f"Successfully loaded the '{target_adapter_name}' adapter!")
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--- 

## Citation

If you use this project in your research, please cite it as follows:

```bibtex
@misc{delaval2025toxifrench,
    title={ToxiFrench: Benchmarking and Investigating SLMs and CoT Finetuning for French Toxicity Detection},
    author={Axel Delaval},
    year={2025},
}
```