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

# ToxiFrench Model

This repository contains the **ToxiFrench** model, a **French language model** fine-tuned for **toxic comment classification**. It is based on the [**Qwen/Qwen3-4B**](https://huggingface.co/Qwen/Qwen3-4B) architecture and is designed to detect and classify toxic comments in French text.

We performed a series of experiments to evaluate the model's performance under different fine-tuning configurations, focusing on the impact of **data selection strategies** and **Chain-of-Thought (CoT)** annotations.

## Finetuning notations

Each experiment follows a naming scheme like: **(r/o)(e/d)(a/b)(s/m/l)**  
Where:

- `r` = random order, `o` = ordered (curriculum)
- `e` = equal toxic/non-toxic, `d` = real-world imbalance
- `a` = with CoT finetuning, `b` = without CoT
- `s` = small (100), `m` = medium (1000), `l` = large (all)

> e.g. `rdal` is the model trained on the natural distribution of toxicity (`d`), on an arbitrary order (`r`), with CoT annotations (`a`), and on the whole dataset (`l`).

If a label like `<cot-step>` is present in the checkpoint name, it indicates that the CoT that was used during training did not include this specific reasoning step.

## Citation

```
@misc{toxifrench2025,
title={ToxiFrench},
author={Delaval Axel},
year={2025},
}
```