# ToxiFrench: Benchmarking and Enhancing Language Models for French Toxicity Detection

[![arXiv](https://img.shields.io/badge/arXiv-2508.11281-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.11281)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deployed-brightgreen?style=flat-square&logo=github)](https://axeldlv00.github.io/ToxiFrench/)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-blue?style=flat-square&logo=huggingface)](https://huggingface.co/AxelDlv00/ToxiFrench)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-blue?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/AxelDlv00/ToxiFrench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](./LICENSE)

**Author:** Axel Delaval  
**Affiliations:** École Polytechnique & Shanghai Jiao Tong University (SJTU)  
**Email:** [name].[surname]@gmail.com

---

> ⚠️ **Content Warning** : This project contains examples of toxic language, including hate speech and insults, included for research purposes.

---

## Overview

**ToxiFrench** addresses the lack of culturally relevant, human-annotated, large-scale French toxicity datasets. While English toxicity detection is well-established, French models often lack a deep grasp of cultural and linguistic nuances, such as coded toxicity.

---

## The ToxiFrench Dataset

We release a native dataset of **53,622 French online comments** spanning from 2011 to 2025. This dataset is available on [Hugging Face](https://huggingface.co/datasets/AxelDlv00/ToxiFrench).

---

## Model & Methodology

We introduce a novel **CoT fine-tuning strategy** using a **Dynamic Weighted Loss (DWL)** to improve "faithfulness"—ensuring the model's final conclusion aligns with its reasoning steps.

In [the code](./training/FineTuning/utils/dynamic_weigthed_loss.py), we implemented DWL to enhance the model's reasoning capabilities. It is designed when we want to assign different weights to tokens belonging to different parts of the output sequence, such as reasoning steps versus final answers.

**Result:** Our fine-tuned **Qwen3-4B** achieves **87% accuracy**, outperforming `GPT-4o` (84%) and `DeepSeek-R1` (84%) on our benchmark.

---

## Installation

Create the optimized environment (Python 3.12, CUDA-enabled) using the provided YAML:

```bash
conda env create -f environment.yml
conda activate ToxiFrench
```

For running training scripts, you might need to configure `accelerate`:

```bash
accelerate config
```

## Citation 

If you find ToxiFrench useful for your research, please consider citing our paper:

```bibtex
@misc{delaval2025toxifrenchbenchmarkingenhancinglanguage,
      title={ToxiFrench: Benchmarking and Enhancing Language Models via CoT Fine-Tuning for French Toxicity Detection}, 
      author={Axel Delaval and Shujian Yang and Haicheng Wang and Han Qiu and Jialiang Lu},
      year={2025},
      eprint={2508.11281},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.11281}, 
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.