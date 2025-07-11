# ToxiFrench: Benchmarking and Investigating SLMs and CoT Finetuning for French Toxicity Detection

<!-- Badges/Tags -->
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deployed-brightgreen?style=flat-square&logo=github)](https://axeldlv00.github.io/ToxiFrench/)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-blue?style=flat-square&logo=huggingface)](https://huggingface.co/Naela00/ToxiFrench)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-blue?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/Naela00/ToxiFrench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](./LICENSE)

**Author:** Axel Delaval

**Affiliations:** École Polytechnique & Shanghai Jiao Tong University (SJTU)

**Email:** [axel.delaval@gmail.com](mailto:axel.delaval@gmail.com)

---

> ⚠️ **Content Warning**
> This project and the associated dataset contain examples of text that may be considered offensive, toxic, or otherwise disturbing. The content is presented for research purposes only.

---

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

## Keywords

* Content Moderation
* Toxicity Detection
* Hate Speech
* Large Language Models (LLMs)
* Small Language Models (SLMs)
* Natural Language Processing (NLP)
* French NLP

---

## Dependencies / Environments

```bash
conda create -n SJTU python=3.10.13
conda activate SJTU
conda install pip
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
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


