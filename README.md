# ToxiFrench: Benchmarking and Enhancing Language Models via CoT Fine-Tuning for French Toxicity Detection

<!-- Badges/Tags -->
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deployed-brightgreen?style=flat-square&logo=github)](https://axeldlv00.github.io/ToxiFrench/)
[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-blue?style=flat-square&logo=huggingface)](https://huggingface.co/Naela00/ToxiFrench)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-blue?style=flat-square&logo=huggingface)](https://huggingface.co/datasets/Naela00/ToxiFrench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](./LICENSE)

**Authors:** Axel Delaval, Shujian Yang, Haicheng Wang, Han Qiu, Jialiang Lu

**Affiliations:** École Polytechnique & Shanghai Jiao Tong University (SJTU) & Tsinghua University

**Email:** [axel.delaval.2022@polytechnique.org](mailto:axel.delaval.2022@polytechnique.org)

---

> ⚠️ **Content Warning**
> This project and the associated dataset contain examples of text that may be considered offensive, toxic, or otherwise disturbing. The content is presented for research purposes only.

---

## Abstract

Detecting toxic content using language models is crucial yet challenging. While substantial progress has been made in English, toxicity detection in French remains underdeveloped, primarily due to the lack of culturally relevant, large-scale datasets. In this work, we introduce TOXIFRENCH, a new public benchmark of 53,622 French online comments, constructed via a semi-automated annotation pipeline that reduces manual labeling to only 10% through high-confidence LLM-based pre-annotation and human verification. Then, we benchmark a broad range of models and uncover a counterintuitive insight: Small Language Models (SLMs) outperform many larger models in robustness and generalization under the toxicity detection task. Motivated by this finding, we propose a novel Chain-of-Thought (CoT) fine-tuning strategy using a dynamic weighted loss that progressively emphasizes the model's final decision, significantly improving faithfulness. Our fine-tuned 4B model achieves state-of-the-art performance, improving its F1 score by 13% over its baseline and outperforming LLMs such as GPT-40 and Gemini-2.5. Further evaluation on a cross-lingual toxicity benchmark demonstrates strong multilingual ability, suggesting that our methodology can be effectively extended to other languages and safety-critical classification tasks.

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
    title={ToxiFrench: Benchmarking and Enhancing Language Models via CoT Fine-Tuning for French Toxicity Detection},
    author={Axel Delaval and Shujian Yang and Haicheng Wang and Han Qiu and Jialiang Lu},
    year={2025},
}
```


