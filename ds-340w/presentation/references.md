# References

## Three parent papers

1. **AutomaTikZ: Text-Guided Synthesis of Scientific Vector Graphics with TikZ.** Jonas Belouadi, Anne Lauscher, and Steffen Eger. ICLR 2024. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f7641940c7dd9e5de58c20e39586eb64-Abstract-Conference.html) · [Code](https://github.com/potamides/AutomaTikZ).
   Connection: a language model produces an editable graphics program. Its scientific-figure results do not establish pixel-art performance.

2. **Text2Reward: Reward Shaping with Language Models for Reinforcement Learning.** Tianbao Xie and colleagues. ICLR 2024. [Project and paper](https://text-to-reward.github.io/) · [Paper record](https://arxiv.org/abs/2309.11489).
   Connection: generated code makes reward logic inspectable. This proposal borrows the emphasis on explicit checks; it does not reproduce the robotics task or claim those results for art.

3. **Diffusion Model Alignment Using Direct Preference Optimization.** Bram Wallace and colleagues. CVPR 2024. [Paper](https://arxiv.org/abs/2311.12908) · [Code](https://github.com/SalesforceAIResearch/DiffusionDPO).
   Connection: learn from preferred and rejected image pairs. Here, people would compare rendered programs, while token-level DPO updates a code language model. The diffusion denoising objective is not used.

## Model and motivation

- [Qwen2.5-Coder-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct). Proposed base model; no tuned checkpoint is claimed in this presentation.
- [RLing Qwen to paint with code](https://surya.website/rling-qwen-to-paint-with-code). Motivation for the topic, not one of the peer-reviewed parent papers.

## Data and figures

The dataset, reference programs, and slide illustrations come from this repository's `pixelcode/dataset.py` and `pixelcode/renderer.py`. They are hand-authored templates rendered by the prototype, not model-generated results. The test split holds out object-color-position combinations. It does not hold out entire object classes.
