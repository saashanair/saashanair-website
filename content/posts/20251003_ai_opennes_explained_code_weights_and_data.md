---
title: "AI Openness Explained: Code, Weights, and Data"
date: 2025-10-03
tags: [sufficiently explained, ai]
---

Organisations building foundation models are often celebrated for their “commitment to openness” (TechCrunch articles: [OpenAI](https://techcrunch.com/2025/08/05/openai-launches-two-open-ai-reasoning-models/?utm_source=chatgpt.com), [Meta](https://techcrunch.com/2024/07/23/meta-releases-its-biggest-open-ai-model-yet/)). But what does *open* really mean in the context of AI? In recent years, the vocabulary has expanded beyond the familiar *open source* to include *open weights* and *open data*. Let us unpack what these terms mean, and why the definition of openness in AI is more complicated than in traditional software.

## Openness is no longer binary

In the world of traditional software, open source was treated as binary. The [Open Source Initiative (OSI) laid out ten principles](https://opensource.org/osd), and if a project followed them all, it was open source. If not, it was closed. Simple.

With AI, the picture is more nuanced. If open source was a light switch, AI openness is more like a dimmer. A model can be slightly open, very open, or open in one sense but closed in another. Unlike traditional software, where releasing source code alone was enough to make an impact, AI systems consist of multiple parts: code, data, configuration settings / hyperparameters, and a new artefact unique to AI, the model weights.

## The two big terms: open source and open weights

When people say an AI model is “open”, they usually mean one of two things.

### Open source

Traditionally, open source means making the full code behind a project publicly available. This includes the model architecture, training and inference pipelines, and sometimes even data-processing scripts. The idea is that anyone can read, modify, and build on the work.

In the early, pre-LLM days, this was the dominant approach. A classic example is the [original Transformer paper](https://arxiv.org/pdf/1706.03762) (2017) published [its codebase in full](https://github.com/tensorflow/tensor2tensor), allowing researchers and developers to replicate experiments and build new models on top of it. True open source fosters transparency, reproducibility, and collaboration. Anyone can inspect the model, retrain it from scratch, or extend it in new directions.

### Open weights

Today, this is the most common form of “openness” in AI. In fact, when people talk about a model being open source, they often actually mean open weights. This refers to the release of the trained parameters, also called checkpoints. Unlike code, these are not human-readable, so you cannot inspect or debug the model in detail. 

Despite that, open weights allow others to run the model locally, fine-tune it for specific tasks, or incorporate it into larger workflows. A good example is [Meta’s LLaMA models](https://www.llama.com/docs/overview/): the weights are available under a non-commercial licence, and some starter code is provided, but the training pipelines and underlying data remain closed.

**Why it matters:** sharing weights lets users build on powerful models without needing to train from scratch, lowering the barrier to experimentation and innovation.

## The third dimension: data

There is also [**open data**](https://opendatahandbook.org/guide/en/what-is-open-data/), which refers to datasets that are shared under licenses allowing anyone to access, use, and redistribute them.

In AI, sharing training and evaluation datasets is hugely valuable. It improves transparency, supports reproducibility, and allows others to experiment and fine-tune models. At the same time, open data is often the trickiest dimension: copyright, privacy, and sheer scale of modern datasets make full openness rare. For many organisations, datasets are also a strategic asset, which is why data openness often lags behind code or weights.

## Putting it all together

Openness in AI is no longer binary. Even when labs label their models “open source”, that [label doesn’t always hold up under scrutiny](https://opensource.org/blog/metas-llama-2-license-is-not-open-source). So, the next time you see a company praised for being “open”, ask yourself:

- Are the **weights** open?
- Is the **code** open, including training pipelines and model architecture?
- Is the **data** open?

| Code | Weights | Data | Example(s) | Notes |
| --- | --- | --- | --- | --- |
| ❌ | ❌ | ❌ | **GPT-4, Claude, Gemini** | Fully closed; only accessible via API. |
| ✅ | ❌ | ❌ | **Original Transformer (Vaswani et al. 2017)**, many research repos | Papers + code open, but no large pretrained weights. |
| ❌ | ✅ | ❌ | **LLaMA (Meta), GPT-OSS (OpenAI)** | Weights released (with non-commercial license), but not training code or data. |
| ✅ | ✅ | ❌ | **BERT (Google, 2018)**, **RoBERTa (Meta), Mixtral (Mistral)** | Code + pretrained weights available, but trained on internal/private corpora. |
| ✅ | ❌ | ✅ | **Academic reproducibility projects**, smaller benchmarks (e.g., models trained from scratch on ImageNet using public repos) | Rare at large scale; you can re-train but no pretrained weights shipped. |
| ❌ | ✅ | ✅ | **Some Kaggle competitions / community distillations** (e.g., pretrained checkpoints + dataset dumps but no training scripts) | Less common at foundation-model scale, but seen in open competitions. |
| ✅ | ✅ | ✅ | **Stable Diffusion (Stability AI)**, **BLOOM (BigScience)** | As close as it gets to “fully open” — code, weights, and datasets (or a fully documented pipeline) all released. |

The table shows that openness is multi-layered. You must look at which components are shared and under what terms. Sharing all three (code, weights and data) is the closest analogue to traditional open source software, enabling anyone to reproduce results, verify claims, and build upon the work. Most models sit somewhere along this spectrum, being open in some ways but closed in others. Understanding these nuances is key to evaluating claims of “openness” in AI.