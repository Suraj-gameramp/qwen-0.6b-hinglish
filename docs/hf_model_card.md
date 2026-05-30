---
license: apache-2.0
language:
  - hi
  - en
base_model: Qwen/Qwen3-ASR-0.6B
base_model_relation: finetune
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
  - automatic-speech-recognition
  - code-switching
  - hinglish
  - hindi
  - speech
  - qwen3-asr
  - srota
  - srota-conv
  - conversational
  - asr
  - english
  - indic
  - indian-languages
datasets:
  - shrutisingh/HiACC
metrics:
  - wer
model-index:
  - name: Srota-Conv
    results:
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: shrutisingh/HiACC
          name: HiACC (conversational Hinglish, test)
        metrics:
          - type: wer
            value: 14.23
            name: WER
---

<div align="center">
  <img src="https://huggingface.co/datasets/Surajgameramp/qwen3-asr-0.6b-hinglish-assets/resolve/main/v1_banner.png" alt="Srota-Conv: Hinglish Conversational Specialist" width="100%"/>
</div>

# Srota-Conv (श्रोत): Hinglish Conversational Specialist

**Srota-Conv is the HiACC conversational specialist sibling of [Srota](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish)**, tuned specifically for spontaneous, code-switched Hinglish speech. It keeps English in Latin and Hindi in Devanagari (`मेरा favourite festival Diwali है`), instead of mangling code-switched speech into all-Devanagari transliteration like the base model does.

<div align="center">

### ▶️ [Try the union model in your browser](https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo)

</div>

Srota (the union model) handles both conversational and tutorial speech; Srota-Conv is the conversational specialist.

<div align="center">

[![Demo](https://img.shields.io/badge/🤗_Demo-Hinglish_ASR-yellow)](https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo)
[![Base model](https://img.shields.io/badge/Base-Qwen3--ASR--0.6B-6633cc)](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
[![HiACC](https://img.shields.io/badge/Data-HiACC-green)](https://zenodo.org/records/15551669)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![Project](https://img.shields.io/badge/Project-susrota.com-orange)](https://www.susrota.com/)
[![Family](https://img.shields.io/badge/Family-Srota-ff6f61)](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish)

</div>

## ℹ️ What is Srota-Conv?

Srota-Conv is an automatic speech recognition (ASR) model for **conversational Hinglish** (spontaneous Hindi-English code-switched speech) that transcribes into natural **mixed Devanagari + Latin** script. It is a **full-parameter fine-tune of [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)** on the HiACC corpus. It cuts WER on the HiACC conversational test set from 24.53% to **14.23%**, a **-10.30 pp** absolute reduction (a 42.0% relative drop).

**On the size.** The base model's name, Qwen3-ASR-0.6B, refers to its LLM backbone (Qwen3-0.6B, ~600M parameters). The full speech model adds a ~180M AuT audio encoder and a small projector, for ~780M parameters total. Srota-Conv is a full-parameter fine-tune of all of them: there are no LoRA adapters and no frozen layers, every native weight is updated.

**Sibling of Srota.** Srota-Conv is the **HiACC conversational specialist** in the Srota family. Its sibling, [Srota](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish), is the union model trained jointly on HiACC and OpenSLR-104, and is the recommended default for general Hinglish use because it handles both conversational and tutorial speech. Pick Srota-Conv when your audio is squarely conversational and you want the lowest conversational WER; pick Srota for everything else.

**Project.** Built by the team behind [susrota.com](https://www.susrota.com/), a voice-dictation tool that currently runs in English. Srota and Srota-Conv will power its upcoming Hinglish support; the live product does not run this model yet.

## ✨ Highlights

- **Best-in-family conversational WER.** HiACC test 24.53% → **14.23%**, a **-10.30 pp** absolute (-42.0% relative) reduction over the base.
- **Balanced across cohorts.** Adult 23.96% → **13.96%** (-10.00 pp); Child 25.61% → **14.73%** (-10.88 pp). Children improve slightly more than adults.
- **Native Hinglish output.** Emits Devanagari for Hindi words and Latin for English words, the way Hinglish is actually written (e.g. `मेरा favourite festival Diwali है`).
- **Compact and fast to train.** ~780M parameters; full fine-tune in ~10 minutes on 2x H100.
- **Honest lineage.** A full-parameter fine-tune of Qwen3-ASR-0.6B (the ~180M AuT audio encoder, the projector, and the Qwen3-0.6B LLM): no frozen layers, no LoRA adapters. The extra ~180M over the "0.6B" name is the audio encoder, not LoRA.
- **Open.** Apache-2.0; HiACC is CC BY 4.0.

## 🎧 Srota-Conv in action

Real examples from the HiACC test set. The base Qwen3-ASR-0.6B transliterates English words into Devanagari (wrong for Hinglish); Srota-Conv keeps the natural mixed script.

| | Base Qwen3-ASR-0.6B | Srota-Conv |
|---|---|---|
| **A** | `विद बड़े भाई नॉन फॉरेज क्यूरिसिटी अंडिया` | `Veer बड़े भाई known for his curiosity and diya` |
| **B** | `ओके सो मेरा होम टाउन न्यू देल्ही है` | `Okay so मेरा hometown New Delhi है` |
| **C** | `रोमांटिक पार्ट भी है मतलब ये टेस्ट वेरी ब्यूटीफुल शो` | `romantic part भी है मतलब it is very beautiful show` |

The base model collapses code-switched English into Devanagari transliteration; Srota-Conv preserves the natural mixed script that Hinglish is actually written in.

## 📊 Results

<div align="center">
  <img src="https://huggingface.co/datasets/Surajgameramp/qwen3-asr-0.6b-hinglish-assets/resolve/main/v1_wer_comparison.png" alt="WER comparison: base vs Srota-Conv on HiACC test, overall and adult/child cohorts" width="80%"/>
</div>

<p align="center"><em>WER (%) on the HiACC conversational test set: base Qwen3-ASR-0.6B vs Srota-Conv, overall and by adult/child cohort. Lower is better.</em></p>

| HiACC cohort | n | Base Qwen3-ASR-0.6B | Srota-Conv | Δ (pp) |
|---|---:|---:|---:|---:|
| Adult | 664 | 23.96% | **13.96%** | **-10.00** |
| Children | 372 | 25.61% | **14.73%** | **-10.88** |
| **Overall** | **1,036** | **24.53%** | **14.23%** | **-10.30** |

Children improve slightly more than adults (-10.88 pp vs -10.00 pp), and the adult/child gap stays modest at 0.77 pp post-fine-tune: the model does not introduce a cohort bias.

> **Normalization.** WER is computed with `jiwer` after a symmetric normalizer (lowercase + strip punctuation) is applied to both predictions and references. These numbers are **not directly comparable** to MUCS-style published baselines, which use a different (Kaldi-style) normalization.

## 🚀 Quickstart

Install the inference package, then load Srota-Conv and call `transcribe`. The minimal path is two lines of setup and one call.

```bash
pip install qwen-asr==0.0.6
```

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Surajgameramp/qwen3-asr-0.6b-hinglish-hiacc-v1",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
)

results = model.transcribe(audio="path/to/your.wav", language=None)
print(results[0].text)
# e.g. "मेरा favourite festival Diwali है"
```

- `language=None` enables the language-agnostic decoding prefix Srota-Conv was trained with. Pass it explicitly.
- Audio should be mono; keep segments <= 30 s per call (chunk longer audio).
- bf16 + FlashAttention 2 is recommended; `attn_implementation` can be dropped on CPU or older GPUs.

No setup? Use the [hosted Srota demo](https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo) (runs the union model).

## 🎯 Intended Use and ⚠️ Not for

**Intended use**

- Transcribing **conversational Hinglish**: casual Q&A, storytelling, image-prompted descriptions, interview-style speech.
- Producing natural mixed Devanagari + Latin Hinglish text.
- Adult and child speakers (both cohorts were in training).

**Not for**

- General-purpose Hinglish ASR across both conversational and tutorial / lecture domains. Use [Srota](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish) (the union model) instead; it handles both at once without trading off.
- Technical-tutorial Hinglish (software walkthroughs, dense code/path/version vocabulary). Srota-Conv was not trained on that distribution.
- Monolingual pure-Hindi or pure-English production ASR, where dedicated models are stronger.
- Languages or dialects outside Hindi-English code-switching.
- High-stakes uses (e.g. medical or legal transcription) without human review.

If you are not sure which to pick, default to [Srota](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish).

## 📚 Training Data

Srota-Conv is trained on a single CC BY 4.0 corpus.

- **[HiACC](https://zenodo.org/records/15551669)** (Singh, Singh & Kadyan, 2025; DOI 10.5281/zenodo.15551669, CC BY 4.0): a conversational Hinglish corpus with adult and child speakers, 16 kHz mono WAV.

| Split | Utterances |
|---|---:|
| Train | 3,622 |
| Val | 518 |
| Test | 1,036 |

**Speaker-overlap caveat.** The HiACC train, val, and test splits share speakers. Reported WER is **in-domain, not novel-speaker**: real-world WER on unseen speakers is likely higher than 14.23%.

## 🧠 Training Procedure

Srota-Conv is a **full-parameter fine-tune** of Qwen3-ASR-0.6B: no frozen layers, no LoRA. Every native weight is updated. The "0.6B" in the base model's name refers only to its LLM backbone (Qwen3-0.6B, ~600M parameters); the full speech model is ~780M parameters, because it also includes a ~180M AuT audio encoder and a small projector. That extra ~180M is the audio encoder, **not** a LoRA adapter: all three components (audio encoder, projector, and LLM) are trained end-to-end.

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-ASR-0.6B` |
| Fine-tune scope | Full-parameter (no frozen layers, no LoRA) |
| Fine-tune script | `qwen3_asr_sft.py` @ commit `c17a131f` ([QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)) |
| Optimizer | AdamW |
| Learning rate | 2e-5, linear schedule, warmup_ratio 0.02 |
| Gradient clipping | norm 1.0 |
| Effective batch | 32 (per-device 8 x grad-accum 2 x 2 GPUs) |
| Precision | bf16 + FlashAttention 2 |
| Epochs | 5 (565 steps) |
| Best checkpoint | step 350 (epoch 3.07), eval_loss 0.1917 |
| Hardware | 2x NVIDIA H100 80GB |
| Wall-clock | 618 s (~10 min) |
| Seed | 42 (data shuffle) |

<div align="center">
  <img src="https://huggingface.co/datasets/Surajgameramp/qwen3-asr-0.6b-hinglish-assets/resolve/main/v1_training_curves.png" alt="Srota-Conv training curves: train/eval loss, gradient norm, learning rate" width="90%"/>
</div>

<p align="center"><em>Training/eval loss, gradient norm, and learning rate over 565 steps; eval_loss bottoms out at step 350 (epoch 3.07, eval_loss 0.1917) and then drifts upward, so the early checkpoint is shipped.</em></p>

**Data format.** Targets use the language-agnostic prefix `language None<asr_text>...` (following Polyglot-Lion / Toshniwal et al., 2018), with transcripts kept in their natural mixed Devanagari + Latin script.

## 📈 Evaluation

**Methodology.** For each test utterance, we call `transcribe(audio=…, language=None)`, strip the leading `language ?<asr_text>` prefix, apply the symmetric lowercase + strip-punctuation normalizer to both hypothesis and reference, and compute WER with `jiwer`. Srota-Conv is evaluated on the HiACC test split (1,036 utts) with adult/child cohort slicing. See the Results section above for the full table.

The base Qwen3-ASR-0.6B is evaluated under the exact same pipeline (same normalizer, same `language=None` call, same prefix stripping) on the same 1,036 utterances, so the 24.53% to 14.23% comparison is apples to apples.

## ⚠️ Limitations & Biases

- **5 hours is small.** HiACC's training portion is roughly 5 hours of speech; a larger or more diverse conversational corpus would likely push WER lower and improve speaker generalization.
- **In-domain speaker overlap.** HiACC's train/val/test splits share speakers, so the reported 14.23% is an in-domain number, not a novel-speaker number. Out-of-distribution speakers will be harder.
- **No MUCS-style comparable normalization.** Reported WER uses a symmetric lowercase + strip-punctuation normalizer with `jiwer`, not the Kaldi-style normalizer used by MUCS-2021 published baselines, so numbers here are not directly comparable.
- **Conversational only.** Srota-Conv was not trained on tutorial-style speech and will likely underperform there. The union model, [Srota](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish), handles both.
- **Single seed, single configuration.** No hyperparameter sweep was run; results are from one training run with seed 42.
- **Bias note.** Data is sourced from a specific conversational corpus that includes children and adults; accent, dialect, and domain coverage is limited and may not generalize to all Hinglish varieties.

## 📬 Contact

Questions, feedback, or want Srota-Conv tuned to your use case? Email **surajprasad8977@gmail.com**.

## 📄 License

**Apache-2.0**, inherited from the base Qwen3-ASR-0.6B model. Apache-2.0 is a permissive open-source license: you may use, modify, and redistribute Srota-Conv freely, including for commercial purposes, with no copyleft. The only obligations are to retain the license/copyright notice and to state significant changes.

The training data is licensed CC BY 4.0: HiACC is **CC BY 4.0** (see [the HiACC Zenodo record](https://zenodo.org/records/15551669) for full terms). Users must comply with the dataset license's attribution requirement.

## 📝 Citation

If you use Srota-Conv, please cite this model and the underlying works.

```bibtex
@misc{srota_conv2026,
  title  = {Srota-Conv: A Hinglish conversational ASR specialist fine-tuned from Qwen3-ASR-0.6B on HiACC},
  author = {Suraj},
  year   = {2026},
  url    = {https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish-hiacc-v1}
}

@article{shi2026qwen3asr,
  title  = {Qwen3-ASR Technical Report},
  author = {Shi, Xian and Wang, Xiong and Guo, Zhifang and Wang, Yongqi and
            Zhang, Pei and Zhang, Xinyu and Guo, Zishan and Hao, Hongkun and
            Xi, Yu and Yang, Baosong and Xu, Jin and Zhou, Jingren and
            Lin, Junyang},
  year   = {2026},
  url    = {https://arxiv.org/abs/2601.21337}
}

@article{dang2026polyglot,
  title  = {Polyglot-Lion: Efficient Multilingual ASR for Singapore via
            Balanced Fine-Tuning of Qwen3-ASR},
  author = {Dang, Quy-Anh and Ngo, Chris},
  year   = {2026},
  url    = {https://arxiv.org/abs/2603.16184}
}

@misc{singh2025hiacc,
  title  = {HiACC: Hinglish Adult \& Children Code-switched Corpus},
  author = {Singh, Shruti and Singh, Muskaan and Kadyan, Virender},
  year   = {2025},
  doi    = {10.5281/zenodo.15551669},
  url    = {https://zenodo.org/records/15551669}
}
```

## 🙏 Acknowledgements

Srota-Conv builds directly on the work of others. We thank the **Qwen team** for [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), the base model (a ~780M speech model: ~600M Qwen3-0.6B LLM, ~180M AuT audio encoder, and a small projector, all fully fine-tuned here with no LoRA), and for the open `qwen3_asr_sft.py` training script. We thank the **HiACC** authors (Singh, Singh & Kadyan) for the conversational Hinglish corpus that made this specialist possible. We also thank the authors of **Polyglot-Lion** (Dang & Ngo) for the language-agnostic decoding prefix recipe that this work builds on.

Built by the team behind [susrota.com](https://www.susrota.com/), a voice-dictation tool that currently runs in English. Srota-Conv and its union sibling [Srota](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish) will power its upcoming Hinglish support; the live product does not run this model yet.

Srota-Conv stands entirely on Qwen3-ASR-0.6B; this work is the Hinglish conversational adaptation, not a new foundation model.
