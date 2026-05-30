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
  - tutorial-speech
  - openslr-104
  - mucs-2021
  - asr
  - english
  - indic
  - indian-languages
datasets:
  - openslr/Hindi-English
metrics:
  - wer
model-index:
  - name: Srota-Tutorial
    results:
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: openslr/Hindi-English
          name: OpenSLR-104 / MUCS-2021 (tutorial Hinglish, test, in-domain)
        metrics:
          - type: wer
            value: 32.83
            name: WER
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: shrutisingh/HiACC
          name: HiACC (conversational Hinglish, test, cross-domain regression vs base)
        metrics:
          - type: wer
            value: 37.64
            name: WER
---

<div align="center">
  <img src="https://huggingface.co/datasets/moorlee/srota-assets/resolve/main/v2_banner.png" alt="Srota-Tutorial: Hinglish Tutorial ASR" width="100%"/>
</div>

# Srota-Tutorial (श्रोत): Hinglish Tutorial Specialist (OpenSLR-104)

**Srota-Tutorial is the OpenSLR-104 tutorial specialist sibling of [Srota](https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish).** It is a full-parameter fine-tune of [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) on Hindi-English spoken-tutorial speech (lectures, software walkthroughs, technical instruction). It cuts WER on OpenSLR-104 tutorials from 50.66% to **32.83%** (a 35% relative drop), but it pays for that with a measurable regression on conversational Hinglish.

<div align="center">

### ▶️ [Try the union model in your browser](https://huggingface.co/spaces/moorlee/hinglish-asr-demo)

Srota (the union model) covers both tutorial and conversational speech; Srota-Tutorial is the tutorial specialist that gives up conversational quality for in-domain gains.

</div>

<div align="center">

[![Demo](https://img.shields.io/badge/🤗_Demo-Hinglish_ASR-yellow)](https://huggingface.co/spaces/moorlee/hinglish-asr-demo)
[![Base model](https://img.shields.io/badge/Base-Qwen3--ASR--0.6B-6633cc)](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
[![OpenSLR-104](https://img.shields.io/badge/Data-OpenSLR--104-green)](https://openslr.org/104/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![Project](https://img.shields.io/badge/Project-susrota.com-orange)](https://www.susrota.com/)
[![Family](https://img.shields.io/badge/Family-Srota-ff69b4)](https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish)

</div>

> ⚠️ **This is a specialist.** Srota-Tutorial is tuned for technical tutorial speech only. On conversational Hinglish it is **+12.91 pp WORSE** than the base model on HiACC (37.64% vs 24.73%). For general Hinglish, use [Srota](https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish) instead.

## ℹ️ What is Srota-Tutorial?

Srota-Tutorial is an automatic speech recognition (ASR) model for **Hindi-English code-switched tutorial speech**: software walkthroughs, lectures, and step-by-step technical instruction from the [IIT Bombay Spoken Tutorial](https://spoken-tutorial.org/) project, as packaged in OpenSLR-104 / MUCS-2021. It is a **full-parameter fine-tune of [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)** trained on OpenSLR-104 alone.

**On the size.** The base model's name, Qwen3-ASR-0.6B, refers to its LLM backbone (Qwen3-0.6B, ~600M parameters). The full speech model adds a ~180M AuT audio encoder and a small projector, for ~780M parameters total. Srota-Tutorial is a full-parameter fine-tune of all of them: there are no LoRA adapters and no frozen layers, every native weight is updated. The extra ~180M over the "0.6B" name is the audio encoder, not a LoRA adapter.

**Sibling model.** For general Hinglish (conversational + tutorial), see [Srota](https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish), the union model. Srota-Tutorial only exists to document the in-domain ceiling and the cross-domain cost of single-domain fine-tuning; Srota is the shippable generalist.

**Project.** Built by the team behind [susrota.com](https://www.susrota.com/), a voice-dictation tool that currently runs in English. Srota will power its upcoming Hinglish support; the live product does not run this model yet.

## ✨ Highlights

- **Large in-domain win.** OpenSLR-104 test WER drops from 50.66% (base) to **32.83%** (−17.83 pp, −35% relative).
- **Preserves natural code-switch.** Keeps English jargon in Latin (`tutorial`, `print button`, `slides handouts notes`) and Hindi narration in Devanagari, instead of romanizing or hallucinating English continuations like the base.
- **Compact.** ~780M parameters total (Qwen3-0.6B LLM + ~180M AuT audio encoder + projector); single-GPU bf16 inference.
- **Honest lineage.** Full-parameter fine-tune of Qwen3-ASR-0.6B: no frozen layers, no LoRA adapters. The extra ~180M over the "0.6B" name is the audio encoder, not LoRA.
- **Specialist trade-off (read this).** Conversational HiACC test WER goes from 24.73% (base) to **37.64%** (**+12.91 pp WORSE than base**). This is a classic single-domain negative-transfer regression, and it is the entire reason the union model [Srota](https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish) exists.
- **Open.** Apache-2.0; training data is OpenSLR-104 (CC BY 4.0).

## ⚠️ Read before downloading

**Srota-Tutorial is a domain specialist, not a drop-in replacement for the base model.** On conversational Hinglish (HiACC test), it scores **37.64% WER, which is +12.91 pp WORSE than Qwen3-ASR-0.6B's 24.73%**. If your audio is anything other than technical Hindi-English tutorial speech (lectures, software walkthroughs), use [Srota](https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish) (the union model) or the base [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) instead.

Additionally, because OpenSLR-104 transcripts are lowercase and unpunctuated by design, **this model emits lowercase, no-punctuation, mixed-script text**. It is not production-formatted output.

## 🎧 Srota-Tutorial in action

Real OpenSLR-104 test-set examples. On tutorial speech, the base model hallucinates English completions or romanizes everything into Devanagari. Srota-Tutorial transcribes what was actually said, preserving the natural code-switch (English jargon in Latin, Hindi narration in Devanagari).

| | Base Qwen3-ASR-0.6B | Srota-Tutorial |
|---|---|---|
| **A** | `In the tutorial, we have seen storage class specifiers, auto keyword, static keyword, extern keyword, register keyword.` | `इस tutorial में हमने सीखा:` |
| **B** | `हम इस वर्ग में नहीं करेंगे अब प्रिंट बटन पर क्लिक` | `अब print button पर click करें` |
| **C** | `प्रिंटिंग के बारे में सीखा, स्लाइड्स, हैंडओउट्स, नोट्स और आउटलाइन` | `slides handouts notes और outline` |

In **A**, the base ignores the actual short Hindi phrase and hallucinates a fluent English summary. In **B**, the base prepends invented content before getting to the command. In **C**, the base romanizes English jargon into Devanagari (`स्लाइड्स`, `हैंडओउट्स`); Srota-Tutorial keeps English in Latin (`slides handouts notes`) the way it appears in the reference transcript.

## 📊 Results

<div align="center">
  <img src="https://huggingface.co/datasets/moorlee/srota-assets/resolve/main/v2_wer_comparison.png" alt="WER comparison: base Qwen3-ASR-0.6B vs Srota-Tutorial on OpenSLR-104 (in-domain) and HiACC (cross-domain)" width="80%"/>
</div>

<p align="center"><em>WER (%) on OpenSLR-104 test (in-domain, tutorial) and HiACC test (cross-domain, conversational). Srota-Tutorial wins big in-domain (50.66 to 32.83) but loses badly out-of-domain (24.73 to 37.64, +12.91 pp WORSE than base). Lower is better.</em></p>

| Test set | Domain | n utts | Base Qwen3-ASR-0.6B | Srota-Tutorial | Δ vs base |
|---|---|---:|---:|---:|---:|
| OpenSLR-104 test | Tutorial (in-domain) | 3,132 | 50.66% | **32.83%** | **−17.83 pp** (−35% rel) |
| HiACC test | Conversational (cross-domain) | 1,036 | 24.73% | 37.64% | **+12.91 pp** (worse) |

> **Normalization.** WER is computed with `jiwer` after a symmetric normalizer (lowercase + strip punctuation) is applied to both predictions and references. These numbers are **not directly comparable** to MUCS-2021 published baselines, which use a different (Kaldi-style) normalization.

The OpenSLR-104 gain is real and large, but the HiACC regression is also real and large: a tutorial-only fine-tune at this scale meaningfully damages conversational performance. This is the central evidence that motivates the union model [Srota](https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish).

## 🚀 Quickstart

Install the inference package, then load Srota-Tutorial and call `transcribe`.

```bash
pip install qwen-asr==0.0.6
```

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "moorlee/qwen3-asr-0.6b-hinglish-openslr104-v2",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
)

results = model.transcribe(audio="path/to/tutorial.wav", language=None)
print(results[0].text)
# e.g. "इस tutorial में हम nested और multilevel if statement के बारे में सीखेंगे"
```

- `language=None` enables the language-agnostic decoding prefix this model was trained with. Pass it explicitly.
- Audio should be mono; keep segments under 30 s per call (chunk longer audio).
- bf16 + FlashAttention 2 is recommended; `attn_implementation` can be dropped on CPU or older GPUs.
- **Output style.** OpenSLR-104 references are lowercase and unpunctuated by design, so this model emits lowercase, no-punctuation, mixed Devanagari + Latin text. Apply your own casing and punctuation if you need production-formatted output.

## 🎯 Intended Use

**Intended use**

- Transcribing **Hindi-English spoken tutorials**: software walkthroughs, lecture-style technical instruction, step-by-step product demos, in the same distribution as the IIT Bombay Spoken Tutorial / OpenSLR-104 corpus.
- Research baseline for in-domain fine-tuning on OpenSLR-104 / MUCS-2021.
- Producing lowercase, no-punctuation mixed Devanagari + Latin Hinglish text (the OpenSLR-104 transcript style).

**Out of scope / not recommended**

- **General conversational Hinglish.** This model is **+12.91 pp WORSE** than the base on HiACC. Use [Srota](https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish) (the union model) for conversational or mixed-domain audio.
- Production text needing case or punctuation without a post-processing layer.
- Monolingual pure-Hindi or pure-English ASR.
- High-stakes uses (medical, legal) without human review.

Full failure modes are described in the Limitations & Biases section below.

## 📚 Training Data

Srota-Tutorial is trained on **[OpenSLR-104](https://openslr.org/104/)** alone (the MUCS-2021 Multilingual & Code-Switching ASR challenge Hindi-English subtask; CC BY 4.0): **89.86 h** of Hindi-English spoken-tutorial speech from the IIT Bombay Spoken Tutorial project, 16 kHz mono WAV. Transcripts are lowercase and unpunctuated by design.

| Split | Utterances | Notes |
|---|---:|---|
| Train | 50,005 | OpenSLR-104 train |
| Val | 2,764 | **Speaker-disjoint** from train: 26 of 520 train speakers held out |
| Test | 3,132 | Official OpenSLR-104 test |

The training audio is sourced from long-form tutorial recordings that were chunked into utterance-length segments before fine-tuning, then re-joined at evaluation time per the official splits.

## 🧠 Training Procedure

Srota-Tutorial is a **full-parameter fine-tune** of Qwen3-ASR-0.6B: no frozen layers, no LoRA. Every native weight is updated. The "0.6B" in the base model's name refers only to its LLM backbone (Qwen3-0.6B, ~600M parameters); the full speech model is ~780M parameters because it also includes a ~180M AuT audio encoder and a small projector. That extra ~180M is the audio encoder, **not** a LoRA adapter: all three components (audio encoder, projector, and LLM) are trained end-to-end.

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-ASR-0.6B` |
| Fine-tune scope | Full-parameter (no frozen layers, no LoRA) |
| Fine-tune script | `qwen3_asr_sft.py` ([QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)) |
| Optimizer | AdamW |
| Learning rate | 2e-5, linear schedule, warmup_ratio 0.02 |
| Gradient clipping | norm 1.0 |
| Effective batch | 32 (per-device 8 × grad-accum 2 × 2 GPUs) |
| Precision | bf16 + FlashAttention 2 |
| Epochs | 3 (4,690 steps) |
| Best checkpoint | step 3000 (epoch 1.92), eval_loss 0.1436 |
| Hardware | 2× NVIDIA H100 80GB |
| Wall-clock | ~72 min (4,351 s) |
| Seed | 42 (data shuffle) |

**Data format.** Targets use the language-agnostic prefix `language None<asr_text>...` (following Polyglot-Lion / Toshniwal et al., 2018), with transcripts kept in OpenSLR-104's native lowercase mixed Devanagari + Latin script.

<div align="center">
  <img src="https://huggingface.co/datasets/moorlee/srota-assets/resolve/main/v2_training_curves.png" alt="Srota-Tutorial training curves: train/eval loss, gradient norm, learning rate" width="90%"/>
</div>

<p align="center"><em>Training/eval loss, gradient norm, and learning rate over 4,690 steps; eval_loss bottoms out at step 3000 (epoch 1.92, eval_loss 0.1436), with the later epoch showing no further improvement.</em></p>

## 📈 Evaluation

**Methodology.** For each test utterance, we call `transcribe(audio=…, language=None)`, strip the leading `language ?<asr_text>` prefix, apply the symmetric lowercase + strip-punctuation normalizer to both hypothesis and reference, and compute WER with `jiwer`. Srota-Tutorial was evaluated on **both** test sets to surface cross-domain transfer behavior: in-domain on OpenSLR-104 test (3,132 utts) and cross-domain on HiACC test (1,036 utts).

**In-domain (OpenSLR-104 test).** WER drops from **50.66% (base)** to **32.83% (Srota-Tutorial)**, a −17.83 pp absolute / −35% relative improvement. This is the headline in-domain result.

**Cross-domain (HiACC test).** WER goes from **24.73% (base)** to **37.64% (Srota-Tutorial)**, a **+12.91 pp regression**: this model is meaningfully worse than the base on conversational Hinglish. This is the **why-Srota-exists** result: a tutorial-only fine-tune at this scale negatively transfers to conversational speech, which is precisely what the union model [Srota](https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish) was built to fix (it converts that +12.91 pp HiACC regression into a −8.88 pp improvement).

## ⚠️ Limitations & Biases

- **Cross-domain regression.** On conversational HiACC, Srota-Tutorial is **+12.91 pp worse** than the base Qwen3-ASR-0.6B (37.64% vs 24.73%). Do not use it on non-tutorial audio; use [Srota](https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish) instead.
- **Lowercase, no-punctuation output.** OpenSLR-104 transcripts are lowercase and unpunctuated by design, so the model emits the same. It is not production-formatted; a casing/punctuation post-processor is required for downstream display.
- **In-domain WER is still substantial (32.83%).** Dense technical vocabulary (commands, file paths, version strings) and rapid Hindi-English code-switching remain hard for a ~780M-parameter model, even after a 35% relative reduction.
- **Not comparable to MUCS-2021 published numbers** without matching their Kaldi-style normalization.
- **Single seed, single configuration.** No hyperparameter sweep was run.
- **Bias note.** All training audio comes from the IIT Bombay Spoken Tutorial project: a specific Indian-accented, lecture-style register. Accent, dialect, speaking-style, and topic coverage outside that distribution may degrade quickly (the HiACC result is a concrete example).

## 📬 Contact

Questions, feedback, or want Srota-Tutorial tailored to your use case? Email **surajprasad8977@gmail.com**.

## 📄 License

**Apache-2.0**, inherited from the base Qwen3-ASR-0.6B model. Apache-2.0 is a permissive open-source license: you may use, modify, and redistribute Srota-Tutorial freely, including for commercial purposes, with no copyleft. The only obligations are to retain the license/copyright notice and to state significant changes.

The training data is licensed CC BY 4.0: OpenSLR-104 is **CC BY 4.0** (see [openslr.org/104](https://openslr.org/104/) for full terms). Users must comply with the dataset license's attribution requirements.

## 📝 Citation

If you use Srota-Tutorial, please cite this model and the underlying works.

```bibtex
@misc{srota_tutorial2026,
  title  = {Srota-Tutorial: A Hinglish tutorial-speech ASR model fine-tuned from Qwen3-ASR-0.6B on OpenSLR-104},
  author = {Suraj},
  year   = {2026},
  url    = {https://huggingface.co/moorlee/qwen3-asr-0.6b-hinglish-openslr104-v2}
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

@inproceedings{diwan2021mucs,
  title     = {{MUCS} 2021: Multilingual and Code-Switching {ASR} Challenges for Low Resource {Indian} Languages},
  author    = {Diwan, Anuj and Vaideeswaran, Rakesh and Shah, Sanket and others},
  booktitle = {Proc. Interspeech 2021},
  year      = {2021}
}
```

## 🙏 Acknowledgements

Srota-Tutorial builds directly on the work of others. We thank the **Qwen team** for [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), the base model, and for the open `qwen3_asr_sft.py` training script. We thank the **IIT Bombay Spoken Tutorial** project and the **MUCS-2021 / OpenSLR-104** organizers for the training and evaluation data. We also thank the authors of **Polyglot-Lion** (Dang & Ngo) for the language-agnostic decoding prefix that this work builds on.

Built by the team behind [susrota.com](https://www.susrota.com/), a voice-dictation tool that currently runs in English. Srota will power its upcoming Hinglish support; the live product does not run this model yet.

Srota-Tutorial stands entirely on Qwen3-ASR-0.6B; this work is the OpenSLR-104 tutorial-domain adaptation, not a new foundation model.
