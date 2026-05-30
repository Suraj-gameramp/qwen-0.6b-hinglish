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
  - asr
  - english
  - indic
  - indian-languages
datasets:
  - shrutisingh/HiACC
  - openslr/Hindi-English
metrics:
  - wer
model-index:
  - name: srota
    results:
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: shrutisingh/HiACC
          name: HiACC (conversational Hinglish, test)
        metrics:
          - type: wer
            value: 15.85
            name: WER
      - task:
          type: automatic-speech-recognition
          name: Automatic Speech Recognition
        dataset:
          type: openslr/Hindi-English
          name: OpenSLR-104 / MUCS-2021 (tutorial Hinglish, test)
        metrics:
          - type: wer
            value: 35.06
            name: WER
---

<div align="center">
  <img src="https://huggingface.co/datasets/Surajgameramp/qwen3-asr-0.6b-hinglish-assets/resolve/main/srota_banner.png" alt="Srota: Hinglish ASR" width="100%"/>
</div>

# Srota (श्रोत)

**Srota transcribes Hinglish the way people actually speak it**, keeping English in Latin and Hindi in Devanagari (`मेरा favourite festival Diwali है`), instead of mangling code-switched speech into all-Devanagari transliteration like the base model does.

<div align="center">

### ▶️ [Try Srota live in your browser](https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo) (no install, record or upload a clip)

</div>

<div align="center">

[![Demo](https://img.shields.io/badge/🤗_Demo-Hinglish_ASR-yellow)](https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo)
[![Base model](https://img.shields.io/badge/Base-Qwen3--ASR--0.6B-6633cc)](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
[![HiACC](https://img.shields.io/badge/Data-HiACC-green)](https://zenodo.org/records/15551669)
[![OpenSLR-104](https://img.shields.io/badge/Data-OpenSLR--104-green)](https://openslr.org/104/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![Project](https://img.shields.io/badge/Project-susrota.com-orange)](https://www.susrota.com/)

</div>

## ℹ️ What is Srota?

Srota is an automatic speech recognition (ASR) model for **Hinglish** (Hindi-English code-switched speech) that transcribes into natural **mixed Devanagari + Latin** script. It is a **full-parameter fine-tune of [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)**, trained on the union of conversational and technical-tutorial Hinglish speech. It improves over the base model on both domains at once: **−8.88 pp** word error rate (WER) on conversational speech and **−15.60 pp** on tutorial speech.

**On the size.** The base model's name, Qwen3-ASR-0.6B, refers to its LLM backbone (Qwen3-0.6B, ~600M parameters). The full speech model adds a ~180M AuT audio encoder and a small projector, for ~780M parameters total. Srota is a full-parameter fine-tune of all of them: there are no LoRA adapters and no frozen layers, every native weight is updated.

**Project.** Built by the team behind [susrota.com](https://www.susrota.com/), a voice-dictation tool that currently runs in English. Srota will power its upcoming Hinglish support; the live product does not run this model yet.

Try it in the [live demo](https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo).

## ✨ Highlights

- **Beats the base on both domains.** Conversational HiACC 24.73% → **15.85%**; tutorial OpenSLR-104 50.66% → **35.06%**.
- **One model, two domains.** Unlike a single-domain specialist, Srota does not trade one domain off against the other; it eliminates the negative transfer seen when training on tutorials alone (see the Evaluation section).
- **Native Hinglish output.** Emits Devanagari for Hindi words and Latin for English words, the way Hinglish is actually written (e.g. `मेरा favourite festival Diwali है`).
- **Compact.** ~780M parameters; runs on a single GPU in bf16.
- **Honest lineage.** A full-parameter fine-tune of Qwen3-ASR-0.6B (the ~180M AuT audio encoder, the projector, and the Qwen3-0.6B LLM): no frozen layers, no LoRA adapters. The extra ~180M over the "0.6B" name is the audio encoder, not LoRA.
- **Open.** Apache-2.0; both training corpora are CC BY 4.0.

## 🎧 Srota in action

Real examples from the test set. The base Qwen3-ASR-0.6B transliterates English words into Devanagari (wrong for Hinglish); Srota keeps the natural mixed script.

| | Base Qwen3-ASR-0.6B | Srota |
|---|---|---|
| **A** | `तो डेट्स वाइ आई` | `तो that's why I` |
| **B** | `इन दिहार ऑफ अबस्लिंग सिटी दो सिब्लिंग रहते थे` | `In the heart of a bustling city दो siblings रहते थे` |
| **C** | `ओके सो मेरा होम टाउन न्यू देल्ही है` | `Okay so मेरा hometown New Delhi है` |

The base collapses code-switched English into Devanagari transliteration; Srota preserves how Hinglish is actually written.

Try your own audio in the [live demo](https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo).

## 📊 Results

<div align="center">
  <img src="https://huggingface.co/datasets/Surajgameramp/qwen3-asr-0.6b-hinglish-assets/resolve/main/srota_wer_comparison.png" alt="WER comparison: base vs v1 vs v2 vs Srota on HiACC and OpenSLR-104 test sets" width="80%"/>
</div>

<p align="center"><em>Grouped WER (%) on the HiACC (conversational) and OpenSLR-104 (tutorial) test sets: base model, the two single-domain fine-tunes (v1, v2), and Srota. Lower is better.</em></p>

| Model | HiACC test (conversational, 1,036 utts) | OpenSLR-104 test (tutorial, 3,132 utts) |
|---|---:|---:|
| Qwen3-ASR-0.6B (base, zero-shot) | 24.73% | 50.66% |
| HiACC-only fine-tune (v1) | **14.23%** | ≈ base (untested) |
| OpenSLR-only fine-tune (v2) | 37.64% (worse than base) | **32.83%** |
| **Srota (union, this model)** | **15.85%** | **35.06%** |
| **Srota Δ vs base** | **−8.88 pp** | **−15.60 pp** |

Srota is the only fine-tune that beats the base on **both** test sets. It gives up only ~1.6 pp versus the conversational specialist and ~2.2 pp versus the tutorial specialist, the expected, small generalist trade-off.

| HiACC cohort | n | Srota WER |
|---|---:|---:|
| Adult | 664 | 15.41% |
| Children | 372 | 16.66% |
| Overall | 1,036 | **15.85%** |

The adult/child gap stays gentle (1.25 pp): the union introduced no cohort bias.

> **Normalization.** WER is computed with `jiwer` after a symmetric normalizer (lowercase + strip punctuation) is applied to both predictions and references. These numbers are **not directly comparable** to MUCS-2021 published baselines, which use a different (Kaldi-style) normalization.

## 🚀 Quickstart

Install the inference package, then load Srota and call `transcribe`. The minimal path is two lines of setup and one call.

```bash
pip install qwen-asr==0.0.6
```

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Surajgameramp/qwen3-asr-0.6b-hinglish",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
)

results = model.transcribe(audio="path/to/your.wav", language=None)
print(results[0].text)
# e.g. "मेरा favourite festival Diwali है"
```

- `language=None` enables the language-agnostic decoding prefix Srota was trained with. Pass it explicitly.
- Audio should be mono; keep segments ≤ 30 s per call (chunk longer audio).
- bf16 + FlashAttention 2 is recommended; `attn_implementation` can be dropped on CPU or older GPUs.

No setup? Use the [hosted demo](https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo).

## 🎯 Intended Use & Limitations of Use

**Intended use**

- Transcribing conversational Hinglish (casual Q&A, storytelling, image-prompted descriptions).
- Transcribing technical-tutorial Hinglish (software walkthroughs, lecture-style instruction).
- Producing natural mixed Devanagari + Latin Hinglish text.

**Out of scope / not recommended**

- Monolingual pure-Hindi or pure-English production ASR, where dedicated models are stronger.
- Languages or dialects outside Hindi-English code-switching.
- High-stakes uses (e.g. medical or legal transcription) without human review.

Full failure modes are described in the Limitations & Biases section below.

## 📚 Training Data

Srota is trained on the **union** of two CC BY 4.0 Hinglish corpora, simply concatenated with **no upsampling**.

- **[HiACC](https://zenodo.org/records/15551669)** (Singh, Singh & Kadyan, 2025; DOI 10.5281/zenodo.15551669, CC BY 4.0): **5.24 h** of conversational Hinglish, 16 kHz mono WAV.
- **[OpenSLR-104](https://openslr.org/104/)** (the MUCS-2021 Multilingual & Code-Switching ASR challenge; CC BY 4.0): **89.86 h** of Hindi-English spoken-tutorial speech from the IIT Bombay Spoken Tutorial project.

| Split | Utterances | Composition |
|---|---:|---|
| Train | 53,627 | HiACC 6.8% + OpenSLR-104 93.2% |
| Val | 3,282 | 518 HiACC + 2,764 OpenSLR-104 |

Each corpus's own official test set is used for evaluation, reported separately in the Results section above.

HiACC is only **6.8%** of the training mix, yet Srota retains ~99% of the conversational specialist's quality (15.85% vs 14.23%): **balanced upsampling was unnecessary at this scale**; a deterministic shuffle (seed 42) was enough.

## 🧠 Training Procedure

Srota is a **full-parameter fine-tune** of Qwen3-ASR-0.6B: no frozen layers, no LoRA. Every native weight is updated. The "0.6B" in the base model's name refers only to its LLM backbone (Qwen3-0.6B, ~600M parameters); the full speech model is ~780M parameters, because it also includes a ~180M AuT audio encoder and a small projector. That extra ~180M is the audio encoder, **not** a LoRA adapter: all three components (audio encoder, projector, and LLM) are trained end-to-end.

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-ASR-0.6B` |
| Fine-tune scope | Full-parameter (no frozen layers, no LoRA) |
| Fine-tune script | `qwen3_asr_sft.py` @ commit `c17a131f` ([QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)) |
| Optimizer | AdamW |
| Learning rate | 2e-5, linear schedule, warmup_ratio 0.02 |
| Gradient clipping | norm 1.0 |
| Effective batch | 32 (per-device 8 × grad-accum 2 × 2 GPUs) |
| Precision | bf16 + FlashAttention 2 |
| Epochs | 2 (3,352 steps) |
| Best checkpoint | step 3200 (epoch 1.91), eval_loss 0.1500 |
| Hardware | 2× NVIDIA H100 80GB |
| Wall-clock | ~49 min (2,943 s) |
| Seed | 42 (data shuffle) |

**Data format.** Targets use the language-agnostic prefix `language None<asr_text>...` (following Polyglot-Lion / Toshniwal et al., 2018), with transcripts kept in their natural mixed Devanagari + Latin script.

<div align="center">
  <img src="https://huggingface.co/datasets/Surajgameramp/qwen3-asr-0.6b-hinglish-assets/resolve/main/training_curves.png" alt="Srota training curves: train/eval loss, gradient norm, learning rate" width="90%"/>
</div>

<p align="center"><em>Training/eval loss, gradient norm, and learning rate over 3,352 steps; eval_loss bottoms out at step 3200 (epoch 1.91) and stays flat, confirming 2 epochs.</em></p>

## 📈 Evaluation

**Methodology.** For each test utterance, we call `transcribe(audio=…, language=None)`, strip the leading `language ?<asr_text>` prefix, apply the symmetric lowercase + strip-punctuation normalizer to both hypothesis and reference, and compute WER with `jiwer`. Srota is evaluated on the HiACC test split (with adult/child cohort slicing) and the OpenSLR-104 official test split. See the Results section above for the full comparison table.

**Union vs. specialists.** A tutorial-only fine-tune (v2) gained −17.82 pp in-domain on OpenSLR-104 but **regressed +12.91 pp versus the base on conversational HiACC**, classic negative transfer, since lectures and spontaneous conversation are far apart distributionally. Adding back HiACC's 5.24 h of conversational speech (even at only 6.8% of the mix) re-anchors the model: Srota turns that +12.91 pp HiACC regression into a **−8.88 pp improvement** (a −21.79 pp swing versus v2 on HiACC) while keeping −15.60 pp on OpenSLR-104. Srota is the shippable generalist; the single-domain specialists are not drop-in replacements for the base across both domains.

## ⚠️ Limitations & Biases

- **Generalist trade-off.** Srota is ~1.6 pp behind the conversational specialist on HiACC and ~2.2 pp behind the tutorial specialist on OpenSLR-104. For a single known domain, a specialist is marginally better.
- **Tutorial WER is still substantial (35.06%).** Dense code/path/version vocabulary (`bash`, `gnu/linux`, `version 1204`) remains hard for a 0.6B model.
- **Not comparable to MUCS-2021 published numbers** without matching their Kaldi-style normalization.
- **Single seed, single configuration.** No hyperparameter sweep was run; the "upsampling unnecessary" claim is observed, not proven via a controlled concat-vs-upsampled ablation.
- **HiACC train/val/test share speakers.** Reported HiACC WER is in-domain, not novel-speaker: real-world conversational WER on unseen speakers may be higher.
- **Bias note.** Data is sourced from specific corpora (Indian spoken-tutorial speech and a defined conversational set that includes children); accent, dialect, and domain coverage is limited and may not generalize to all Hinglish varieties.

## 📬 Contact

Questions, feedback, or want Srota tailored to your use case? Email **surajprasad8977@gmail.com**.

## 📄 License

**Apache-2.0**, inherited from the base Qwen3-ASR-0.6B model. Apache-2.0 is a permissive open-source license: you may use, modify, and redistribute Srota freely, including for commercial purposes, with no copyleft. The only obligations are to retain the license/copyright notice and to state significant changes.

The training data is licensed CC BY 4.0: HiACC is **CC BY 4.0**, and OpenSLR-104 is **CC BY 4.0** (see [openslr.org/104](https://openslr.org/104/) for full terms). Users must comply with the dataset licenses' attribution requirements.

## 📝 Citation

If you use Srota, please cite this model and the underlying works.

```bibtex
@misc{srota2026,
  title  = {Srota: A Hinglish ASR model fine-tuned from Qwen3-ASR-0.6B},
  author = {Suraj},
  year   = {2026},
  url    = {https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish}
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

@inproceedings{diwan2021mucs,
  title     = {{MUCS} 2021: Multilingual and Code-Switching {ASR} Challenges for Low Resource {Indian} Languages},
  author    = {Diwan, Anuj and Vaideeswaran, Rakesh and Shah, Sanket and others},
  booktitle = {Proc. Interspeech 2021},
  year      = {2021}
}

@inproceedings{toshniwal2018multilingual,
  title     = {Multilingual speech recognition with a single end-to-end model},
  author    = {Toshniwal, Shubham and Sainath, Tara N. and Weiss, Ron J. and
               Li, Bo and Moreno, Pedro and Weinstein, Eugene and Rao, Kanishka},
  booktitle = {2018 IEEE International Conference on Acoustics, Speech and
               Signal Processing (ICASSP)},
  pages     = {4904--4908},
  year      = {2018},
  doi       = {10.1109/ICASSP.2018.8461972}
}
```

## 🙏 Acknowledgements

Srota builds directly on the work of others. We thank the **Qwen team** for [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), the base model, and for the open `qwen3_asr_sft.py` training script. We thank the **HiACC** authors (Singh, Singh & Kadyan) and the **MUCS-2021 / OpenSLR-104 / IIT Bombay Spoken Tutorial** contributors for the training and evaluation data. We also thank the authors of **Polyglot-Lion** (Dang & Ngo) for the balanced-fine-tuning recipe and language-agnostic decoding prefix that this work builds on.

Srota stands entirely on Qwen3-ASR-0.6B; this work is the Hinglish adaptation, not a new foundation model.
