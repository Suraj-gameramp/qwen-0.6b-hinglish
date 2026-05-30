---
license: apache-2.0
language:
- hi
- en
base_model: Qwen/Qwen3-ASR-0.6B
tags:
- automatic-speech-recognition
- code-switching
- hinglish
- speech
datasets:
- shrutisingh/HiACC
- openslr/Hindi-English
metrics:
- wer
library_name: transformers
pipeline_tag: automatic-speech-recognition
---

# qwen3-asr-0.6b-hinglish-union-v3

A fine-tune of [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) on the **union of HiACC (conversational) + MUCS-2021 / OpenSLR-104 (technical tutorials)** Hindi-English code-switched speech (95.10 h combined training data).

## ✅ This is the recommended variant

This is the **general-purpose Hinglish model** of the series. It is trained on the union of both datasets and **improves over the base Qwen3-ASR-0.6B on BOTH domains at once** — conversational speech and technical-tutorial speech (see results below).

Unlike the v2 OpenSLR specialist, **v3 does not regress on conversational speech**: it bridges the two domains instead of trading one off against the other. If you want a single model that just works across conversational and tutorial Hinglish, use this one.

If you instead want to squeeze out the last 1-2 pp in a single, known domain, the domain-specialists are still available:
- [`Surajgameramp/qwen3-asr-0.6b-hinglish-hiacc-v1`](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish-hiacc-v1) — conversational (HiACC) specialist.
- [`Surajgameramp/qwen3-asr-0.6b-hinglish-openslr104-v2`](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish-openslr104-v2) — technical-tutorial (OpenSLR-104) specialist (note: regresses on conversational speech).

## Result matrix

|                                | HiACC test (conversational) | OpenSLR-104 test (tutorials) |
|--------------------------------|----------------------------:|-----------------------------:|
| Base Qwen3-ASR-0.6B (0-shot)   |                      24.53% |                       50.66% |
| v1 (HiACC specialist)          |                      14.23% |                  ~50% (≈ base) |
| v2 (OpenSLR specialist)        |          37.64% (worse than base!) |                32.83% |
| **This model (v3, union)**     |                  **15.85%** |                   **35.06%** |
| Δ vs base (v3)                 |             **−8.88 pp**    |              **−15.60 pp**    |

v3 is the only variant that beats the base model on **both** test sets. It sacrifices only ~1.6 pp vs the HiACC specialist on conversational speech and ~2.2 pp vs the OpenSLR specialist on tutorial speech, in exchange for being usable across both domains.

HiACC cohort split for v3: adult **15.41%**, children **16.66%**.

WER computed with `jiwer` after lowercase + punctuation-stripping normalization applied symmetrically to predictions and references.

## Intended use

- Transcribing **both** conversational Hindi-English code-switched speech (casual Q&A, storytelling, image-prompted descriptions) **and** technical-tutorial Hinglish (software walkthroughs, lecture-style instruction).
- Mixed Devanagari + Latin script output is the natural format — the model emits Devanagari for Hindi words and Latin for English words, matching how Hinglish is typically written.

## Usage

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Surajgameramp/qwen3-asr-0.6b-hinglish-union-v3",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2",
)

results = model.transcribe(audio="path/to/your.wav", language=None)
print(results[0].text)
# Example: "मेरा favourite festival Diwali है"
```

Inference requires the `qwen-asr` package (`pip install qwen-asr==0.0.6`).

## Training data

The **union** of two Hindi-English code-switched corpora, simply concatenated with **no upsampling**:

- [HiACC](https://zenodo.org/records/15551669) (Singh, Singh & Kadyan, 2025): 5.24 h of conversational Hinglish, 16 kHz mono WAV. CC BY 4.0.
- [OpenSLR-104 Hindi-English](https://openslr.org/104/) (released for the MUCS-2021 Multilingual & Code-Switching ASR challenge): 89.86 h of Hindi-English code-switched spoken-tutorial speech from IIT Bombay's Spoken Tutorial project. CC BY 4.0.

Combined splits:
- **Train** 53,627 utterances (HiACC 6.8% + OpenSLR-104 93.2%)
- **Val** 3,282 utterances

Each corpus's own test set is used for evaluation, reported separately in the result matrix above.

## Training procedure

- **Base model**: `Qwen/Qwen3-ASR-0.6B`
- **Script**: `qwen3_asr_sft.py` from [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) at commit `c17a131f`
- **Fine-tune scope**: full-parameter (no frozen layers, no LoRA — all AuT encoder + projector + LLM weights updated)
- **Hardware**: 2× NVIDIA H100 80GB on Modal
- **Optimizer**: AdamW, LR 2e-5, linear schedule, warmup_ratio 0.02
- **Effective batch**: 32 (per-device 8 × grad-accum 2 × 2 GPUs)
- **Precision**: bf16 + FlashAttention 2
- **Epochs**: 2 (3,352 total steps); best checkpoint by `eval_loss` was step 3200 (epoch 1.91, eval_loss=0.1500)
- **Wall-clock**: 2,943 s (~49 min)

### Data format

Each training example uses `language None<asr_text>...` as the target prefix (language-agnostic decoding, following [Polyglot-Lion](https://arxiv.org/abs/2603.16184) which adapts [Toshniwal et al., 2018](https://arxiv.org/abs/1711.01694)). Transcripts are kept in their natural mixed-script form (Devanagari for Hindi, Latin for English).

## Limitations

- **Generalist trade-off**: slightly worse than each specialist in its own domain (~1.6 pp behind v1 on HiACC, ~2.2 pp behind v2 on OpenSLR-104).
- **OpenSLR-104 / tutorial test WER (35.06%) is still substantial.** Tutorial speech with dense code/path/version vocabulary remains harder than conversational data.
- **Single-seed, single-config training.** No hyperparameter sweep.
- **WERs not directly comparable to MUCS-2021 published baselines** without matching their Kaldi-style normalization conventions.

## Citation

If you use this model, please cite the underlying works:

```bibtex
@article{shi2026qwen3asr,
  title   = {Qwen3-ASR Technical Report},
  author  = {Shi, Xian and Wang, Xiong and others},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21337}
}

@article{dang2026polyglot,
  title   = {Polyglot-Lion: Efficient Multilingual ASR for Singapore via
             Balanced Fine-Tuning of Qwen3-ASR},
  author  = {Dang, Quy-Anh and Ngo, Chris},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.16184}
}

@misc{singh2025hiacc,
  title  = {HiACC: Hinglish Adult \& Children Code-switched Corpus},
  author = {Singh, Shruti and Singh, Muskaan and Kadyan, Virender},
  year   = {2025},
  doi    = {10.5281/zenodo.15551669},
  url    = {https://zenodo.org/records/15551669}
}

@inproceedings{diwan2021mucs,
  title     = {{MUCS} 2021: Multilingual and Code-Switching {ASR} Challenges for Low Resource Indian Languages},
  author    = {Diwan, Anuj and Vaideeswaran, Rakesh and Shah, Sanket and others},
  booktitle = {Proc. Interspeech 2021},
  year      = {2021}
}
```

## License

Apache 2.0, inherited from the base Qwen3-ASR-0.6B model. HiACC training data is CC BY 4.0; OpenSLR-104 corpus is CC BY 4.0 (see [openslr.org/104](https://openslr.org/104/) for full license terms).
