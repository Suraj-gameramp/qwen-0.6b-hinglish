# V2 Results — Qwen3-ASR-0.6B fine-tuned on OpenSLR-104

## Headline

Fine-tuning `Qwen/Qwen3-ASR-0.6B` on 89.86 h of OpenSLR-104 Hindi-English tutorial speech yields a strong **in-domain gain** (−17.8 pp WER on OpenSLR test) but a **negative cross-domain transfer** to HiACC's conversational Hinglish (+12.9 pp WER, i.e. *worse* than the base model). This is the classic specialist trade-off: lectures and conversations are far enough apart that an 89 h-trained specialist overfits its domain.

The clean takeaway: **v1 ≠ v2 ≠ shippable model**. Both are specialists; v3 (union) is the experiment designed to bridge them.

## Full 2×2 result matrix

|                            | OpenSLR-104 test (in-domain, tutorials) | HiACC test (cross-domain, conversational) |
|----------------------------|----------------------------------------:|------------------------------------------:|
| **Base (Qwen3-ASR-0.6B, zero-shot)** | 50.66%                       | 24.73%                                    |
| **v2 fine-tuned** (this work) | **32.83%**                   | **37.64%**                                |
| Δ vs base                  | **−17.82 pp** (−35% relative)           | **+12.91 pp** (negative transfer)         |
| Δ pp adult / children      | — / —                                   | +12.0 / +14.6                             |

For reference (from v1 eval, same base, same HiACC test split):

| Model                                 | HiACC test WER |
|---------------------------------------|---------------:|
| Base Qwen3-ASR-0.6B (zero-shot)       | 24.53%         |
| **v1** (HiACC fine-tune)              | **14.23%**     |
| **v2** (OpenSLR fine-tune, this work) | 37.64%         |

The base WER on HiACC matches across runs (24.53% vs 24.73%) — consistency check.

## Why two specialists diverge

| Aspect | v1 (HiACC) | v2 (OpenSLR-104) |
|---|---|---|
| Audio domain | Everyday Q&A + storytelling + image prompts | Technical spoken-tutorial lectures |
| Train hours | 5.24 h (3,622 utts) | 89.86 h (50,005 utts) |
| Distinct speakers | ~60 (with leakage) | 520 (disjoint splits) |
| Label style | Mixed-script, cased, punctuated | Mixed-script, lowercase, no punctuation |
| Vocabulary | Conversational Hinglish | Heavy tech English-in-Hindi (`bash`, `tutorial`, `gnu/linux`, `version 1204`) |
| Best val_loss | 0.1917 (step 350, epoch 3.07) | 0.1436 (step 3000, epoch 1.92) |

Two reasons v2 hurts HiACC:

1. **Distribution shift.** OpenSLR-104 is dominated by technical jargon and acoustic patterns of a lecturer reading from a script. HiACC is spontaneous everyday speech. The model adapts to the lecture distribution and degrades on conversational input.
2. **Output-schema shift.** OpenSLR labels are lowercased and stripped of punctuation, so v2's output schema matches that. After eval-time normalization both refs and hyps go through the same lowercase+strip-punct, but the model's *internal* representation may have shifted in ways that affect word choice and segmentation on HiACC's natural speech (e.g. compound word boundaries, contractions).

## Configuration

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-ASR-0.6B` |
| Fine-tune script | `qwen3_asr_sft.py` @ commit `c17a131f` from `QwenLM/Qwen3-ASR` |
| Training data | OpenSLR-104 Hindi-English train, chunked by `segments` file into 50,005 utterances |
| Validation data | OpenSLR-104 val (2,764 utts, speaker-disjoint random 5% by speaker_id) |
| In-domain test | OpenSLR-104 official test (3,132 utts) |
| Cross-domain test | HiACC test (1,036 utts) — same as v1 eval |
| Language prefix | `language None<asr_text>...` (language-agnostic, [Polyglot-Lion]) |
| Script | Mixed Devanagari + Latin, preserved as-is |
| Audio | All ≤30 s segments (60 longer dropped, 0 too-short, 0 load failures of 55,901 chunks) |
| Optimizer | AdamW, LR 2e-5, linear schedule, warmup_ratio 0.02 |
| Effective batch | 32 (per-device 8 × grad-accum 2 × 2 GPUs) |
| Precision | bf16 + FlashAttention 2 |
| Hardware | 2× NVIDIA H100 80GB (Modal) |
| Epochs | 3 (~4,690 steps, save every 200) — best at step 3000 (epoch 1.92) |
| Wall-clock | 4,351 s (~72 min) training; ~30 min eval (4 inference passes) |
| Compute cost | training ~$9.55, eval (incl. one restart at batch=32) ~$5 |

## Training trajectory

eval-loss plateaus at ~0.144 by step 3000 (epoch 1.92) and stays there for the remaining ~1,700 steps:

| Step | Epoch | eval_loss |
|---:|---:|---:|
|  200 | 0.13 | 0.2405 |
|  600 | 0.38 | 0.1750 |
| 1000 | 0.64 | 0.1615 |
| 1400 | 0.90 | 0.1544 |
| 1800 | 1.15 | 0.1496 |
| 2200 | 1.41 | 0.1465 |
| 2600 | 1.66 | 0.1455 |
| **3000** | **1.92** | **0.1436** ← best |
| 3400 | 2.18 | 0.1450 |
| 3800 | 2.43 | 0.1450 |
| 4200 | 2.69 | 0.1449 |
| 4600 | 2.94 | 0.1449 |

Curves: [`figures/v2_training_curves.png`](figures/v2_training_curves.png).

**Implication for v3**: 2 epochs is sufficient for OpenSLR's data scale. With v3's union (90 h + 5 h = ~95 h), we should expect convergence around epoch 2 as well — possibly earlier since HiACC adds new conversational variety the model hasn't memorized.

## Methodology notes

- **Pre-chunking.** OpenSLR-104 ships as 521 long-form WAVs (median ~12 min each) with a Kaldi `segments` file providing per-utterance timestamps. The training script expects per-utterance audio files, so we pre-chunked into `/data/openslr104/chunks/{train,test}/<utt_id>.wav` (~11 GB on the Modal volume).
- **Speaker-disjoint val split.** Re-split the train pool by `utt2spk` (520 unique speakers → 5% = 26 held out as val). Audit showed the `rec_id`-based split was already 0-leak in practice (each lecture is one speaker), but the explicit speaker partitioning is the more defensible approach.
- **Filter**: segments > 30 s dropped (60 train + 4 test = 64 / 55,961 utterances). Standard preprocessing convention from Polyglot-Lion.
- **Normalization at training**: whitespace-collapse only — but the input data was already lowercase + no-punct, so the effective normalizer was a no-op.
- **Normalization at eval**: Polyglot-Lion-style (lowercase + strip punctuation), applied symmetrically to hyps and refs. Used the same normalizer for v1 and v2 evaluations.

## Limitations

1. **Negative transfer is real and worth disclosing.** v2 is not a drop-in replacement for the base model in conversational settings — it's worse. Users should treat v2 as a specialist that pairs with a domain classifier or is used only when the input is known to be tutorial-style speech.
2. **OpenSLR-104 WERs aren't directly comparable to MUCS-2021 published numbers** without re-running their normalization (Kaldi tokenization conventions differ from Polyglot-Lion's eval normalizer).
3. **Single configuration.** No hyperparameter sweep, no seed averaging.
4. **No segment-end-of-audio bounds check** in chunking — if any segment timestamp overruns audio length we'd silently produce a short chunk. None observed in practice (forced-alignment quality was reliable).

## Cost summary

| Run | Compute |
|---|---|
| Download + inspect + chunk OpenSLR-104 | ~$0.20 |
| Smoke test (v2) | ~$0.50 |
| Full train (v2) | ~$9.55 |
| Eval (batch=16 attempt + batch=32 successful) | ~$5 |
| **v2 total** | **~$15** |
| (Cumulative v1 + v2) | **~$18.40** |

## Citations

Same as v1 — see [`v1_results.md`](v1_results.md). The OpenSLR-104 corpus citation is folded into the dataset notes:

```bibtex
@inproceedings{diwan2021mucs,
  title     = {{MUCS} 2021: Multilingual and Code-Switching {ASR} Challenges for Low Resource {Indian} Languages},
  author    = {Diwan, Anuj and Vaideeswaran, Rakesh and Shah, Sanket and others},
  booktitle = {Proc. Interspeech 2021},
  year      = {2021}
}
```
