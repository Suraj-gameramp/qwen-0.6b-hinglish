# V1 Results — Qwen3-ASR-0.6B fine-tuned on HiACC

## Headline

Fine-tuning the publicly released `Qwen/Qwen3-ASR-0.6B` on HiACC (5.24 h of Hinglish code-switched speech) reduces WER on the HiACC test split by **−10.30 percentage points absolute** (−42% relative) over the unmodified base model.

|                | n     | Baseline (Qwen3-ASR-0.6B, zero-shot) | Fine-tuned (this work) | Δ (pp) | Δ (relative) |
|----------------|------:|-------------------------------------:|------------------------:|-------:|-------------:|
| **Overall**    | 1,036 | **24.53%**                           | **14.23%**              | −10.30 | −42.0% |
| Adult          |   664 | 23.96%                               | 13.96%                  | −10.00 | −41.7% |
| Children       |   372 | 25.61%                               | 14.73%                  | −10.88 | −42.5% |

WER is computed with `jiwer` after Polyglot-Lion-style normalization applied symmetrically to predictions and references (lowercase + strip punctuation). Training itself preserves HiACC's labels verbatim (case, punctuation, mixed script).

## Configuration

| Parameter                | Value                                                              |
|--------------------------|--------------------------------------------------------------------|
| Base model               | `Qwen/Qwen3-ASR-0.6B`                                              |
| Fine-tune script         | `qwen3_asr_sft.py` @ commit `c17a131f` from `QwenLM/Qwen3-ASR`     |
| Training data            | HiACC (3,622 utts, ~3.7 h) — train split of all 5,176 utterances    |
| Validation data          | HiACC val (518 utts, ~0.5 h)                                       |
| Test data                | HiACC test (1,036 utts, ~1.1 h)                                    |
| Language prefix          | `language None<asr_text>...` (language-agnostic, [Toshniwal et al., 2018], [Polyglot-Lion §4.3]) |
| Script                   | Mixed Devanagari + Latin, preserved as-is                          |
| Hardware                 | 2× NVIDIA H100 80GB (Modal)                                        |
| Optimizer                | AdamW (HF Trainer default)                                         |
| Learning rate            | 2e-5, linear schedule, `warmup_ratio=0.02`                         |
| Effective batch          | 32 (per-device 8 × grad-accum 2 × 2 GPUs)                           |
| Precision                | bf16 + FlashAttention 2                                            |
| Epochs                   | 5 (~565 steps, save every 50 steps, 12 checkpoints kept)            |
| Best checkpoint by val   | step 350 (epoch 3.07), eval_loss = 0.1917                          |
| Wall-clock               | 618 s (~10 min)                                                    |
| Compute cost             | ~$1.36 (Modal H100×2)                                              |
| Seed                     | HF Trainer default                                                  |

## Approach (summary)

The technical detail and reproducibility checklist is in [`technical_v1_hiacc.md`](technical_v1_hiacc.md). The short version:

1. **No per-token language tags.** Following [Polyglot-Lion] (which adapts the design from [Toshniwal et al., 2018] and [Li et al., 2013]), every training example uses `language None<asr_text>...` as the target prefix. The model never learns to predict a language tag for code-switched audio, removing a failure mode where a misidentified language token cascades into spurious output. The `<asr_text>` literal is required structure in Qwen3-ASR's tokenizer — we don't strip it, we just set the language to `None`.
2. **Mixed-script labels kept untouched.** Qwen3-ASR's tokenizer round-trips Devanagari and Latin without splitting characters (verified empirically). Romanizing Hindi tokens would discard the model's existing Hindi prior and inflate WER through inconsistent romanization conventions.
3. **HiACC labels preserved verbatim in training.** Punctuation, case, contractions (`it's`, `don't`), and mixed scripts pass through with only whitespace collapse applied. The base model already emits punctuated cased text; stripping these would *un-teach* a useful capability. The WER-comparable normalizer (Whisper / Polyglot-Lion convention: lowercase + strip punctuation) is applied at **eval time only**, symmetrically to predictions and references.
4. **HiACC's own train/val/test splits are used as shipped.** The audio folder layout (`audio/{train,val,test}_split/`) is the source of truth (the adult cohort is missing the per-split TXT files but has the audio folders). Children's official splits share ~20 speakers with train; we accept this caveat and report in-domain WER rather than novel-speaker WER.
5. **Best checkpoint by val-loss.** All 12 checkpoints saved; the lowest-`eval_loss` checkpoint (step 350) is chosen post-hoc. After step 350, eval-loss plateaus and drifts up slightly — 5 epochs is the right stopping point at this data scale.

## Evaluation methodology

For each test utterance:

1. Call `Qwen3ASRModel.transcribe(audio=<path>, language=None)`.
2. Strip the leading `language ?<asr_text>` prefix from the model's output.
3. Apply the eval-time normalizer to both prediction and reference (identical pipeline on both sides).
4. Compute WER with `jiwer.wer(refs_normalized, hyps_normalized)`.

The eval-time normalizer:

```python
EVAL_PUNCT = re.compile(r"[\.,\?!\"'|\-\/“”…]")
def eval_normalize(text):
    t = text.strip().lower()
    t = EVAL_PUNCT.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
```

Cohort slicing uses `test.meta.jsonl`, written alongside `test.jsonl` during data prep, which carries the `cohort` field per utterance (`adult` vs `children`).

## Training trajectory

Eval loss bottoms out at step 350 (epoch 3.07) and plateaus thereafter:

| Step  | Epoch | eval_loss |
|------:|------:|----------:|
|   50  | 0.44  | 3.300     |
|  100  | 0.88  | 0.380     |
|  150  | 1.32  | 0.237     |
|  200  | 1.76  | 0.206     |
|  250  | 2.19  | 0.200     |
|  300  | 2.63  | 0.194     |
| **350** | **3.07** | **0.192** ← best |
|  400  | 3.51  | 0.197     |
|  450  | 3.95  | 0.196     |
|  500  | 4.39  | 0.199     |
|  550  | 4.83  | 0.199     |

Curves: [`figures/v1_training_curves.png`](figures/v1_training_curves.png).

## Limitations

1. **5 hours is small for code-switching ASR.** The −10 pp improvement is real but the absolute 14.23% WER has room — HiACC's diversity (≈30 distinct speakers per cohort, two domains: everyday questions and storytelling) bounds what this corpus alone can teach. Ablations on OpenSLR-104 (89.86 h Hindi-English tutorials) and the union are queued as v2/v3.
2. **HiACC train/val/test share speakers** (we measured ~20 speakers overlapping each pair of children splits). Reported WER is in-domain; novel-speaker performance will be worse.
3. **Tested on HiACC only.** No cross-domain evaluation on OpenSLR-104 yet. A model fine-tuned on conversational/everyday speech may underperform on technical-lecture speech.
4. **Baseline is only Qwen3-ASR-0.6B zero-shot.** Stronger baselines (Whisper-large-v3, Qwen3-ASR-1.7B zero-shot, MERaLiON, MUCS-2021 entries) not yet measured.
5. **Single seed, single configuration.** No hyperparameter ablations or seed averaging.

## Comparison points (for context, not direct comparison)

- **Polyglot-Lion-1.7B** on Mesolitica (Malay, 49h training): 21.51 WER. Their setup, 17× more training data, larger model.
- **MERaLiON-2-10B-ASR** on Mesolitica: 25.90 WER. 10B params, 120,000 h pretraining + Singapore-specific fine-tuning.
- **MUCS-2021** code-switching baselines (Hindi-English on OpenSLR-104): published WERs in the 25–35% range.

These aren't apples-to-apples — different corpora, model sizes, languages, data scales — but help anchor what "14% on a code-switched eval" means in 2026: a competitive number for a 0.6B model fine-tuned for 10 minutes on 5 hours of speech.

## Citations

```
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

@article{li2013spoken,
  title   = {Spoken language recognition: from fundamentals to practice},
  author  = {Li, Haizhou and Ma, Bin and Lee, Kong Aik},
  journal = {Proceedings of the IEEE},
  volume  = {101},
  number  = {5},
  pages   = {1136--1159},
  year    = {2013},
  doi     = {10.1109/JPROC.2012.2237151}
}

@inproceedings{radford2023whisper,
  title     = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author    = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg
               and McLeavey, Christine and Sutskever, Ilya},
  booktitle = {Proceedings of the 40th International Conference on Machine
               Learning},
  year      = {2023}
}
```
