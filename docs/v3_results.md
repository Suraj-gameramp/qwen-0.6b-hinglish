# V3 Results — Qwen3-ASR-0.6B fine-tuned on the union (HiACC + OpenSLR-104)

## Headline

Fine-tuning `Qwen/Qwen3-ASR-0.6B` on the **union** of HiACC (5.24 h conversational) and OpenSLR-104 (89.86 h tutorial) Hinglish speech yields the **first shippable model** in this series: it beats the base model on **both** domains (−8.88 pp on HiACC, −15.60 pp on OpenSLR) while giving up only ~1.6–2.2 pp versus each domain's dedicated specialist.

The clean takeaway: **v3 is the model to ship.** v1 and v2 are specialists — v1 only wins on conversational speech, v2 only wins on tutorials (and actively *hurts* conversational speech). v3 closes the gap on both, eliminating v2's negative transfer entirely.

Two findings stand out:

1. **v3 fixed v2's catastrophic negative transfer.** v2 was +12.91 pp *worse* than base on HiACC conversational speech; v3 is −8.88 pp *better* than base — a swing of −21.79 pp on HiACC versus v2.
2. **Balanced upsampling was not needed at this data scale.** HiACC is only **6.8%** of the union training data (3,622 of 53,627 utterances), yet v3 retained ~99% of v1's HiACC quality (15.85% vs 14.23%). The Qwen3-ASR base evidently has enough capacity to fit both distributions cleanly from even a small slice of each — Polyglot-Lion-style balanced upsampling appears unnecessary here.

## Full 3-variant 2×2 result matrix

|                                       | HiACC test (1,036 utts, conversational) | OpenSLR test (3,132 utts, tutorials) |
|---------------------------------------|----------------------------------------:|-------------------------------------:|
| **Base (Qwen3-ASR-0.6B, zero-shot)**  | 24.53%                                  | 50.66%                               |
| **v1** (HiACC specialist)             | **14.23%**                              | ~50% (untested, ≈ base)              |
| **v2** (OpenSLR specialist)           | 37.64%                                  | **32.83%**                           |
| **v3** (union, this work)             | **15.85%**                              | **35.06%**                           |

Reductions and gaps for v3:

| Comparison                       | HiACC                | OpenSLR              |
|----------------------------------|---------------------:|---------------------:|
| v3 Δ vs base                     | **−8.88 pp**         | **−15.60 pp**        |
| v3 Δ vs domain specialist        | +1.62 pp (vs v1)     | +2.23 pp (vs v2)     |
| v3 Δ vs v2 (the other specialist)| **−21.79 pp**        | —                    |

v3 beats base on both domains. The only regressions are the small generalist-vs-specialist gaps (+1.62 pp vs v1 on HiACC, +2.23 pp vs v2 on OpenSLR) — the expected price of a single model that covers both distributions. Critically, v3 turns v2's +12.91 pp HiACC *regression* into a −8.88 pp *improvement*.

## v3 cohort breakdown (HiACC)

| Cohort     | n     | v3 WER  |
|------------|------:|--------:|
| Adult      |   664 | 15.41%  |
| Children   |   372 | 16.66%  |
| **Overall**| 1,036 | **15.85%** |

The adult/child gap stays gentle (1.25 pp), mirroring v1's behavior — the union did not introduce a cohort bias.

## Why the union works where v2 alone failed

v2 (OpenSLR-only) overfit the tutorial distribution and degraded on spontaneous conversation. Adding back HiACC's 5.24 h of conversational speech — even at only 6.8% of the mix — was enough to keep the model anchored to the conversational distribution:

- **Distribution coverage restored.** The model sees both lecture-style and spontaneous everyday speech every epoch, so it no longer collapses toward the dominant tutorial acoustics/vocabulary.
- **Output schema balanced.** HiACC's cased, punctuated, mixed-script labels are mixed in with OpenSLR's lowercase, punctuation-free labels; the eval-time normalizer (lowercase + strip punctuation) is applied symmetrically, but the training mix keeps the model from drifting fully into one schema.
- **No upsampling required.** We deliberately did *not* balance the mix. Simple concatenation (deterministic shuffle, seed 42) sufficed; the base model's capacity absorbed both distributions.

## Configuration

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-ASR-0.6B` |
| Fine-tune script | `qwen3_asr_sft.py` @ commit `c17a131f` from `QwenLM/Qwen3-ASR` |
| Training data | UNION = 53,627 utterances = 3,622 HiACC (6.8%) + 50,005 OpenSLR-104 (93.2%), deterministically shuffled (seed 42) |
| Mixing strategy | Simple concatenation — **no balanced upsampling** |
| Validation data | 3,282 utterances (518 HiACC + 2,764 OpenSLR), mixed |
| HiACC test | HiACC test (1,036 utts, conversational) — same as v1/v2 eval |
| OpenSLR test | OpenSLR-104 official test (3,132 utts, tutorials) — same as v2 eval |
| Language prefix | `language None<asr_text>...` (language-agnostic, [Polyglot-Lion]) |
| Script | Mixed Devanagari + Latin, preserved as-is |
| Fine-tune scope | **Full-parameter** — no layers frozen, no LoRA |
| Optimizer | AdamW, LR 2e-5, linear schedule, warmup_ratio 0.02 |
| Gradient clipping | norm 1.0 (bf16 gradients) |
| Effective batch | 32 (per-device 8 × grad-accum 2 × 2 GPUs) |
| Precision | bf16 + FlashAttention 2 |
| Hardware | 2× NVIDIA H100 80GB (Modal, workspace `glitchcraft-inc`) |
| Epochs | 2 (3,352 steps, save every 200, 17 checkpoints kept) |
| Best checkpoint by eval_loss | step 3200 (epoch 1.91), eval_loss = 0.1500 |
| Wall-clock | 2,943 s (~49 min) training; ~30 min eval |
| Compute cost | training ~$6.50, eval ~$3.50 |
| Seed | 42 (data shuffle) |

### Full-parameter fine-tune

No layers were frozen and no LoRA adapters were used. Every weight is updated: the AuT audio encoder (~180M params), the projector, and the Qwen3-0.6B LLM. Optimization is AdamW over bf16 gradients clipped to norm 1.0.

## Training trajectory

eval_loss bottoms out at step 3200 (epoch 1.91) and stays flat — confirming 2 epochs is the right stopping point for the union, exactly as v2's trajectory predicted:

| Step | Epoch | eval_loss |
|---:|---:|---:|
|  200 | 0.12 | 0.2463 |
|  600 | 0.36 | 0.1796 |
| 1000 | 0.60 | 0.1641 |
| 1400 | 0.84 | 0.1565 |
| 1800 | 1.07 | 0.1531 |
| 2200 | 1.31 | 0.1515 |
| 2600 | 1.55 | 0.1509 |
| 3000 | 1.79 | 0.1502 |
| **3200** | **1.91** | **0.1500** ← best |

Curves: [`figures/v3_training_curves.png`](figures/v3_training_curves.png).

## Evaluation methodology

Identical to v1/v2 (see [`v1_results.md`](v1_results.md)). For each test utterance: call `Qwen3ASRModel.transcribe(audio=<path>, language=None)`, strip the leading `language ?<asr_text>` prefix, apply the eval-time normalizer (lowercase + strip punctuation) symmetrically to prediction and reference, then compute WER with `jiwer`. HiACC cohort slicing uses the `cohort` field in `test.meta.jsonl`. v3 is evaluated on both the HiACC test split and the OpenSLR-104 official test split.

## Limitations

1. **v3 is slightly worse than each specialist in-domain** (+1.62 pp vs v1 on HiACC, +2.23 pp vs v2 on OpenSLR). This is the expected generalist trade-off; for single-domain deployment a specialist is marginally better.
2. **OpenSLR 35.06% WER is still substantial.** Tutorial speech with dense code/path/version vocabulary (`bash`, `gnu/linux`, `version 1204`) remains hard for a 0.6B model.
3. **WERs are not directly comparable to MUCS-2021 published baselines** — different normalization conventions (Kaldi tokenization vs Polyglot-Lion's eval normalizer).
4. **Single seed, single configuration, no upsampling ablation.** We observe that balanced upsampling appears unnecessary at this data scale, but we did not run the controlled comparison (concat vs upsampled) to prove it.
5. **HiACC train/val/test share speakers.** Reported HiACC WER is in-domain, not novel-speaker.

## Cost summary

| Run | Compute |
|---|---|
| Build union dataset (shuffle + manifest) | folded into prior runs |
| Full train (v3) | ~$6.50 |
| Eval (HiACC + OpenSLR test) | ~$3.50 |
| **v3 total** | **~$10** |
| (Cumulative v1 + v2 + v3) | **~$28** |

For reference across the series:

| Run | Compute |
|---|---|
| v1 (HiACC specialist) | ~$3.40 |
| v2 (OpenSLR specialist) | ~$15 |
| v3 (union, this work) | ~$10 |
| **Cumulative** | **~$28** |

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

@inproceedings{diwan2021mucs,
  title     = {{MUCS} 2021: Multilingual and Code-Switching {ASR} Challenges for Low Resource {Indian} Languages},
  author    = {Diwan, Anuj and Vaideeswaran, Rakesh and Shah, Sanket and others},
  booktitle = {Proc. Interspeech 2021},
  year      = {2021}
}
```
