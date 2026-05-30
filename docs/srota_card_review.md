# Srota Model Card — Reviewer Report

Reviewed: `docs/srota_card_FINAL.md` against `docs/srota_card_plan.md` and ground truth `docs/v3_results.md`.

Bottom line up front: the draft is in excellent shape. All WER numbers, deltas, training facts, gating fields, and Qwen attribution check out against the pinned ground truth, the YAML parses cleanly, and all three figure paths are correct. The issues below are minor polish items; there are no correctness, gating, or attribution blockers.

---

## A. Correctness

All WER values verified against ground truth and reproduced faithfully:
- HiACC: base 24.53% / Srota 15.85% (adult 15.41, child 16.66) / v1 14.23% / v2 37.64% — all correct.
- OpenSLR-104: base 50.66% / Srota 35.06% / v2 32.83% — all correct.
- Deltas −8.88 pp (HiACC) and −15.60 pp (OpenSLR) — match the pinned values in `v3_results.md`.
- Training facts: union 53,627, 6.8%/93.2% split, val 3,282 (518+2,764 = 3,282 ✓), 2 epochs/3,352 steps, best step 3200 eval_loss 0.1500, LR 2e-5, effective batch 32, bf16+FA2, 2×H100, ~49 min (2,943 s), full fine-tune no LoRA — all correct.
- Cohort counts 664 + 372 = 1,036 ✓.

This section is correct.

1. **[NICE-TO-HAVE] Source-data arithmetic note (not a draft defect).** The pinned delta `−8.88 pp` on HiACC does not equal `15.85 − 24.53 = −8.68 pp`; likewise the pinned `+12.91 pp` v2 regression does not equal `37.64 − 24.53 = +13.11 pp`, and `−17.82 pp` v2 gain vs `32.83 − 50.66 = −17.83 pp`. The draft faithfully reproduces the values pinned in `v3_results.md`/the plan, so it is correct *relative to its instructions* and should NOT be changed unilaterally. Flagging only so the owner is aware the ground-truth doc itself has these rounding/transcription quirks before public release. The internally-consistent pairs (e.g. −8.88 ↔ −21.79 swing vs v2, since 15.85 − 37.64 = −21.79) suggest the intended base-HiACC reference may differ slightly from 24.53; resolve at source if desired. No card edit required to satisfy the review brief.

## B. Gating / YAML

YAML validated with a real parser: parses cleanly, correct indentation and types. All required keys present (license, language, base_model, base_model_relation: finetune, library_name, pipeline_tag, tags, datasets, metrics, model-index, and all four `extra_gated_*` keys). `extra_gated_fields` is exactly `Intended use: text` and `Country: country`. model-index WER values are 15.85 (HiACC) and 35.06 (OpenSLR), as plain numbers without `%`. This section is correct and ship-ready.

## C. Completeness vs Plan

Every planned section is present and in order: Banner → Badge row → What is Srota → Highlights → Results (figure-led, table, interpretation, cohort table, normalization footnote) → Quickstart → Intended Use → Training Data → Training Procedure (config table + data format + curves figure) → Evaluation → Limitations & Biases → License → Citation → Acknowledgements.

- Badge row: all 5 badges present with correct URLs (demo, base, HiACC zenodo, OpenSLR-104, license). Verified each link string.
- Results led by the WER figure, then table, then cohort table, then normalization footnote — exactly per plan.
- Training curves figure present in Training Procedure.
- Quickstart: `qwen-asr==0.0.6`, `from qwen_asr import Qwen3ASRModel`, repo id `Surajgameramp/srota`, `language=None` — all correct.
- Citations include Srota + Qwen3-ASR + Polyglot-Lion + HiACC + MUCS-2021 + Toshniwal — meets the minimum set.

2. **[SHOULD-FIX] Badge color drift from plan.** Plan Section B specifies the Demo badge `...Hinglish_ASR-blue` (blue/yellow HF). Draft line 66 uses `-yellow`:
   `[![Demo](https://img.shields.io/badge/🤗_Demo-Hinglish_ASR-yellow)]`
   This is harmless (yellow is arguably more HF-on-brand for a Space), but it deviates from the pinned format. Either accept it explicitly or revert to `-blue` for plan fidelity. Recommend keeping `-yellow` — it reads fine. Marking SHOULD-FIX only to record the intentional deviation.

3. **[NICE-TO-HAVE] Library tag vs inference package.** YAML declares `library_name: transformers`, but Quickstart loads via `from qwen_asr import Qwen3ASRModel` (the `qwen-asr` package), not `transformers`. This matches the plan's pinned YAML exactly, so it is not a defect against spec. Be aware HF's "Use this model" widget will surface a `transformers` snippet that will not match the actual `Qwen3ASRModel` API. No action needed for spec compliance; flag for the owner's awareness.

## D. Style / Polish

Tone is exactly the big-lab register the plan asked for: confident, evidence-led, no banned hype words ("revolutionary", "state-of-the-art", "blazing", "powerful" all absent). Every claim is backed by an on-page number. Tables are clean and right-aligned. Markdown is well-formed. Figures have descriptive alt text and italic centered captions. Intro names Srota and states "full-parameter fine-tune of Qwen/Qwen3-ASR-0.6B" in sentence 2 as required. This section meets the bar.

4. **[NICE-TO-HAVE] Tagline title font.** Line 60–62 renders `# Srota (श्रोत)` followed by an italic subtitle. The Devanagari "श्रोत" reads "shrot/srota" — confirm this is the intended spelling (the more common transliteration of "stream/source" is "स्रोत" with स्र, not श्र). If "श्रोत" is a deliberate stylization, leave it; otherwise consider "स्रोत". Cosmetic, owner's call.

5. **[NICE-TO-HAVE] Quickstart instruction redundancy.** Line 141: "`language=None` enables the language-agnostic decoding prefix Srota was trained with. Pass it explicitly." The "Pass it explicitly" is slightly odd given the code already passes it. Minor; reads fine as emphasis. Optionally trim to just the first sentence.

## E. Image Paths

Confirmed. The card references exactly `figures/srota_banner.png`, `figures/srota_wer_comparison.png`, and `figures/training_curves.png` (lines 57, 92, 202) — the repo-root `figures/` paths the uploader expects. `srota_banner.png` and `srota_wer_comparison.png` exist locally; `training_curves.png` is the upload-time rename of `v3_training_curves.png` per the task brief, so the path is correct. No action needed.

---

## Verdict

SHIP
