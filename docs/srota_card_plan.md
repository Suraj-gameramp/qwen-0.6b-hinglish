# Srota — Hugging Face Model Card BLUEPRINT (Planner output)

> This is the **structural plan** the executor agent will follow to write the final
> `README.md` for `Surajgameramp/srota`. It is NOT the final prose. Every section below
> specifies: what it contains, which figure/table/code goes there, exact facts to use,
> and tone. Do not invent numbers — every metric is pinned here. Brand the model
> **"Srota"** everywhere, but in the *first* mention and again in the Training Procedure /
> Citation sections, clearly state it is a **full-parameter fine-tune of Qwen3-ASR-0.6B**.

---

## 0. Global tone & style guide (read first)

- **Voice:** confident, precise, understated — the "big lab" register (Qwen/Llama/Mistral/DeepSeek).
  Lead with evidence, never with adjectives. No "revolutionary", "state-of-the-art",
  "blazing", "powerful". Claims must be backed by a number that appears in a table on this page.
- **Attribution is mandatory and honest.** Srota is a fine-tune, not a from-scratch model.
  Say so plainly and early. This is a feature (transparency), not a disclaimer to bury.
- **Numbers discipline:** WER always as `XX.XX%`; deltas as `−N.NN pp` (use the real minus sign
  `−`, not hyphen). Always state the normalization caveat near the first WER mention.
- **Person:** describe the model in third person ("Srota transcribes…"), use second person only
  in Quickstart / Intended Use ("you can…").
- **Length:** comprehensive but skimmable. Headers, bullets, one figure leading the results.
- **No emojis** anywhere except (optionally) a single check in the "Highlights" — prefer none.
- **Markdown hygiene:** real tables, fenced code blocks with language tags, `<div align="center">`
  for the banner + badge row so it renders centered on HF.

---

## 1. YAML frontmatter (EXACT spec — executor must reproduce this)

The card MUST open with this YAML. Note the **gating fields** — this is a GATED repo with
auto-approval. Required keys:

```yaml
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
extra_gated_heading: "Access Srota"
extra_gated_prompt: >-
  Srota is released for research and responsible use. Please tell us briefly how
  you intend to use the model. Access is granted automatically.
extra_gated_fields:
  Intended use: text
  Country: country
extra_gated_button_content: "Request access"
---
```

Notes for executor:
- Keep `base_model_relation: finetune` — reinforces honest attribution at the metadata level.
- `extra_gated_fields` must contain exactly **Intended use (text)** and **Country (country)**.
- `model-index` values are plain numbers (no `%`).
- Do not add a `co2_eq_emissions` block (we don't have a defensible figure).

---

## 2. Section-by-section blueprint (ordered)

### Section A — Banner (top of body, immediately after YAML)
- Centered banner image: `figures/srota_banner.png`.
- Use:
  ```html
  <div align="center">
    <img src="figures/srota_banner.png" alt="Srota — Hinglish ASR" width="100%"/>
  </div>
  ```
- Optionally a centered H1/tagline UNDER the banner if the banner has no text:
  `# Srota (श्रोत)` + one-line italic subtitle: *"A Hinglish (Hindi–English code-switched)
  speech recognition model."* Keep it to one line.

### Section B — Badge / link row (centered, directly under banner)
- A single centered row of shields.io-style badges. Recommended set (left→right):
  1. **Demo** → `https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo` (color: blue/yellow HF)
  2. **Base model: Qwen3-ASR-0.6B** → `https://huggingface.co/Qwen/Qwen3-ASR-0.6B`
  3. **Dataset: HiACC** → `https://zenodo.org/records/15551669`
  4. **Dataset: OpenSLR-104** → `https://openslr.org/104/`
  5. **License: Apache-2.0** → standard apache badge
- Format suggestion (executor may use markdown image-link badges):
  ```html
  <div align="center">

  [![Demo](https://img.shields.io/badge/🤗_Demo-Hinglish_ASR-blue)](https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo)
  [![Base model](https://img.shields.io/badge/Base-Qwen3--ASR--0.6B-6633cc)](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
  [![HiACC](https://img.shields.io/badge/Data-HiACC-green)](https://zenodo.org/records/15551669)
  [![OpenSLR-104](https://img.shields.io/badge/Data-OpenSLR--104-green)](https://openslr.org/104/)
  [![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)

  </div>
  ```
- Keep it to ONE row. No counts/downloads badges.

### Section C — "What is Srota?" intro (2–3 sentences, no header or a plain `##` is fine)
- **Wording direction (executor writes final prose, follow this content & order):**
  - Sentence 1: Srota is an automatic speech recognition (ASR) model for **Hinglish** —
    Hindi–English code-switched speech — that outputs natural **mixed Devanagari + Latin** script.
  - Sentence 2: It is a **full-parameter fine-tune of [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)**
    (~780M params: ~180M AuT audio encoder + projector + Qwen3-0.6B LLM), trained on the union of
    conversational and technical-tutorial Hinglish speech.
  - Sentence 3 (the headline claim): Srota **improves over the base model on both domains at once**
    — **−8.88 pp** WER on conversational speech and **−15.60 pp** on tutorial speech.
- One inline pointer: "Try it in the [live demo](…)."

### Section D — Highlights / Key Features (bullets)
Tight bullet list (5–6 bullets). Content:
- **Beats the base on both domains.** Conversational HiACC 24.53% → **15.85%**; tutorial
  OpenSLR-104 50.66% → **35.06%**.
- **One model, two domains.** Unlike a single-domain specialist, Srota does not trade one
  domain off against the other (it eliminates the negative transfer seen when training on
  tutorials alone — see Evaluation).
- **Native Hinglish output.** Emits Devanagari for Hindi words, Latin for English words —
  the way Hinglish is actually written (e.g. `मेरा favourite festival Diwali है`).
- **Compact.** ~780M parameters; runs on a single GPU in bf16.
- **Honest lineage.** Full-parameter fine-tune of Qwen3-ASR-0.6B — no frozen layers, no LoRA.
- **Open.** Apache-2.0; training corpora are CC BY 4.0.

### Section E — Results (LED BY THE FIGURE, then the table)
This is the centerpiece. Order matters:
1. **Figure first**, centered:
   ```html
   <div align="center">
     <img src="figures/srota_wer_comparison.png" alt="WER comparison: base vs v1 vs v2 vs Srota on HiACC and OpenSLR-104 test sets" width="80%"/>
   </div>
   ```
   One-line caption under it: grouped WER (%) on HiACC (conversational) and OpenSLR-104
   (tutorial) test sets — base model, the two single-domain fine-tunes (v1, v2), and Srota.
   *Lower is better.*
2. **Then the precise table.** Use this exact data (do not alter):

   | Model | HiACC test (conversational, 1,036 utts) | OpenSLR-104 test (tutorial, 3,132 utts) |
   |---|---:|---:|
   | Qwen3-ASR-0.6B (base, zero-shot) | 24.53% | 50.66% |
   | HiACC-only fine-tune (v1) | **14.23%** | ≈ base (untested) |
   | OpenSLR-only fine-tune (v2) | 37.64% (worse than base) | **32.83%** |
   | **Srota (union, this model)** | **15.85%** | **35.06%** |
   | **Srota Δ vs base** | **−8.88 pp** | **−15.60 pp** |

3. **Two short interpretive sentences** below the table:
   - Srota is the only fine-tune that beats the base on **both** test sets.
   - It gives up only ~1.6 pp vs the conversational specialist and ~2.2 pp vs the tutorial
     specialist — the expected, small generalist trade-off.
4. **Cohort sub-table** (HiACC adult/child) — small, right after:

   | HiACC cohort | n | Srota WER |
   |---|---:|---:|
   | Adult | 664 | 15.41% |
   | Children | 372 | 16.66% |
   | Overall | 1,036 | **15.85%** |

   One line: the adult/child gap stays gentle (1.25 pp) — the union introduced no cohort bias.
5. **Normalization footnote** (MUST appear here): WER computed with `jiwer` after a symmetric
   normalizer (lowercase + strip punctuation) applied to both predictions and references.
   These numbers are **not directly comparable** to MUCS-2021 published baselines, which use
   different (Kaldi-style) normalization.

### Section F — Quickstart
- 2–3 sentence lead: install + minimal transcribe call.
- Install line FIRST (call it out explicitly):
  ```bash
  pip install qwen-asr==0.0.6
  ```
- Then the code block (use EXACTLY this API surface):
  ```python
  import torch
  from qwen_asr import Qwen3ASRModel

  model = Qwen3ASRModel.from_pretrained(
      "Surajgameramp/srota",
      dtype=torch.bfloat16,
      device_map="cuda:0",
      attn_implementation="flash_attention_2",
  )

  results = model.transcribe(audio="path/to/your.wav", language=None)
  print(results[0].text)
  # e.g. "मेरा favourite festival Diwali है"
  ```
- Note bullets under the code:
  - `language=None` enables the language-agnostic decoding prefix Srota was trained with.
    Pass it explicitly.
  - Audio should be mono; ≤30 s segments per call (longer audio should be chunked).
  - bf16 + FlashAttention 2 recommended; `attn_implementation` can be dropped on CPU/older GPUs.
- One pointer line: "No setup? Use the [hosted demo](https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo)."

### Section G — Intended Use & Limitations of Use
- **Intended use** bullets:
  - Transcribing conversational Hinglish (casual Q&A, storytelling, image-prompted descriptions).
  - Transcribing technical-tutorial Hinglish (software walkthroughs, lecture-style instruction).
  - Producing natural mixed Devanagari+Latin Hinglish text.
- **Out of scope / not recommended:**
  - Monolingual pure-Hindi or pure-English production ASR where dedicated models are stronger.
  - Languages/dialects outside Hindi–English code-switching.
  - High-stakes uses (medical/legal transcription) without human review.
- (Brief — full failure modes go in Limitations & Biases below.)

### Section H — Training Data
- Lead sentence: Srota is trained on the **union** of two CC BY 4.0 Hinglish corpora,
  **simply concatenated with no upsampling**.
- Dataset bullets with links:
  - **[HiACC](https://zenodo.org/records/15551669)** (Singh, Singh & Kadyan, 2025; DOI
    10.5281/zenodo.15551669, CC BY 4.0) — **5.24 h** conversational Hinglish, 16 kHz mono WAV.
  - **[OpenSLR-104](https://openslr.org/104/)** (MUCS-2021 Multilingual & Code-Switching ASR
    challenge; CC BY 4.0) — **89.86 h** Hindi–English spoken-tutorial speech (IIT Bombay
    Spoken Tutorial project).
- Splits table:

  | Split | Utterances | Composition |
  |---|---:|---|
  | Train | 53,627 | HiACC 6.8% + OpenSLR-104 93.2% |
  | Val | 3,282 | 518 HiACC + 2,764 OpenSLR-104 |

- One sentence: each corpus's own official test set is used for evaluation, reported
  separately above.
- One sentence (the key finding, framed for data people): HiACC is only **6.8%** of the
  training mix, yet Srota retains ~99% of the conversational specialist's quality —
  **balanced upsampling was unnecessary at this scale** (a deterministic shuffle, seed 42,
  was enough).

### Section I — Training Procedure
- Lead: **Full-parameter fine-tune** of Qwen3-ASR-0.6B — no frozen layers, no LoRA. Every
  weight updated: AuT audio encoder (~180M), projector, and the Qwen3-0.6B LLM.
- Config table (use EXACTLY these values):

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

- **Data format** subsection (1–2 sentences): targets use the language-agnostic prefix
  `language None<asr_text>...` (following Polyglot-Lion / Toshniwal et al. 2018); transcripts
  kept in natural mixed Devanagari+Latin script.
- **Training curves** figure, centered, at the end of this section:
  ```html
  <div align="center">
    <img src="figures/training_curves.png" alt="Srota training curves: train/eval loss, gradient norm, learning rate" width="90%"/>
  </div>
  ```
  Caption: training/eval loss, gradient norm, and learning rate over 3,352 steps;
  eval_loss bottoms out at step 3200 (epoch 1.91) and stays flat, confirming 2 epochs.

### Section J — Evaluation (methodology + the negative-transfer story)
- **Methodology** (1 short paragraph): for each test utterance, call
  `transcribe(audio=…, language=None)`, strip the `language ?<asr_text>` prefix, apply the
  symmetric lowercase + strip-punctuation normalizer to hyp and ref, compute WER with `jiwer`.
  Evaluated on the HiACC test split (with adult/child cohort slicing) and the OpenSLR-104
  official test split.
- **The union vs. specialists finding** (this is the "why it works" narrative — pull from
  v2_results for the negative-transfer motivation):
  - A tutorial-only fine-tune (v2) gained −17.82 pp in-domain on OpenSLR-104 but **regressed
    +12.91 pp vs base on conversational HiACC** — classic negative transfer (lectures and
    conversations are far apart distributionally).
  - Adding back HiACC's 5.24 h of conversational speech — even at only 6.8% of the mix —
    re-anchors the model. Srota turns that +12.91 pp HiACC regression into a **−8.88 pp
    improvement** (a −21.79 pp swing vs v2 on HiACC) while keeping −15.60 pp on OpenSLR-104.
  - Conclusion sentence: Srota is the shippable generalist; the specialists are not drop-in
    replacements for the base across both domains.
- Do NOT repeat the full results table here (it's in Section E); reference it.

### Section K — Limitations & Biases
Bullet list, factual:
- **Generalist trade-off.** ~1.6 pp behind the conversational specialist on HiACC, ~2.2 pp
  behind the tutorial specialist on OpenSLR-104. For a single known domain a specialist is
  marginally better.
- **Tutorial WER is still substantial (35.06%).** Dense code/path/version vocabulary
  (`bash`, `gnu/linux`, `version 1204`) remains hard for a 0.6B model.
- **Not comparable to MUCS-2021 published numbers** without matching their Kaldi-style
  normalization.
- **Single seed, single configuration.** No hyperparameter sweep; the "upsampling unnecessary"
  claim is observed, not proven via a controlled concat-vs-upsampled ablation.
- **HiACC train/val/test share speakers.** Reported HiACC WER is in-domain, not novel-speaker —
  real-world conversational WER on unseen speakers may be higher.
- **Bias note.** Data is sourced from specific corpora (Indian spoken-tutorial + a defined
  conversational set incl. children); accent/dialect/domain coverage is limited and may not
  generalize to all Hinglish varieties.

### Section L — License
- Short: **Apache-2.0**, inherited from the base Qwen3-ASR-0.6B model. Training data: HiACC
  is **CC BY 4.0**; OpenSLR-104 is **CC BY 4.0** (see [openslr.org/104](https://openslr.org/104/)
  for full terms). Users must comply with the dataset licenses' attribution requirements.

### Section M — Citation
- Lead sentence: "If you use Srota, please cite this model and the underlying works."
- Provide a `bibtex` block. Include AT MINIMUM:
  - A **Srota** misc entry (executor creates it):
    ```bibtex
    @misc{srota2026,
      title  = {Srota: A Hinglish ASR model fine-tuned from Qwen3-ASR-0.6B},
      author = {Suraj},
      year   = {2026},
      url    = {https://huggingface.co/Surajgameramp/srota}
    }
    ```
  - `shi2026qwen3asr` (Qwen3-ASR Technical Report) — REQUIRED (base model).
  - `dang2026polyglot` (Polyglot-Lion) — recipe lineage.
  - `singh2025hiacc` (HiACC) — REQUIRED (data).
  - `diwan2021mucs` (MUCS-2021 / OpenSLR-104) — REQUIRED (data).
  - (Optional, keep if room: `toshniwal2018multilingual`.)
  - Copy the exact bibtex bodies from `docs/v3_results.md` "Citations" section.

### Section N — Acknowledgements
- 2–4 sentences:
  - The **Qwen team** for Qwen3-ASR-0.6B (the base) and the open `qwen3_asr_sft.py` training script.
  - **HiACC** authors (Singh, Singh & Kadyan) and the **MUCS-2021 / OpenSLR-104 / IIT Bombay
    Spoken Tutorial** contributors for the data.
  - **Polyglot-Lion** (Dang & Ngo) for the balanced-fine-tuning recipe and language-agnostic
    prefix that this work builds on.
- Close with one honest line: Srota stands entirely on Qwen3-ASR-0.6B; this work is the
  Hinglish adaptation, not a new foundation model.

---

## 3. Executor checklist (do not skip)

- [ ] YAML frontmatter exactly as Section 1, including `extra_gated_fields` (Intended use: text,
      Country: country), `extra_gated_heading`, `extra_gated_prompt`, `extra_gated_button_content`.
- [ ] Banner `figures/srota_banner.png` at top, centered.
- [ ] Single centered badge row (Demo, Base model, 2 datasets, License).
- [ ] Intro names Srota AND states "full-parameter fine-tune of Qwen3-ASR-0.6B" in sentence 2.
- [ ] Results section: figure FIRST (`figures/srota_wer_comparison.png`), then the table, then
      interpretation, then cohort table, then normalization footnote.
- [ ] Training curves figure (`figures/training_curves.png`) in Training Procedure.
- [ ] Quickstart: `pip install qwen-asr==0.0.6`, `from qwen_asr import Qwen3ASRModel`,
      repo id `Surajgameramp/srota`, `language=None`.
- [ ] All WER numbers match the pinned values; deltas use `−` and `pp`.
- [ ] Demo link present in intro AND quickstart.
- [ ] Citation block includes Srota + Qwen3-ASR + HiACC + MUCS-2021 (+ Polyglot-Lion).
- [ ] No unbacked hype; honest attribution preserved throughout.

## 4. Pinned facts quick-reference (single source of truth for executor)

- Repo: `Surajgameramp/srota` · Demo: https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo
- Base: `Qwen/Qwen3-ASR-0.6B`, ~780M params (AuT encoder ~180M + projector + Qwen3-0.6B LLM)
- Full-parameter FT, no LoRA, no frozen layers
- Train union: 53,627 utts (HiACC 6.8% / OpenSLR 93.2%); Val 3,282 (518 HiACC + 2,764 OpenSLR)
- HiACC 5.24 h conversational; OpenSLR-104 89.86 h tutorial
- Recipe: AdamW, LR 2e-5, linear, warmup 0.02, clip 1.0, eff batch 32 (8×2×2), bf16+FA2,
  2 epochs / 3,352 steps, best step 3200 (epoch 1.91, eval_loss 0.1500), 2×H100 80GB, ~49 min, seed 42
- Prefix: `language None<asr_text>...`; mixed Devanagari+Latin output
- WER (jiwer, lowercase+strip-punct, symmetric):
  - HiACC test (1,036): base 24.53 · v1 14.23 · v2 37.64 · **Srota 15.85** (adult 15.41 / child 16.66)
  - OpenSLR test (3,132): base 50.66 · v2 32.83 · **Srota 35.06**
  - Srota Δ vs base: **−8.88 pp** HiACC, **−15.60 pp** OpenSLR
  - v2 negative transfer: +12.91 pp vs base on HiACC; Srota vs v2 on HiACC = −21.79 pp
- Datasets: HiACC https://zenodo.org/records/15551669 (DOI 10.5281/zenodo.15551669, CC BY 4.0);
  OpenSLR-104 https://openslr.org/104/ (MUCS-2021, CC BY 4.0)
- License: Apache-2.0 (from Qwen3-ASR-0.6B)
- Install: `pip install qwen-asr==0.0.6`
