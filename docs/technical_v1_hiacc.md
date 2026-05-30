# V1 Technical Doc — HiACC Fine-tuning of Qwen3-ASR-0.6B

This is the reproducible plan for **v1**: fine-tune Qwen3-ASR-0.6B on the HiACC corpus only. OpenSLR-104-only and the union run are queued as ablations after v1 produces a clean WER number.

This doc is updated as we run. Every command listed here is the one that was actually executed.

---

## 1. Pinned versions

| Component | Version / Commit |
|---|---|
| Base model | `Qwen/Qwen3-ASR-0.6B` (HF Hub) |
| Qwen3-ASR repo (local mirror) | `third_party/Qwen3-ASR/` @ commit `c17a131fe028b2e428b6e80a33d30bb4fa57b8df` |
| `qwen-asr` PyPI package | `0.0.6` |
| `transformers` | `4.57.6` (pinned by qwen-asr) |
| `accelerate` | `1.12.0` (pinned by qwen-asr) |
| HiACC dataset | Zenodo DOI `10.5281/zenodo.15551669`, file `Corpus.zip` (531.6 MB) |
| Modal workspace | `suraj-77958` |
| Local conda env | `finetune` |

Random seed: `42` (set everywhere — `torch`, `numpy`, `random`, `DataLoader`).

---

## 2. Decisions locked in

| Decision | Value | Rationale |
|---|---|---|
| Language prefix in target | `language None<asr_text>...` | Polyglot-Lion language-agnostic strategy; matches Qwen3-ASR's official "no language info" path. |
| Script | Mixed (Devanagari + Latin) — train as-is | Tokenizer round-trips both natively (verified); HiACC ships mixed-script; romanization would inflate WER and discard model priors. |
| Punctuation | **Preserved as-is** | HiACC labels are deliberately punctuated; base Qwen3-ASR already emits punctuation. Stripping would un-teach a capability and split contractions like `it's` → `it s`. The Polyglot-Lion "strip punctuation" recipe exists to normalize across 12 inconsistent corpora — irrelevant for our single dataset. |
| Casing | **Preserved as-is** | Same reasoning. Base model emits cased text; HiACC is cased; lowercasing un-teaches. Devanagari has no case (no-op there). |
| Normalization scope | Collapse internal whitespace; trim ends | Only correction applied at training time. Eval-time WER normalization is a separate step (§6.2). |
| Splits | HiACC's own (audio folder layout) | Reproducible against any other HiACC user. |
| Cohort handling | Adult + children mixed in one pool, no balancing | v1 keeps it simple. Slice WER by cohort at eval. |
| Speaker leakage | Accepted as-is | HiACC's own splits leak ~20 speakers across train/val/test. Eval flagged as in-domain WER, not novel-speaker WER. |
| GPU | Modal `H100:2` | DDP via `torchrun --nproc_per_node=2`. |
| Precision | bf16 | H100 native. |
| Batch | `per_device=8, grad_acc=2` → effective 32 | Matches Polyglot-Lion. Default `32×4×2=256` would give only ~14 steps/epoch on our data. |
| Optimizer | AdamW (HF Trainer default) | Built into `qwen3_asr_sft.py`. |
| LR | `2e-5` | Official default, also Polyglot-Lion's value. |
| Scheduler | `linear` with `warmup_ratio=0.02` | Official default. |
| Epochs | 5 with eval every 50 steps, save every 50, keep all (~11) checkpoints | ~113 steps/epoch × 5 = ~565 steps. Pick best checkpoint by `eval_loss` post-hoc; resume from last checkpoint if eval_loss still decreasing at epoch 5. |

---

## 3. File layout (on disk + Modal)

```
hinglish_finetuning/                 (this repo, local)
├── modal_app.py                     Modal app: volumes, image, download/inspect/prepare/train fns
├── docs/
│   ├── objectives.md
│   ├── plan.md
│   └── technical_v1_hiacc.md        (this file)
├── third_party/
│   └── Qwen3-ASR/                   git clone, commit c17a131f
│       └── finetuning/
│           ├── README.md
│           └── qwen3_asr_sft.py     (the official trainer we invoke)
├── dataset_samples/                 (local copies for spot-checking)
└── test.py                          (Modal hello-world)

Modal volumes:
  hiacc-data/                        (read-only at train time, ~600 MB)
    Corpus.zip
    hiacc/Corpus/
      readme.txt
      adult/
        audio/{train,val,test}_split/*.wav
        annotations/code_switched_labels.json
        metadata/{sentence_stats,speaker_info}.csv
        transcription/                (note: missing the per-split TXT files)
      children/
        audio/{train,val,test}_split/*.wav
        annotations/code_switched_labels.json
        metadata/{sentence_stats,speaker_info}.csv
        transcript/{train,val,test}_output.txt
    jsonl/                           (produced by prepare_hiacc_jsonl)
      train.jsonl
      val.jsonl
      test.jsonl

  hinglish-ckpts/                    (read-write only by rank 0)
    v1-hiacc-h100x2/
      checkpoint-50/
      checkpoint-100/
      ...
      train.log
      run.json                       (config + final metrics)

  openslr104-data/                   (download in progress; used for ablation v2/v3)
```

---

## 4. Data preparation

### 4.1 Source files in HiACC

Per cohort, the source of truth is **`annotations/code_switched_labels.json`** (5,176 entries total: 3,318 adult + 1,858 child). Each entry: `{audio | audio_filepath, transcription, label}`.

Splits come from the **audio folder layout** (`audio/{train,val,test}_split/*.wav`), not the transcript TXTs (adult's TXTs are missing).

### 4.2 Normalization

HiACC labels are preserved as-is. The only correction is whitespace collapse — necessary because runs of multiple spaces inside transcripts (e.g., `"reason  है"`) would tokenize differently than single spaces.

```python
import re
WS_RE = re.compile(r"\s+")

def normalize(text: str) -> str:
    return WS_RE.sub(" ", text).strip()
```

Not stripped, not lowercased: punctuation (`. , ? ! ' " -`), case, and mixed scripts all pass through unchanged. Apostrophes inside contractions (`it's`, `don't`) stay intact.

For **eval-time WER**, both the prediction and reference go through a separate, more aggressive normalizer that lowercases + strips punctuation (Polyglot-Lion style) — this is for **comparability with published WER numbers**, not for training (§6.2).

### 4.3 Basename resolution

JSON entry's path → strip → look up in the cohort's `audio/{split}_split/` folder.

```python
basename = pathlib.Path(entry["audio"] or entry["audio_filepath"]).name.strip()
audio_path = root / cohort / "audio" / f"{split_name}_split" / basename
```

The `.strip()` is **required** to fix the 9 children entries with trailing whitespace in their basename.

### 4.4 JSONL row format

```json
{"audio": "/data/hiacc/Corpus/adult/audio/train_split/AD09002.wav",
 "text":  "language None<asr_text>So my favourite festival is Diwali. The reason being इसका जो reason है कि that it is a festival of light."}
```

- `audio`: absolute path inside the Modal container (mount point `/data`).
- `text`: the target sequence the model must produce. The `<asr_text>` literal is required by Qwen3-ASR's tokenizer; `language None` tells the model not to predict a language tag.

### 4.5 Expected output

```
hiacc-data:/hiacc/jsonl/train.jsonl   3,622 lines
hiacc-data:/hiacc/jsonl/val.jsonl       518 lines
hiacc-data:/hiacc/jsonl/test.jsonl    1,036 lines
```

### 4.6 Command

```bash
modal run modal_app.py::prepare_jsonl
```

Implemented as `prepare_hiacc_jsonl` in `modal_app.py`. Runs on CPU, ~30 s, ~$0. Prints first 3 lines of each output file to stdout for eyeball verification.

---

## 5. Training

### 5.1 Modal image

```python
train_image = (
    modal.Image.debian_slim()
    .apt_install("git", "ffmpeg", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "qwen-asr==0.0.6",
        "datasets",
        "librosa",
        "soundfile",
        "jiwer",
        "huggingface_hub",
    )
    .pip_install("flash-attn==2.7.4.post1", extra_options="--no-build-isolation")
    .add_local_dir("third_party/Qwen3-ASR/finetuning", "/opt/qwen3-asr-finetuning")
)
```

`flash-attn` is built from source; one-time image build is the slow part (~10 min). Subsequent runs reuse the image.

### 5.2 Modal function

```python
@app.function(
    image=train_image,
    gpu="H100:2",
    volumes={"/data": vol_data, "/ckpt": vol_ckpt},
    timeout=6 * 3600,
)
def train_v1_hiacc():
    import subprocess, json, pathlib, time

    output_dir = pathlib.Path("/ckpt/v1-hiacc-h100x2")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "torchrun", "--nproc_per_node=2",
        "/opt/qwen3-asr-finetuning/qwen3_asr_sft.py",
        "--model_path",  "Qwen/Qwen3-ASR-0.6B",
        "--train_file",  "/data/hiacc/jsonl/train.jsonl",
        "--eval_file",   "/data/hiacc/jsonl/val.jsonl",
        "--output_dir",  str(output_dir),
        "--batch_size",  "8",
        "--grad_acc",    "2",
        "--lr",          "2e-5",
        "--epochs",      "5",
        "--log_steps",   "10",
        "--save_strategy", "steps",
        "--save_steps",  "50",
        "--save_total_limit", "15",
        "--lr_scheduler_type", "linear",
        "--warmup_ratio", "0.02",
        "--num_workers", "4",
        "--pin_memory",  "1",
        "--persistent_workers", "1",
        "--prefetch_factor", "2",
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    vol_ckpt.commit()
    (output_dir / "run.json").write_text(json.dumps({
        "config": " ".join(cmd),
        "wall_clock_seconds": time.time() - t0,
    }))
```

### 5.3 Expected wall-clock (revised after smoke)

- Image build: ~10 min (one-time, cached after first run).
- Steps per epoch: 3,622 / 32 = ~113.
- 5 epochs: ~565 steps.
- Smoke measured ~19 s/step pure training (audio loading dominates I/O).
- Likely steady-state: ~10 s/step after warmup. Full run ≈ 1.5–3 h.
- Cost estimate: 2 × H100 at $3.95/h × ~3 h ≈ **$12–24** end-to-end.

### 5.4 Smoke test before full run

Same command with `--epochs 0.05` (about 6 steps) and `--save_steps 5`. Verifies: image builds, data loads, model forwards, gradients flow, checkpoint writes to volume, no DDP rank deadlock. Expected: ~5 min, ~$0.50.

---

## 6. Evaluation

### 6.1 What we measure

- **WER** on HiACC test split (1,036 utterances), with our normalizer applied to both prediction and reference.
- Sliced by cohort: adult test (664) vs child test (372).
- Baselines on the same test split:
  - Qwen3-ASR-0.6B **zero-shot** (no fine-tune).
  - Whisper-large-v3 zero-shot (optional sanity check).

### 6.2 Eval pipeline

```python
from qwen_asr import Qwen3ASRModel
import torch, jiwer, json

model = Qwen3ASRModel.from_pretrained(
    "/ckpt/v1-hiacc-h100x2/checkpoint-<best>",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

refs, hyps = [], []
with open("/data/hiacc/jsonl/test.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        ref = ex["text"].split("<asr_text>")[-1]            # strip the language prefix from ref
        result = model.transcribe(audio=ex["audio"], language=None)[0]
        hyps.append(normalize(result.text))
        refs.append(normalize(ref))

wer = jiwer.wer(refs, hyps)
```

**Eval-time normalizer** (applied to both prediction and reference, identical on both sides — and distinct from the training-time `normalize()` in §4.2):

```python
import re
EVAL_PUNCT_RE = re.compile(r"[\.,\?!\"'|\-\/“”]")
def eval_normalize(text: str) -> str:
    text = text.strip().lower()
    text = EVAL_PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

Training preserves HiACC labels (punct + case). Eval normalizes both sides for WER comparability with Polyglot-Lion-style published numbers.

### 6.3 Reporting

`/ckpt/v1-hiacc-h100x2/eval.json`:
```json
{
  "checkpoint": "checkpoint-XXX",
  "wer_overall": 0.??,
  "wer_adult":   0.??,
  "wer_child":   0.??,
  "baseline_zero_shot_wer_overall": 0.??,
  "n_test": 1036
}
```

---

## 7. Reproducibility checklist

To re-run v1 from a fresh laptop:

1. `conda activate finetune` (assumes the env already exists).
2. `python -m modal setup` (or `modal token new`) → log in to workspace `suraj-77958`.
3. `git clone <this-repo>` and `cd hinglish_finetuning`.
4. `git clone --depth 1 https://github.com/QwenLM/Qwen3-ASR.git third_party/Qwen3-ASR` then `cd third_party/Qwen3-ASR && git checkout c17a131fe028b2e428b6e80a33d30bb4fa57b8df`.
5. `modal run modal_app.py::download` (HiACC; idempotent; skips if already populated).
6. `modal run modal_app.py::inspect_deep` (sanity; should match §4.1 counts).
7. `modal run modal_app.py::prepare_jsonl` (produces the three JSONLs on the volume).
8. `modal run modal_app.py::train_v1_hiacc` (full run).
9. `modal run modal_app.py::eval_v1_hiacc` (WER + slices).

Outputs:
- Checkpoint at `hinglish-ckpts:/v1-hiacc-h100x2/checkpoint-<best>/`
- Metrics at `hinglish-ckpts:/v1-hiacc-h100x2/eval.json`

---

## 8. Known risks

1. **5h of training data is on the low end for code-switching.** v1 result is a baseline; if WER is poor we have OpenSLR-104 ready as the next experiment.
2. **HiACC test split shares speakers with train.** Reported WER is in-domain; treat it as the upper bound of what's achievable, not a generalization claim.
3. **`flash-attn` build can be flaky** on first image build. If it fails, fall back to dropping the flash-attn install and letting Transformers use its default SDPA implementation.
4. **`qwen-asr` PyPI version drift.** Pinned to `0.0.6` here; if it advances and changes the JSONL contract we re-pin and re-prepare.

---

## 9. Iteration log

(Filled in as we go.)

- `2026-05-26`: doc created; HiACC downloaded + inspected; preprocessing recipe locked in.
- `2026-05-26`: switched normalizer to whitespace-only (preserve HiACC labels as-is); JSONLs written (3622/518/1036 lines).
- `2026-05-26`: Modal image built (flash-attn + qwen-asr 0.0.6 + Qwen3-ASR repo @ c17a131); smoke test passed end-to-end (exit 0, 6 steps, eval ran, both checkpoints saved). Pure step time ~19 s, slower than projected.
- `2026-05-26`: hyperparams audited; epochs lowered 10→5, save_total_limit raised 5→15 to keep all checkpoints. Kicking off full v1 run.
- `2026-05-26`: v1 training complete (exit 0). 565 steps, 618s wall, ~$1.36. 12 checkpoints saved. Best by `eval_loss`: step 350 (epoch 3.07, eval_loss=0.1917). Loss-curve plot at `figures/v1_training_curves.png`.
- `2026-05-26`: v1 eval complete. Fine-tuned WER 14.23% vs zero-shot baseline 24.53% on HiACC test (1,036 utts) — **−10.30 pp absolute, −42% relative**. Adult and child cohorts improved evenly (−10.0 / −10.9 pp).
- `2026-05-26`: discovered upstream bug in `qwen3_asr_sft.py` — `MakeEveryCheckpointInferableCallback` silently skips file copy when `model_path` is an HF Hub ID (uses `os.path.exists()` which returns False). Workaround in eval pipeline: pull preprocessor/chat_template files from `snapshot_download(base_id)` and copy into checkpoint before loading. Same repair applied in `push_v1_to_hub`.
- `2026-05-26`: results doc + HF model card drafted; Modal secret `huggingface` created from `.env`; pushing checkpoint-350 to `Surajgameramp/qwen3-asr-0.6b-hinglish-hiacc-v1` (public).
