# Plan

## 1. Dataset on Modal

- Source: HiACC, https://zenodo.org/records/15551669, DOI 10.5281/zenodo.15551669, CC BY 4.0.
- Size: 5.24 h, 5,176 utterances (3,318 adult, 1,858 child), 531.6 MB zip.
- Modal flow: a Modal function downloads the zip from Zenodo into a Modal `Volume`, unzips there, and all subsequent jobs mount that volume read-only. Nothing lands on the laptop.
- First Modal job to run: inspect — list file tree, print 5 sample transcripts, print audio sample rate / channels / duration distribution, count tokens per utterance, detect script (Devanagari vs Latin). Output a short report to stdout.

## 2. Code-switching annotation: how Qwen3-ASR expects it

Qwen3-ASR SFT format (from the tech report) is a single sequence:

```
<|im_start|>assistant
language English<asr_text>the transcript<|im_end|>
```

There is exactly **one** `language X` tag at the start. The model was never trained to emit per-token or interleaved language tags inside `<asr_text>`. Hindi (`hi`) is one of the 30 supported languages.

So **do not use two language tags per token / per span**. The thesis in the objective is correct, and the reason is concrete: such tokens are out-of-distribution for Qwen3-ASR's decoder and would force the model to learn a new output grammar from 5 hours of data — a near-guaranteed regression.

**Chosen approach: B — drop the `language X` prefix entirely.** The model is trained to emit only `<asr_text>...<|im_end|>` and learns to handle Hindi-English mixing implicitly from acoustics.

### Lineage of the "no tag" choice

Polyglot-Lion did not invent this. From §2 of their paper ("Language identification in ASR"):

> *Language-agnostic approaches, in which the model infers the language implicitly from acoustic features, have been explored in the context of spoken language identification (Li et al., 2013) and multilingual ASR (Toshniwal et al., 2018), but remain less common in recent large-scale systems. Our work revisits this design choice...*

So the prior art is:
- **Toshniwal et al., 2018** — "Multilingual speech recognition with a single end-to-end model", ICASSP. Original argument for omitting language conditioning in multilingual ASR.
- **Li et al., 2013** — "Spoken language recognition: from fundamentals to practice", Proc. IEEE. Implicit language ID from acoustics.

Polyglot-Lion's contribution is empirically re-validating this older design for a moderate-scale LALM (Qwen3-ASR) on a code-switched setting (Singapore). We apply the same recipe to Hinglish.

## 3. Transcript script

Hinglish corpora vary: some use Devanagari for Hindi tokens + Latin for English, some romanize everything to Latin. This choice has to match whatever HiACC ships, because changing it (e.g., transliterating Devanagari → Latin) introduces label noise.

Decision deferred to the inspection job in §1. Once we see the actual transcripts:

- If HiACC is fully romanized → train as-is.
- If HiACC is mixed-script → train as-is; Qwen3 tokenizer handles both Devanagari and Latin natively.
- Do **not** normalize / transliterate as a preprocessing step before we've seen what HiACC actually contains.

## 4. Training recipe (multi-GPU H100, mirroring Polyglot-Lion)

- Base: `Qwen/Qwen3-ASR-0.6B`.
- Hardware: Modal, `gpu="H100:2"` (two H100s). For 5 h of data and a 0.6B model, more than 2 H100s adds DDP coordination cost without throughput gain — we'd be data-bound, not compute-bound. Bump to `H100:4` only if ablations stack up.
- Optimizer: AdamW, peak LR 2e-5, cosine annealing.
- Per-device batch 8, gradient accumulation 2 → effective batch = 8 × 2 grad-accum × 2 GPUs = 32 (same as Polyglot-Lion).
- Mixed precision: bf16 (H100-native).
- Lowercase, strip punctuation (Whisper / Qwen3-ASR convention).
- Discard utterances > 30 s.
- Split: 80/10/10 train/val/test, stratified on adult/child.
- **No language tag** during training (option B).
- Epochs: 5–20 with early stopping on val WER. One epoch ≈ minutes at this scale.

### Multi-GPU plan — abstract

We run distributed data-parallel (DDP) training on Modal. A single Modal function declares `gpu="H100:2"`; Modal provisions a 2×H100 container. Inside, `accelerate launch --num_processes 2` spawns one worker per GPU. Each worker holds a full replica of Qwen3-ASR-0.6B; a `DistributedSampler` shards the HiACC train split so every worker sees a disjoint subset; PyTorch's DDP AllReduces gradients after each backward pass; only rank 0 logs metrics, writes checkpoints, and pushes to the Hub. The HiACC volume is mounted read-only on all ranks; the checkpoint volume is mounted read-write but only rank 0 writes to it.

### Multi-GPU plan — pseudocode

```python
# modal_app.py
import modal

vol_data = modal.Volume.from_name("hiacc-data", create_if_missing=True)
vol_ckpt = modal.Volume.from_name("hinglish-ckpts", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate",
                 "datasets", "soundfile", "librosa",
                 "jiwer", "huggingface_hub")
)
app = modal.App("hinglish-finetune")

# Step 1 — download HiACC straight from Zenodo into a Modal Volume
@app.function(image=image, volumes={"/data": vol_data}, timeout=1800)
def download_hiacc():
    import urllib.request, zipfile, pathlib
    pathlib.Path("/data/hiacc").mkdir(exist_ok=True)
    urllib.request.urlretrieve(
        "https://zenodo.org/records/15551669/files/Corpus.zip",
        "/data/Corpus.zip",
    )
    with zipfile.ZipFile("/data/Corpus.zip") as z:
        z.extractall("/data/hiacc")
    vol_data.commit()

# Step 2 — multi-GPU DDP training
@app.function(
    image=image,
    gpu="H100:2",
    volumes={"/data": vol_data, "/ckpt": vol_ckpt},
    secrets=[modal.Secret.from_name("huggingface")],   # HF_TOKEN
    timeout=6 * 3600,
)
def train():
    import subprocess
    subprocess.run([
        "accelerate", "launch",
        "--num_processes", "2",
        "--mixed_precision", "bf16",
        "train.py",
        "--data_dir",        "/data/hiacc",
        "--output_dir",      "/ckpt/run-1",
        "--base_model",      "Qwen/Qwen3-ASR-0.6B",
        "--lr",              "2e-5",
        "--per_device_batch","8",
        "--grad_accum",      "2",
        "--epochs",          "10",
        "--no_language_tag",                 # option B
        "--push_to_hub",     "<hf-user>/qwen3-asr-hinglish",
    ], check=True)
    vol_ckpt.commit()

@app.local_entrypoint()
def main():
    download_hiacc.remote()
    train.remote()
```

`train.py` uses `accelerate.Accelerator()` to wrap model + optimizer + DataLoader; Accelerate handles DDP init, gradient sync, and device placement. Final checkpoint is pushed to the Hub from rank 0 via `model.push_to_hub(...)`.

## 5. Evaluation

- Metric: WER on the test split, lowercased, punctuation-stripped, computed with `jiwer`.
- Baselines on the same test split:
  - Qwen3-ASR-0.6B zero-shot.
  - Whisper-large-v3 zero-shot (sanity check).
- Slice WER by adult / child to surface child-speech regression.

## 6. Publishing

- Push final checkpoint + tokenizer + training config to Hugging Face Hub under `<hf-user>/qwen3-asr-hinglish`.
- `HF_TOKEN` lives in a Modal Secret named `huggingface` (created once via `modal secret create huggingface HF_TOKEN=...`).
- Push from rank 0 only; include a model card listing dataset (HiACC), license (CC BY 4.0), hyperparameters, and eval WER.

## 7. How to inspect HiACC yourself

I download it automatically — you don't touch your laptop. Once `download_hiacc.remote()` finishes, the volume is browsable in three ways:

1. **CLI:**
   ```
   modal volume ls hiacc-data /hiacc
   modal volume get hiacc-data /hiacc/<some-file>.wav ./
   ```
2. **Dashboard:** modal.com → Volumes → `hiacc-data` shows the tree.
3. **Inspect function:** I write a small `@app.function` that prints transcripts, audio durations, sample-rate distribution, and detected script (Devanagari vs Latin) — runs in ~30 s, output goes to your terminal.

## 8. Open questions for the user

1. **HF Hub repo name** — `<hf-user>/qwen3-asr-hinglish`? Public or private?
2. **Script convention for HiACC** — confirmable only after the inspection job; flagging here so it doesn't get lost.

## 9. Suggested additions to `instruction_for_claude.md`

- Hugging Face username (so the repo name in code isn't a placeholder).
- Whether ablations (option A vs B, single-GPU vs multi-GPU) should auto-run after the main run, or wait for your approval.
