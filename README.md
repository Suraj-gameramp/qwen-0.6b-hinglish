# Srota — Hinglish speech recognition

Open-source ASR for Hindi-English code-switched speech, fine-tuned from [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) on ~95 hours of Hinglish.

Output stays in natural mixed script (English in Latin, Hindi in Devanagari) instead of collapsing into all-Devanagari transliteration like the base model does.

> _"मेरा favourite festival Diwali है"_ — what people actually write, what Srota outputs.

## Result

| Model | HiACC test (conversational) | OpenSLR-104 test (tutorial) |
|---|---:|---:|
| Qwen3-ASR-0.6B (base, zero-shot) | 24.73% WER | 50.66% WER |
| **Srota (this work)** | **15.85%** | **35.06%** |

About a 36% relative WER reduction on conversational Hinglish. Full numbers, cohort slices, and the negative-transfer ablation that motivated the union recipe are in [`docs/v3_results.md`](docs/v3_results.md).

## Models, demo, datasets

- **Recommended model**: [`Surajgameramp/qwen3-asr-0.6b-hinglish`](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish) (Srota, the union model)
- **Conversational specialist**: [`Surajgameramp/qwen3-asr-0.6b-hinglish-hiacc-v1`](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish-hiacc-v1)
- **Tutorial specialist**: [`Surajgameramp/qwen3-asr-0.6b-hinglish-openslr104-v2`](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish-openslr104-v2)
- **Live demo (free)**: <https://huggingface.co/spaces/Surajgameramp/hinglish-asr-demo>
- **Collection (all of the above)**: [Hinglish ASR family](https://huggingface.co/collections/Surajgameramp/hinglish-asr-qwen3-asr-06b-fine-tunes-6a1a79f2faffc4321317fb19)

### Training data

- [HiACC](https://zenodo.org/records/15551669) (Singh, Singh & Kadyan, 2025): 5.24 h of conversational Hinglish. CC BY 4.0.
- [OpenSLR-104](https://openslr.org/104/) (MUCS-2021 Hindi-English): 89.86 h of spoken-tutorial Hinglish. CC BY 4.0.

## Recipe (one paragraph)

Full-parameter fine-tune of all ~780M weights of Qwen3-ASR-0.6B (the ~600M Qwen3 LLM + ~180M AuT audio encoder + projector; no LoRA, no frozen layers) on the union of HiACC and OpenSLR-104, concatenated and shuffled with no upsampling. Language-agnostic decoding (every example uses the `language None<asr_text>...` target prefix, following [Polyglot-Lion](https://arxiv.org/abs/2603.16184) which adapts [Toshniwal et al., 2018](https://arxiv.org/abs/1711.01694)) so the model handles code-switching without needing a language tag. Transcripts kept in their natural mixed Devanagari + Latin form. Trained with AdamW at LR 2e-5, linear schedule with warmup_ratio 0.02, effective batch 32 (8 × grad-accum 2 × 2 GPUs), bf16 + FlashAttention 2, on 2× NVIDIA H100 80GB via Modal. 2 epochs (~3,352 steps), best checkpoint by val loss at step 3200. ~49 min wall-clock, ~$6.50 compute.

Step-by-step technical reproduction is in [`docs/technical_v1_hiacc.md`](docs/technical_v1_hiacc.md) (the original v1 walkthrough; v2 and v3 follow the same recipe with different data).

## Reproduce on Modal

```bash
# 1. download HiACC + OpenSLR-104 directly to Modal Volumes
modal run modal_app.py::download                    # HiACC (~3 min)
modal run modal_app.py::download_openslr            # OpenSLR-104 (~10 min)

# 2. prepare per-utterance JSONL (chunks long-form OpenSLR audio via segments file)
modal run modal_app.py::prepare_jsonl
modal run modal_app.py::prepare_openslr_jsonl
modal run modal_app.py::respit_openslr              # speaker-disjoint val resplit
modal run modal_app.py::prepare_union               # union JSONL for v3

# 3. fine-tune on 2x H100
modal run modal_app.py::smoke_v3                    # ~5 min sanity
modal run modal_app.py::train_v3                    # ~50 min, ~$6.50

# 4. evaluate on both test sets
modal run modal_app.py::evaluate_v3 --batch-size 32

# 5. publish to HF Hub
modal run modal_app.py::push_v3
```

All the model card content (Srota, v1, v2) is in [`docs/`](docs/). Figures rendered on the HF pages live in the public [`Surajgameramp/srota-assets`](https://huggingface.co/datasets/Surajgameramp/srota-assets) dataset.

## Try the model locally

A push-to-talk CLI that streams audio from your microphone to a warm Modal container running Srota:

```bash
python3 scripts/realtime_transcribe.py --model v3   # uses Srota (recommended)
python3 scripts/realtime_transcribe.py --model base # comparison vs zero-shot Qwen3-ASR-0.6B
```

## Citations

```bibtex
@article{shi2026qwen3asr,
  title  = {Qwen3-ASR Technical Report},
  author = {Shi, Xian and Wang, Xiong and others},
  year   = {2026},
  url    = {https://arxiv.org/abs/2601.21337}
}

@article{dang2026polyglot,
  title  = {Polyglot-Lion: Efficient Multilingual ASR for Singapore via Balanced Fine-Tuning of Qwen3-ASR},
  author = {Dang, Quy-Anh and Ngo, Chris},
  year   = {2026},
  url    = {https://arxiv.org/abs/2603.16184}
}

@misc{singh2025hiacc,
  title  = {HiACC: Hinglish Adult \& Children Code-switched Corpus},
  author = {Singh, Shruti and Singh, Muskaan and Kadyan, Virender},
  year   = {2025},
  doi    = {10.5281/zenodo.15551669}
}

@inproceedings{diwan2021mucs,
  title     = {{MUCS} 2021: Multilingual and Code-Switching {ASR} Challenges for Low Resource {Indian} Languages},
  author    = {Diwan, Anuj and Vaideeswaran, Rakesh and Shah, Sanket and others},
  booktitle = {Proc. Interspeech 2021},
  year      = {2021}
}
```

## License

Apache-2.0, inherited from the base Qwen3-ASR-0.6B. Training data (HiACC, OpenSLR-104) is CC BY 4.0; preserve attribution.

## Contact

Suraj Prasad: surajprasad8977@gmail.com

Built by the team behind [susrota.com](https://www.susrota.com/), a voice-dictation tool. Currently English; Srota will power its upcoming Hinglish support.
