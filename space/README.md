---
title: "Srota: Hinglish Speech Recognition"
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
models:
- Surajgameramp/qwen3-asr-0.6b-hinglish
---

# Srota: Hinglish Speech Recognition

A public demo of [`Surajgameramp/qwen3-asr-0.6b-hinglish`](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish),
a full-parameter fine-tune of Qwen3-ASR-0.6B for Hinglish (Hindi-English code-switched) speech.

Srota transcribes Hinglish the way people actually speak it, keeping English in Latin and Hindi in Devanagari instead of collapsing code-switched speech into all-Devanagari transliteration. Record or upload a clip and it returns a mixed-script transcript.

Runs on a free CPU Space: the 0.6B model transcribes short clips in a few seconds once warm.

Built by the team behind [susrota.com](https://www.susrota.com/) (a voice-dictation tool; currently English).
