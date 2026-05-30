#!/usr/bin/env python3
"""Push-to-talk Hinglish ASR test against a deployed Modal class.

Usage:
    # Deploy once (from project root):
    modal deploy modal_app.py

    # Then run this script (press any key to start/stop recording):
    python3 scripts/realtime_transcribe.py --model v1
    python3 scripts/realtime_transcribe.py --model v2
    python3 scripts/realtime_transcribe.py --model base

`--model` chooses which weights the Modal container loads:
    v1   = your HF repo `Surajgameramp/qwen3-asr-0.6b-hinglish-hiacc-v1`
    v2   = `/ckpt/v2-openslr-h100x2/checkpoint-3000` on the Modal volume
    base = `Qwen/Qwen3-ASR-0.6B` (zero-shot, no fine-tune)

Notes:
- The terminal raw-mode key capture catches *any printable key in this terminal
  window* (space / enter / letters). True global <fn> key support would require
  pynput + Accessibility permissions; left out for simplicity.
- First call after a cold container takes ~30 s (model load). Subsequent calls
  are ~0.5–1 s end-to-end. Container stays warm for 3 min of idle.
"""

from __future__ import annotations

import argparse
import io
import sys
import termios
import threading
import time
import tty
import wave

import numpy as np
import sounddevice as sd

import modal


SR = 16_000

MODEL_CHOICES = {
    "v1":   "Surajgameramp/qwen3-asr-0.6b-hinglish-hiacc-v1",
    "v2":   "/ckpt/v2-openslr-h100x2/checkpoint-3000",
    "base": "Qwen/Qwen3-ASR-0.6B",
}


def getch() -> str:
    """Block until any single key is pressed in the terminal; return it."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def record_until_keypress(samplerate: int = SR) -> tuple[np.ndarray, float]:
    """Stream mic audio into memory; return (audio_f32, duration_s)."""
    frames: list[np.ndarray] = []
    stop_event = threading.Event()

    def on_audio(indata, frames_count, time_info, status):
        if status:
            print(f"\n[audio status] {status}", flush=True)
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        callback=on_audio,
    )
    stream.start()
    t0 = time.time()

    def wait_key():
        getch()
        stop_event.set()

    threading.Thread(target=wait_key, daemon=True).start()

    print("🎙  recording… press any key to stop", flush=True)
    while not stop_event.is_set():
        time.sleep(0.05)

    stream.stop()
    stream.close()

    audio = (np.concatenate(frames, axis=0).squeeze()
             if frames else np.zeros(0, dtype=np.float32))
    return audio.astype(np.float32), time.time() - t0


def to_wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    """Encode mono float32 audio in [-1,1] as a WAV byte-string."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)        # 16-bit PCM
        wf.setframerate(sr)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODEL_CHOICES.keys(), default="v1",
                    help="Which weights the Modal container loads (default: v1)")
    ap.add_argument("--language", default=None,
                    help="Force a language tag (e.g. English, Hindi). "
                         "Omit to let the model decide (recommended for code-switching).")
    ap.add_argument("--app", default="hinglish-finetune",
                    help="Modal app name (default: hinglish-finetune)")
    ap.add_argument("--skip-warmup", action="store_true",
                    help="Skip the dummy warmup call.")
    args = ap.parse_args()

    model_id = MODEL_CHOICES[args.model]
    print(f"[modal] connecting to {args.app}::RealtimeASR  model_id={model_id}")
    try:
        RealtimeASR = modal.Cls.from_name(args.app, "RealtimeASR")
    except AttributeError:
        RealtimeASR = modal.Cls.lookup(args.app, "RealtimeASR")  # legacy SDK
    asr = RealtimeASR(model_id=model_id)

    if not args.skip_warmup:
        print("[modal] warming container (first call ~30 s)…", end=" ", flush=True)
        t0 = time.time()
        _ = asr.transcribe.remote(to_wav_bytes(np.zeros(SR, dtype=np.float32)))
        print(f"ready ({time.time()-t0:.1f} s)")

    while True:
        print(f"\n▶  press any key to record  [model={args.model}, q to quit]: ",
              end="", flush=True)
        ch = getch()
        if ch.lower() == "q":
            print("\nbye 👋")
            return
        audio, dur = record_until_keypress()
        if len(audio) < SR // 4:
            print(f"\n   (only {dur:.2f}s captured — ignored)")
            continue
        print(f"\n   captured {dur:.1f}s of audio, sending to Modal…", flush=True)

        wav_bytes = to_wav_bytes(audio)
        t0 = time.time()
        try:
            result = asr.transcribe.remote(wav_bytes, language=args.language)
        except Exception as e:
            print(f"   ✗ inference failed: {e}")
            continue
        rtt = time.time() - t0
        print(f"📝  [{result['language'] or '?':<7}]  "
              f"model={result['ms']}ms  rtt={rtt:.2f}s  "
              f"speech_dur={result['dur_s']:.2f}s")
        print(f"    → {result['text']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nbye 👋")
