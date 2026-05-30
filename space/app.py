import os
import torch
import gradio as gr
from qwen_asr import Qwen3ASRModel

MODEL_ID = os.environ.get("MODEL_ID", "Surajgameramp/qwen3-asr-0.6b-hinglish")

print(f"Loading {MODEL_ID} on CPU (float32)…", flush=True)
model = Qwen3ASRModel.from_pretrained(
    MODEL_ID,
    dtype=torch.float32,        # CPU-friendly; bf16 ops are slow/unsupported on CPU
    device_map="cpu",
    # no flash_attention_2 here: that's CUDA-only; default attention works on CPU
)
print("Model loaded.", flush=True)


def transcribe(audio_path):
    if not audio_path:
        return "⚠️ Please record or upload an audio clip first."
    try:
        results = model.transcribe(audio=audio_path, language=None)
        text = (results[0].text or "").strip()
        return text if text else "(no speech detected)"
    except Exception as e:
        return f"Error during transcription: {e}"


DESCRIPTION = """
# 🎙️ Srota: Hinglish Speech Recognition

**Srota transcribes Hinglish the way people actually speak it**, keeping English in Latin and Hindi in Devanagari (`मेरा favourite festival Diwali है`), instead of mangling code-switched speech into all-Devanagari transliteration. Record or upload a clip to try it.

Srota is a full-parameter fine-tune of [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), trained on the union of **HiACC** (conversational) and **OpenSLR-104** (technical tutorials). Full model card: [`Surajgameramp/qwen3-asr-0.6b-hinglish`](https://huggingface.co/Surajgameramp/qwen3-asr-0.6b-hinglish).

Built by the team behind [susrota.com](https://www.susrota.com/), a voice-dictation tool (currently English; Srota will power its Hinglish support).

**Word Error Rate (lower is better):**

| Model | HiACC test (conversational) | OpenSLR-104 test (tutorials) |
|---|---:|---:|
| Base Qwen3-ASR-0.6B (zero-shot) | 24.73% | 50.66% |
| **Srota** | **15.85%** | **35.06%** |

_Running on a free CPU Space: the first transcription after the Space wakes may take ~10-20 s._
"""

demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath",
                    label="Hinglish speech (record or upload)"),
    outputs=gr.Textbox(label="Transcription", lines=4, show_copy_button=True),
    title=None,
    description=DESCRIPTION,
    flagging_mode="never",
    examples=None,
)

if __name__ == "__main__":
    demo.launch()
