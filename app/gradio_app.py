from __future__ import annotations
import argparse, gradio as gr, torch, numpy as np, librosa, soundfile as sf
from transformers import WhisperProcessor, Wav2Vec2Processor
from src.speech_mtl.models.multitask_whisper import WhisperForASRAndIntensity
from src.speech_mtl.models.multitask_wav2vec2 import Wav2Vec2ForCTCAndIntensity
from src.speech_mtl.utils.audio import rms_dbfs, norm01_from_dbfs

def load_models(kind, ckpt):
    if kind == "whisper":
        proc = WhisperProcessor.from_pretrained(ckpt)
        mdl = WhisperForASRAndIntensity.from_pretrained(ckpt).eval()
    else:
        proc = Wav2Vec2Processor.from_pretrained(ckpt)
        mdl = Wav2Vec2ForCTCAndIntensity.from_pretrained(ckpt).eval()
    return proc, mdl

def transcribe_and_intensity(audio, model, checkpoint, max_tokens):
    if audio is None:
        return "", 0.0, 0.0
    sr, wav = audio
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    wav = np.clip(wav.astype(np.float32), -1.0, 1.0)

    proc, mdl = load_models(model, checkpoint)

    if model == "whisper":
        inputs = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            tokens = mdl.generate(inputs.input_features, max_new_tokens=max_tokens)
        text = proc.batch_decode(tokens, skip_special_tokens=True)[0]
    else:
        inputs = proc(wav, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = mdl(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = proc.batch_decode(pred_ids, skip_special_tokens=True)[0]

    db = rms_dbfs(wav)
    return text, float(db), float(norm01_from_dbfs(db))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='whisper', choices=['whisper','wav2vec2'])
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--max_new_tokens', type=int, default=128)
    args = ap.parse_args()

    with gr.Blocks() as demo:
        gr.Markdown("""# Multitask Speech: ASR + Intensity (dBFS)""")
        with gr.Row():
            model = gr.Dropdown(choices=["whisper","wav2vec2"], value=args.model, label="Model")
            checkpoint = gr.Textbox(value=args.checkpoint, label="Checkpoint path")
            max_tokens = gr.Slider(16, 256, step=1, value=args.max_new_tokens, label="Max new tokens (ASR)")
        audio = gr.Audio(sources=["microphone","upload"], type="numpy", label="Speak or upload audio")
        btn = gr.Button("Transcribe")
        text = gr.Textbox(label="Transcript")
        dbfs = gr.Number(label="Intensity (dBFS)")
        norm = gr.Number(label="Intensity (normalized [0,1])")

        btn.click(transcribe_and_intensity, [audio, model, checkpoint, max_tokens], [text, dbfs, norm])

    demo.launch()

if __name__ == "__main__":
    main()
