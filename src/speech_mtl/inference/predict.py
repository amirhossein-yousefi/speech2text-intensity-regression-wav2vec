from __future__ import annotations
import argparse, torch, soundfile as sf, numpy as np
from transformers import WhisperProcessor, Wav2Vec2Processor
from ..models.multitask_whisper import WhisperForASRAndIntensity
from ..models.multitask_wav2vec2 import Wav2Vec2ForCTCAndIntensity
from ..utils.audio import rms_dbfs, norm01_from_dbfs

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, choices=['whisper','wav2vec2'], default='whisper')
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--audio', type=str, required=True)
    ap.add_argument('--max_new_tokens', type=int, default=128)
    return ap.parse_args()

def read_audio(path, sr=16000):
    wav, sr0 = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr0 != sr:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr0, target_sr=sr)
    wav = np.clip(wav.astype(np.float32), -1.0, 1.0)
    return wav, sr

def main():
    args = parse_args()
    wav, sr = read_audio(args.audio, 16000)

    if args.model == 'whisper':
        processor = WhisperProcessor.from_pretrained(args.checkpoint)
        model = WhisperForASRAndIntensity.from_pretrained(args.checkpoint).eval()
        inputs = processor.feature_extractor(wav, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            tokens = model.generate(inputs.input_features, max_new_tokens=args.max_new_tokens)
        text = processor.batch_decode(tokens, skip_special_tokens=True)[0]
        # intensity: use ground truth from audio as reference, model's regression is used in training only
        db = rms_dbfs(wav)
        print({"text": text, "intensity_dbfs": float(db), "intensity_norm": norm01_from_dbfs(db)})

    else:
        processor = Wav2Vec2Processor.from_pretrained(args.checkpoint)
        model = Wav2Vec2ForCTCAndIntensity.from_pretrained(args.checkpoint).eval()
        inputs = processor(wav, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        db = rms_dbfs(wav)
        print({"text": text, "intensity_dbfs": float(db), "intensity_norm": norm01_from_dbfs(db)})

if __name__ == "__main__":
    main()
