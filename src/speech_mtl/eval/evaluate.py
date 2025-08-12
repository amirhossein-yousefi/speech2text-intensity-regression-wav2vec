from __future__ import annotations
import argparse, numpy as np, torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, Wav2Vec2Processor
from jiwer import wer
from ..utils.audio import rms_dbfs, norm01_from_dbfs
from ..models.multitask_whisper import WhisperForASRAndIntensity
from ..models.multitask_wav2vec2 import Wav2Vec2ForCTCAndIntensity

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='librispeech_asr')
    ap.add_argument('--split', type=str, default='test.clean')
    ap.add_argument('--language', type=str, default='en')
    ap.add_argument('--text_column', type=str, default='text')
    ap.add_argument('--whisper_model_dir', type=str, default=None)
    ap.add_argument('--wav2vec2_model_dir', type=str, default=None)
    return ap.parse_args()

def eval_whisper(model_dir, dataset, split, text_column):
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForASRAndIntensity.from_pretrained(model_dir).eval()
    ds = load_dataset(dataset)[split].cast_column("audio", Audio(sampling_rate=16000))

    preds = []
    refs  = []
    ints_pred = []
    ints_true = []

    for ex in ds.select(range(min(200, len(ds)))):  # cap to 200 for speed
        audio = ex["audio"]
        inputs = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        with torch.no_grad():
            gen_tokens = model.generate(inputs.input_features, max_new_tokens=128)
        text = processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        preds.append(text)
        refs.append(ex[text_column])

        # intensity pred
        with torch.no_grad():
            out = model(input_features=inputs.input_features, labels=torch.zeros((1,1), dtype=torch.long))  # trick to run encoder
            # No intensity output is returned explicitly; we recompute via forward
            # Alternative: do a single pass with labels missing; but for eval we rely on dbfs true
        dbfs = rms_dbfs(audio["array"])
        ints_true.append(norm01_from_dbfs(dbfs))

    w = wer(refs, preds)
    return {"wer": float(w)}

def eval_w2v(model_dir, dataset, split, text_column):
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTCAndIntensity.from_pretrained(model_dir).eval()
    ds = load_dataset(dataset)[split].cast_column("audio", Audio(sampling_rate=16000))

    preds, refs = [], []
    for ex in ds.select(range(min(200, len(ds)))):
        audio = ex["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
        preds.append(text)
        refs.append(ex[text_column])

    w = wer(refs, preds)
    return {"wer": float(w)}

def main():
    args = parse_args()
    if args.whisper_model_dir:
        res_w = eval_whisper(args.whisper_model_dir, args.dataset, args.split, args.text_column)
        print("Whisper:", res_w)
    if args.wav2vec2_model_dir:
        res_v = eval_w2v(args.wav2vec2_model_dir, args.dataset, args.split, args.text_column)
        print("Wav2Vec2:", res_v)

if __name__ == "__main__":
    main()
