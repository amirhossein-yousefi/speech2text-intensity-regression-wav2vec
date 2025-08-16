# sagemaker/code/inference.py
# Custom SageMaker inference script for multitask ASR + intensity regression.
# Works with Hugging Face DLC via HuggingFaceModel + custom entry point.
# Returns: {"text": str, "intensity_norm": float (0..1), "intensity_dbfs": float [-60,0]}

import os
import io
import json
import base64
import wave
import numpy as np
import torch

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoModelForCTC,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SR = 16000  # typical for wav2vec2; Whisper resamples internally via processor


def _decode_wav_base64(b64_str):
    """Decode 16-bit PCM or 32-bit PCM WAV from base64 -> (float32 mono, sr)."""
    raw = base64.b64decode(b64_str)
    with wave.open(io.BytesIO(raw), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()  # bytes per sample
        pcm = wf.readframes(n)

    if sw == 2:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / (2 ** 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sw} bytes")

    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    return audio, sr


def _resample_np(x, sr, target_sr=TARGET_SR):
    if sr == target_sr:
        return x, sr
    # Simple linear resampling to avoid extra deps
    old_idx = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    new_len = int(round(len(x) * (target_sr / float(sr))))
    new_idx = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    y = np.interp(new_idx, old_idx, x).astype(np.float32)
    return y, target_sr


def _rms_dbfs(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return -60.0
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    if rms <= 0:
        return -60.0
    db = 20.0 * np.log10(rms)
    return float(np.clip(db, -60.0, 0.0))


def _norm_from_dbfs(db):
    # From your README: norm = (dbfs + 60) / 60
    return float((db + 60.0) / 60.0)


def model_fn(model_dir):
    """
    Load model + processor. We try to infer kind from env or config.
    Env (optional):
      MODEL_KIND: "wav2vec2" or "whisper"  (fallback: infer from config.json)
      LANGUAGE: e.g. "en" for whisper decoding
    """
    model_kind = os.environ.get("MODEL_KIND", "").lower()
    language = os.environ.get("LANGUAGE", "en")

    # Heuristic: read config.json to guess the model type if not provided
    cfg_path = os.path.join(model_dir, "config.json")
    cfg_text = ""
    if os.path.exists(cfg_path):
        try:
            cfg_text = open(cfg_path, "r", encoding="utf-8").read()
        except Exception:
            pass

    if not model_kind:
        if "whisper" in cfg_text.lower():
            model_kind = "whisper"
        else:
            model_kind = "wav2vec2"

    state = {"kind": model_kind, "language": language}

    if model_kind == "whisper":
        processor = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
        model.to(DEVICE)
        model.eval()
        state.update({"model": model, "processor": processor})
    else:
        # default to wav2vec2-CTC
        processor = AutoProcessor.from_pretrained(model_dir)
        try:
            model = AutoModelForCTC.from_pretrained(model_dir)
        except Exception:
            # Fallback: attempt to import your custom multitask class
            # so custom intensity head can load, if present.
            try:
                from speech_mtl.models.multitask_wav2vec2 import MultiTaskWav2Vec2ForCTC
                model = MultiTaskWav2Vec2ForCTC.from_pretrained(model_dir)
            except Exception as e:
                raise RuntimeError(
                    "Could not load wav2vec2 model. Ensure your model artifacts "
                    "and code are packaged. Error: %s" % str(e)
                )
        model.to(DEVICE)
        model.eval()
        state.update({"model": model, "processor": processor})

    return state


def input_fn(request_body, request_content_type="application/json"):
    """
    Accepts:
      {
        "audio_base64": "<...>",       # base64 WAV (PCM16/32)
        "audio": [floats],             # OR raw float array in [-1,1]
        "sampling_rate": 16000         # required if "audio" array is provided
      }
    """
    if request_content_type and "json" not in request_content_type:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    data = json.loads(request_body)

    if "audio_base64" in data:
        x, sr = _decode_wav_base64(data["audio_base64"])
    elif "audio" in data and "sampling_rate" in data:
        x = np.asarray(data["audio"], dtype=np.float32)
        sr = int(data["sampling_rate"])
    else:
        raise ValueError('Provide either "audio_base64" or "audio"+"sampling_rate".')

    # Keep a copy for intensity regardless of model kind
    return {"audio": x, "sr": sr}


@torch.inference_mode()
def predict_fn(inputs, state):
    audio = inputs["audio"]
    sr = inputs["sr"]

    kind = state["kind"]
    model = state["model"]
    processor = state["processor"]
    language = state.get("language", "en")

    # Compute intensity from audio as a robust fallback.
    # If the model returns a custom intensity head, we’ll use it instead.
    dbfs_from_audio = _rms_dbfs(audio)
    intensity_norm = _norm_from_dbfs(dbfs_from_audio)
    intensity_dbfs = dbfs_from_audio

    if kind == "whisper":
        # Whisper: processor expects raw audio; it will handle features internally.
        # Forcing English decoding; adjust with LANGUAGE env if needed.
        features = processor(
            audio=audio, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(DEVICE)
        gen_ids = model.generate(features, max_new_tokens=256)
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        # If a custom regression head exists, try to compute intensity from it
        try:
            # Whisper forward needs input_features (already computed)
            outputs = model(features)
            # Guess common names for a scalar regression output
            maybe = getattr(outputs, "intensity", None) or getattr(outputs, "regression", None)
            if maybe is not None:
                val = maybe if isinstance(maybe, torch.Tensor) else torch.tensor(maybe)
                val = val.squeeze().float()
                # Treat as raw value in [-inf, +inf], squash to 0..1
                intensity_norm = float(torch.sigmoid(val).item())
                intensity_dbfs = float(intensity_norm * 60.0 - 60.0)
        except Exception:
            pass

        return {"text": text, "intensity_norm": intensity_norm, "intensity_dbfs": intensity_dbfs}

    # wav2vec2 path
    audio_16k, sr_16k = _resample_np(audio, sr, TARGET_SR)
    pc = processor(audio=audio_16k, sampling_rate=sr_16k, return_tensors="pt")
    pc = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in pc.items()}

    outputs = model(**pc)
    # CTC decode
    logits = outputs.logits
    pred_ids = torch.argmax(logits, dim=-1)
    try:
        text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    except Exception:
        # Fallback if tokenizer missing special tokens map
        text = processor.batch_decode(pred_ids)[0]

    # Optional: try to pull a custom intensity head from model output
    try:
        maybe = getattr(outputs, "intensity", None) or getattr(outputs, "intensity_pred", None)
        if maybe is not None:
            val = maybe if isinstance(maybe, torch.Tensor) else torch.tensor(maybe)
            val = val.squeeze().float()
            intensity_norm = float(torch.sigmoid(val).item())
            intensity_dbfs = float(intensity_norm * 60.0 - 60.0)
    except Exception:
        pass

    return {"text": text, "intensity_norm": intensity_norm, "intensity_dbfs": intensity_dbfs}


def output_fn(prediction, accept="application/json"):
    if "json" not in accept:
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction), "application/json"
