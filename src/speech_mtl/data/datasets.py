from __future__ import annotations
import os
from typing import Optional, Dict, Any

import numpy as np
from datasets import load_dataset, Audio
from ..utils.audio import rms_dbfs, norm01_from_dbfs, lufs_level


def _ensure_cache_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    return path


def _maybe_rename_text_column(dset, text_col: str):
    """Ensure a column named `text_col` exists by renaming common alternatives."""
    if text_col in dset.column_names:
        return dset
    for alt in ("sentence", "transcript", "transcription", "normalized_text", "text"):
        if alt in dset.column_names:
            if alt != text_col:
                dset = dset.rename_column(alt, text_col)
            return dset
    # If none exist, create empty text (will be filtered out later)
    return dset.map(lambda e: {text_col: ""})


def load_asr_dataset_with_intensity(
    dataset: str = "librispeech_asr",
    language: Optional[str] = None,           # For Common Voice use language code; for openslr/librispeech_asr use config (e.g., "clean")
    train_split: str = "train.clean.100",
    eval_split: str = "validation.clean",
    test_split: Optional[str] = None,
    audio_column: str = "audio",
    text_column: str = "text",
    sample_rate: int = 16000,
    cache_dir: Optional[str] = None,
    num_proc: int = 1,
    # NEW: hard caps for tiny datasets
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
):
    """
    Load an ASR dataset and add:
        - intensity_dbfs: float in [-60, 0]
        - intensity_norm: float in [0, 1]
        - (optionally) intensity_lufs
    Returns: (ds_train, ds_eval, ds_test)
    """
    cache_dir = _ensure_cache_dir(cache_dir)

    def _ld(split: Optional[str]):
        if not split:
            return None
        # Common Voice (scriptless >= 16.x/17.x): needs language as config
        if dataset.startswith("mozilla-foundation/common_voice_"):
            cfg = language or "en"
            return load_dataset(dataset, cfg, split=split, cache_dir=cache_dir)
        # openslr/librispeech_asr: needs config like "clean" or "other"
        if dataset == "openslr/librispeech_asr":
            cfg = language or "clean"
            return load_dataset(dataset, cfg, split=split, cache_dir=cache_dir)
        # classic librispeech_asr accepts "train.clean.100" style splits
        if dataset == "librispeech_asr":
            return load_dataset(dataset, split=split, cache_dir=cache_dir)
        # generic case: pass language as config if provided
        if language:
            return load_dataset(dataset, language, split=split, cache_dir=cache_dir)
        return load_dataset(dataset, split=split, cache_dir=cache_dir)

    ds_train = _ld(train_split)
    ds_eval  = _ld(eval_split)
    ds_test  = _ld(test_split)

    # ---- Limit size BEFORE heavy processing ----
    def _limit(d, n):
        if d is None or n is None or n <= 0:
            return d
        n = min(n, d.num_rows)
        return d.shuffle(seed=seed).select(range(n))

    ds_train = _limit(ds_train, max_train_samples)
    ds_eval  = _limit(ds_eval,  max_eval_samples)
    ds_test  = _limit(ds_test,  max_test_samples)

    # ---- Decode audio at target sample rate (arrays in-memory; OS-agnostic) ----
    def cast_audio_decode(dset):
        return dset.cast_column(audio_column, Audio(sampling_rate=sample_rate))

    ds_train = cast_audio_decode(ds_train)
    ds_eval  = cast_audio_decode(ds_eval)
    if ds_test is not None:
        ds_test = cast_audio_decode(ds_test)

    # ---- Ensure we have the expected text column name ----
    ds_train = _maybe_rename_text_column(ds_train, text_column)
    ds_eval  = _maybe_rename_text_column(ds_eval,  text_column)
    if ds_test is not None:
        ds_test = _maybe_rename_text_column(ds_test, text_column)

    # ---- Compute intensity from decoded arrays (fast, no filesystem paths) ----
    def add_intensity(batch: Dict[str, Any]):
        audio = batch[audio_column]
        wav = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        if wav.ndim > 1:
            wav = wav.mean(axis=1).astype(np.float32, copy=False)

        db = rms_dbfs(wav)            # clamped to [-60, 0]
        norm = norm01_from_dbfs(db)   # [0, 1]
        try:
            lufs = lufs_level(wav, sr)
        except Exception:
            lufs = None

        batch["intensity_dbfs"] = float(db)
        batch["intensity_norm"] = float(norm)
        if lufs is not None:
            batch["intensity_lufs"] = float(lufs)
        return batch

    ds_train = ds_train.map(add_intensity, num_proc=num_proc, desc="Computing intensity (train)")
    ds_eval  = ds_eval.map(add_intensity, num_proc=num_proc, desc="Computing intensity (eval)")
    if ds_test is not None:
        ds_test = ds_test.map(add_intensity, num_proc=num_proc, desc="Computing intensity (test)")

    # ---- Filter empty transcripts ----
    def has_text(b):
        return (text_column in b) and isinstance(b[text_column], str) and len(b[text_column].strip()) > 0

    ds_train = ds_train.filter(has_text)
    ds_eval  = ds_eval.filter(has_text)
    if ds_test is not None:
        ds_test = ds_test.filter(has_text)

    return ds_train, ds_eval, ds_test
