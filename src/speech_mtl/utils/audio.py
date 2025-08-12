from __future__ import annotations
import numpy as np

def rms_dbfs(x: np.ndarray, eps: float = 1e-12) -> float:
    """Root-mean-square in dBFS (bounded to [-60, 0]).
    Input x should be float waveform in [-1, 1].
    """
    x = np.asarray(x, dtype=np.float32)
    r = np.sqrt(np.maximum(np.mean(x ** 2), eps))
    db = 20.0 * np.log10(r + eps)
    db = float(np.clip(db, -60.0, 0.0))
    return db

def norm01_from_dbfs(dbfs: float) -> float:
    """Normalize [-60, 0] dBFS to [0,1]."""
    return (dbfs + 60.0) / 60.0

def lufs_level(x: np.ndarray, sr: int) -> float | None:
    """Compute LUFS with pyloudnorm if available, else None."""
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(np.asarray(x, dtype=np.float32))
        return float(lufs)
    except Exception:
        return None
