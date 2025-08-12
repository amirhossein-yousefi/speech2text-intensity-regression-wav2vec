from src.speech_mtl.utils.audio import rms_dbfs, norm01_from_dbfs
import numpy as np

def test_rms_dbfs_range():
    x = np.zeros(16000, dtype=np.float32)
    db = rms_dbfs(x)
    assert -60.0 <= db <= 0.0

def test_norm():
    assert abs(norm01_from_dbfs(-60) - 0.0) < 1e-6
    assert abs(norm01_from_dbfs(0) - 1.0) < 1e-6
