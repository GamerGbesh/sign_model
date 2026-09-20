"""Fixed-duration, timestamp-based causal window sampling.

Provides a single function used identically across training and live inference
to ensure zero train/live temporal mismatch.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def sample_window(
    frames: np.ndarray,
    ts_ms: np.ndarray,
    t_end_ms: float,
    window_s: float = C.WINDOW_S,
    n: int = C.SEQ_LEN,
) -> np.ndarray:
    """(n, 258) sample at n uniform times in [t_end - window_s, t_end].

    Causal: each sample is the latest frame with ts <= sample time. Times before the
    first frame get a zero vector. Frames without a pose stay in the array as zero vectors.
    """
    frames = np.asarray(frames, dtype=np.float32)
    ts_ms = np.asarray(ts_ms)
    feat_dim = frames.shape[1] if frames.ndim == 2 and frames.shape[0] > 0 else C.FEATURE_DIM
    out = np.zeros((n, feat_dim), dtype=np.float32)

    if len(frames) == 0 or len(ts_ms) == 0:
        return out

    t_start_ms = t_end_ms - window_s * 1000.0
    sample_times = np.linspace(t_start_ms, t_end_ms, n)

    # Causal lookup: latest frame with ts <= sample_time
    indices = np.searchsorted(ts_ms, sample_times, side="right") - 1
    valid = indices >= 0
    if np.any(valid):
        out[valid] = frames[indices[valid]]

    return out
