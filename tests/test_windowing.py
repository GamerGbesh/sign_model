"""Unit tests for fixed-duration timestamp-based windowing."""
import numpy as np
import pytest

from asl import config as C
from asl.windowing import sample_window


def test_sample_window_shape_and_dtype():
    frames = np.ones((60, C.FEATURE_DIM), dtype=np.float32)
    ts_ms = np.linspace(0, 2000, 60)
    out = sample_window(frames, ts_ms, t_end_ms=2000.0, window_s=2.0, n=C.SEQ_LEN)
    assert out.shape == (C.SEQ_LEN, C.FEATURE_DIM)
    assert out.dtype == np.float32


def test_sample_window_zero_padding_at_start():
    # First frame at 1000ms. Window spans [-1000ms, 1000ms].
    # All sample times < 1000ms should be zero vectors.
    frames = np.ones((10, C.FEATURE_DIM), dtype=np.float32) * 5.0
    ts_ms = np.linspace(1000, 2000, 10)
    out = sample_window(frames, ts_ms, t_end_ms=1000.0, window_s=2.0, n=32)

    sample_times = np.linspace(-1000.0, 1000.0, 32)
    for i, t in enumerate(sample_times):
        if t < 1000.0:
            assert np.all(out[i] == 0.0), f"Sample at {t}ms should be zero-padded"
        else:
            assert np.all(out[i] == 5.0), f"Sample at {t}ms should be 5.0"


def test_sample_window_causality():
    # Frame values encode their timestamp
    ts_ms = np.array([0, 500, 1000, 1500, 2000, 2500, 3000], dtype=np.float32)
    frames = np.tile(ts_ms[:, None], (1, C.FEATURE_DIM))

    # Query at 1200ms
    out = sample_window(frames, ts_ms, t_end_ms=1200.0, window_s=1.0, n=5)
    sample_times = np.linspace(200.0, 1200.0, 5)  # [200, 450, 700, 950, 1200]

    for i, st in enumerate(sample_times):
        # Frame value should be <= st
        val = out[i, 0]
        assert val <= st, f"Sample at {st}ms has value {val} from future!"
        # And should be the latest frame <= st
        expected_latest = ts_ms[ts_ms <= st][-1] if np.any(ts_ms <= st) else 0.0
        assert val == expected_latest


def test_sample_window_irregular_fps():
    # A ramp signal sampled at 30 fps vs irregular ~15-25 fps
    t_max = 3000.0
    # Regular 30 fps: ~33.3ms intervals
    ts_reg = np.arange(0, t_max, 33.3)
    frames_reg = np.tile(ts_reg[:, None], (1, C.FEATURE_DIM))

    # Irregular timestamps
    rng = np.random.default_rng(42)
    dt = rng.uniform(25.0, 65.0, size=100)
    ts_irreg = np.cumsum(np.concatenate([[0], dt]))
    ts_irreg = ts_irreg[ts_irreg <= t_max]
    frames_irreg = np.tile(ts_irreg[:, None], (1, C.FEATURE_DIM))

    out_reg = sample_window(frames_reg, ts_reg, t_end_ms=2500.0, window_s=2.0, n=32)
    out_irreg = sample_window(frames_irreg, ts_irreg, t_end_ms=2500.0, window_s=2.0, n=32)

    # Because frames are step-held between discrete samples, max difference
    # between regular and irregular at any sample time cannot exceed the max frame interval (~65ms)
    diff = np.abs(out_reg[:, 0] - out_irreg[:, 0])
    assert np.all(diff <= 70.0), f"Max discrepancy {diff.max()} exceeded tolerance"


def test_sample_window_empty_and_zeros():
    empty_frames = np.zeros((0, C.FEATURE_DIM), dtype=np.float32)
    empty_ts = np.array([], dtype=np.float32)
    out = sample_window(empty_frames, empty_ts, t_end_ms=1000.0)
    assert out.shape == (C.SEQ_LEN, C.FEATURE_DIM)
    assert np.all(out == 0.0)
