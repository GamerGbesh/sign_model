"""Unit tests for new augmentation operations."""
import numpy as np
import pytest

from asl import augment, config as C


def _sample_seq(T=32):
    rng = np.random.default_rng(42)
    return rng.standard_normal((T, C.FEATURE_DIM)).astype(np.float32)


def test_dropout_hands_shape_and_dtype():
    rng = np.random.default_rng(1)
    seq = _sample_seq()
    out = augment.dropout_hands(seq, rng, p=0.3)
    assert out.shape == seq.shape
    assert out.dtype == np.float32


def test_dropout_hands_zeroes_hand_slice():
    rng = np.random.default_rng(2)
    seq = np.ones((32, C.FEATURE_DIM), dtype=np.float32)
    # With p=1.0, hands should be dropped in 2-8 frame spans
    out = augment.dropout_hands(seq, rng, p=1.0)
    lh_zero = np.any(out[:, C.LH_SLICE] == 0.0)
    rh_zero = np.any(out[:, C.RH_SLICE] == 0.0)
    assert lh_zero or rh_zero, "Expected at least one hand to have dropped frames"
    # Pose slice should not be touched
    assert np.all(out[:, C.POSE_SLICE] == 1.0)


def test_frame_dropout_shape_and_dtype():
    rng = np.random.default_rng(3)
    seq = _sample_seq()
    out = augment.frame_dropout(seq, rng, p=0.03)
    assert out.shape == seq.shape
    assert out.dtype == np.float32


def test_frame_dropout_zeroes_frames():
    rng = np.random.default_rng(4)
    seq = np.ones((32, C.FEATURE_DIM), dtype=np.float32)
    out = augment.frame_dropout(seq, rng, p=0.5)
    # Check that any zeroed frame is completely zero across all 258 features
    zero_rows = np.all(out == 0.0, axis=1)
    nonzero_rows = np.all(out == 1.0, axis=1)
    assert np.all(zero_rows | nonzero_rows)


def test_augment_sequence_p_flip_zero():
    # If left hand has 2.0 and right hand has 0.0, with p_flip=0.0 left hand must never swap to right hand
    rng = np.random.default_rng(5)
    seq = np.zeros((32, C.FEATURE_DIM), dtype=np.float32)
    seq[:, C.LH_SLICE] = 2.0

    for _ in range(10):
        out = augment.augment_sequence(seq, rng, p_flip=0.0, p_dropout=False)
        # Left hand remains around 2.0, right hand only has small jitter noise (< 0.2)
        assert np.max(np.abs(out[:, C.RH_SLICE])) < 0.2, "Right hand received swapped hand coordinates"
        assert np.mean(out[:, C.LH_SLICE]) > 1.0, "Left hand lost its coordinates"
