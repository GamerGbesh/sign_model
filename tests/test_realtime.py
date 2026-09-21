"""Unit tests for StreamingRecognizer with stub extractor and stub model."""
import numpy as np
import torch
import torch.nn as nn
import pytest

from asl import config as C
from asl.realtime import StreamingRecognizer


class StubExtractor:
    def __init__(self, hand_present=True):
        self.hand_present = hand_present

    def __call__(self, frame, timestamp_ms=0):
        vec = np.zeros(C.FEATURE_DIM, dtype=np.float32)
        if self.hand_present:
            # Mark LH slice present
            vec[C.LH_SLICE] = 1.0
            vec[C.POSE_SLICE] = 1.0
        return vec, None, True

    def close(self):
        pass


class StubModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.target_class = 1  # 0: idle, 1: "hello", 2: "yes"

    def forward(self, x):
        # x: (B, T, F)
        b = x.shape[0]
        logits = torch.zeros((b, self.num_classes), dtype=torch.float32)
        # Give high confidence to target_class
        logits[:, self.target_class] = 10.0
        return logits


def test_commit_after_debounce_n():
    labels = ["idle", "hello", "yes"]
    model = StubModel(num_classes=3)
    model.target_class = 1  # "hello"
    extractor = StubExtractor(hand_present=True)

    recognizer = StreamingRecognizer(
        extractor=extractor,
        model=model,
        labels=labels,
        debounce_n=3,
        infer_stride_ms=100,
        window_s=1.0,
        conf_threshold=0.5,
    )

    # Push frames to pass warming up (750 ms)
    # Stride is 100 ms
    commits = []
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Frames at 0, 100, 200, ..., 1200 ms
    for t_ms in range(0, 1500, 100):
        res = recognizer.push(dummy_frame, t_ms)
        if res["commit"]:
            commits.append((t_ms, res["commit"]))

    # Expected: exactly one commit of "hello"
    assert len(commits) == 1
    assert commits[0][1] == "hello"
    assert "hello" in recognizer.phrase


def test_idle_resets_streak():
    labels = ["idle", "hello", "yes"]
    model = StubModel(num_classes=3)
    model.target_class = 1  # "hello"
    extractor = StubExtractor(hand_present=True)

    recognizer = StreamingRecognizer(
        extractor=extractor,
        model=model,
        labels=labels,
        debounce_n=3,
        infer_stride_ms=100,
        window_s=1.0,
    )
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Warm up: 0 to 700 ms
    for t in range(0, 800, 100):
        recognizer.push(dummy_frame, t)

    # Two strides of "hello" (need 3 for commit)
    r1 = recognizer.push(dummy_frame, 800)
    assert r1["commit"] is None
    r2 = recognizer.push(dummy_frame, 900)
    assert r2["commit"] is None
    assert recognizer.streak_count == 2

    # Now predict idle (target_class = 0)
    model.target_class = 0
    r3 = recognizer.push(dummy_frame, 1000)
    assert r3["label"] == "idle"
    assert recognizer.streak_count == 0  # Streak was reset!

    # Switch back to "hello": should start counting from 1, not commit on stride 1
    model.target_class = 1
    r4 = recognizer.push(dummy_frame, 1100)
    assert r4["commit"] is None
    assert recognizer.streak_count == 1


def test_repeated_word_cooldown():
    labels = ["idle", "hello", "yes"]
    model = StubModel(num_classes=3)
    extractor = StubExtractor(hand_present=True)

    recognizer = StreamingRecognizer(
        extractor=extractor,
        model=model,
        labels=labels,
        debounce_n=2,
        infer_stride_ms=100,
        window_s=1.0,
    )
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Warm up: 0 to 800ms with idle
    model.target_class = 0
    for t in range(0, 900, 100):
        recognizer.push(dummy_frame, t)

    # First sign: "hello"
    model.target_class = 1
    recognizer.push(dummy_frame, 900)
    r = recognizer.push(dummy_frame, 1000)
    assert r["commit"] == "hello"

    # Keep predicting "hello" - should NOT commit again (cooldown)
    r = recognizer.push(dummy_frame, 1100)
    assert r["commit"] is None
    r = recognizer.push(dummy_frame, 1200)
    assert r["commit"] is None

    # Now idle for >= 400ms (5 strides of 100ms)
    model.target_class = 0
    for t in range(1300, 1800, 100):
        recognizer.push(dummy_frame, t)

    # Now sign "hello" again!
    model.target_class = 1
    recognizer.push(dummy_frame, 1800)
    r = recognizer.push(dummy_frame, 1900)
    assert r["commit"] == "hello"
    assert recognizer.phrase == ["hello", "hello"]


def test_out_of_order_timestamps_dropped():
    labels = ["idle", "hello", "yes"]
    model = StubModel(num_classes=3)
    extractor = StubExtractor(hand_present=True)

    recognizer = StreamingRecognizer(
        extractor=extractor,
        model=model,
        labels=labels,
        debounce_n=2,
        infer_stride_ms=100,
    )
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    recognizer.push(dummy_frame, 500)
    recognizer.push(dummy_frame, 600)
    # Out of order: 550 < 600
    res = recognizer.push(dummy_frame, 550)
    # Should not break internal state; last_t_ms stays 600
    assert recognizer.last_t_ms == 600


def test_cooldown_real_time_accumulation():
    """Verifies that at 30 FPS (~33ms/frame), 400ms of real-time idle satisfies the cooldown."""
    labels = ["idle", "hello", "yes"]
    model = StubModel(num_classes=3)
    model.target_class = 1  # "hello"
    extractor = StubExtractor(hand_present=True)

    recognizer = StreamingRecognizer(
        extractor=extractor,
        model=model,
        labels=labels,
        debounce_n=2,
        infer_stride_ms=100,
        window_s=1.0,
    )
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Warm up buffer with 30 fps idle frames
    model.target_class = 0
    t = 0
    for _ in range(30):
        t += 33
        recognizer.push(dummy_frame, t)

    # Now sign "hello" and commit
    model.target_class = 1
    r = None
    for _ in range(10):
        t += 33
        r = recognizer.push(dummy_frame, t)
        if r["commit"] == "hello":
            break
    assert r["commit"] == "hello"
    assert recognizer.cooldown_satisfied is False

    # Now simulate ~550ms of real-time idle at 30 fps (17 frames)
    model.target_class = 0  # idle
    for _ in range(17):
        t += 33
        recognizer.push(dummy_frame, t)

    # Cooldown should now be satisfied because ~550ms real time elapsed
    assert recognizer.cooldown_satisfied is True

