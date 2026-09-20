"""Unit tests for custom dataset builder."""
from pathlib import Path
import numpy as np
import pytest

from asl.dataset_custom import (
    compute_overlap,
    parse_label,
    split_videos,
    thin_frames,
)


def test_parse_label():
    base = Path("/workspace/videos")
    # Subdirectory layout
    assert parse_label(base / "hello" / "clip1.mp4", base) == "hello"
    assert parse_label(base / "THANK_YOU" / "clip_02.mp4", base) == "thank_you"

    # Flat file layout
    assert parse_label(base / "hello_1.mp4", base) == "hello"
    assert parse_label(base / "thank_you_002.mp4", base) == "thank_you"
    assert parse_label(base / "goodbye-3.webm", base) == "goodbye"
    assert parse_label(base / "yes.avi", base) == "yes"


def test_compute_overlap():
    # Segment S = [1000, 2000], duration = 1000
    s_start, s_end = 1000.0, 2000.0

    # No overlap (before)
    assert compute_overlap(0.0, 800.0, s_start, s_end) == 0.0

    # No overlap (after)
    assert compute_overlap(2200.0, 3000.0, s_start, s_end) == 0.0

    # Full containment: W = [500, 2500] covers all 1000ms of S
    assert compute_overlap(500.0, 2500.0, s_start, s_end) == 1.0

    # Partial overlap: W = [500, 1700] covers [1000, 1700] (700ms / 1000ms = 0.7)
    assert abs(compute_overlap(500.0, 1700.0, s_start, s_end) - 0.7) < 1e-5

    # Partial overlap: W = [1300, 2500] covers [1300, 2000] (700ms / 1000ms = 0.7)
    assert abs(compute_overlap(1300.0, 2500.0, s_start, s_end) - 0.7) < 1e-5


def test_split_isolation_and_strategies():
    video_groups = {
        "hello": [Path(f"videos/hello/{i}.mp4") for i in range(8)],      # >= 4: video_split
        "goodbye": [Path(f"videos/goodbye/{i}.mp4") for i in range(2)],  # < 4: chronological
    }
    splits, strategies = split_videos(video_groups, seed=42)

    assert strategies["hello"] == "video_split"
    assert strategies["goodbye"] == "chronological"

    # Assert disjointness for video_split
    h_tr = {p.name for p in splits["hello"]["train"]}
    h_va = {p.name for p in splits["hello"]["val"]}
    h_te = {p.name for p in splits["hello"]["test"]}
    assert len(h_tr) > 0 and len(h_va) > 0 and len(h_te) > 0
    assert h_tr.isdisjoint(h_va)
    assert h_tr.isdisjoint(h_te)
    assert h_va.isdisjoint(h_te)


def test_thin_frames():
    rng = np.random.default_rng(0)
    fps = 30.0
    n = 90  # 3 seconds
    ts_ms = np.arange(n) * (1000.0 / fps)
    frames = np.ones((n, 258), dtype=np.float32)

    aug_frames, aug_ts = thin_frames(frames, ts_ms, target_fps=15.0, rng=rng, jitter_ms=10.0)
    # At 15 fps for 3 seconds, should have ~45 frames
    assert 40 <= len(aug_frames) <= 50
    # Strictly non-decreasing timestamps
    assert np.all(np.diff(aug_ts) > 0)
