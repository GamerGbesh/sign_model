"""Unit tests for video feature extraction, activity detection, and segmentation."""
import numpy as np
import pytest

from asl import config as C
from asl.video_features import activity, segments


def _create_synthetic_frame(pose_present=True, lh_present=False, rh_present=False,
                            lh_wrist_y=0.0, rh_wrist_y=0.0, hip_y=0.8):
    """Create a single 258-dim landmark vector with specified properties."""
    vec = np.zeros(C.FEATURE_DIM, dtype=np.float32)
    if pose_present:
        # Fill some dummy pose landmarks
        vec[C.POSE_SLICE] = 0.1
        # Set hips (lm 23 and 24)
        vec[23 * 4 + 1] = hip_y
        vec[24 * 4 + 1] = hip_y
    if lh_present:
        vec[C.LH_SLICE] = 0.2
        # Set left wrist y (index 133)
        vec[C.LH_SLICE.start + 1] = lh_wrist_y
    if rh_present:
        vec[C.RH_SLICE] = 0.2
        # Set right wrist y (index 196)
        vec[C.RH_SLICE.start + 1] = rh_wrist_y
    return vec


def test_activity_empty():
    act = activity(np.zeros((0, C.FEATURE_DIM), dtype=np.float32))
    assert len(act) == 0


def test_activity_detection_wrist_above_hip():
    # Hip at y=0.8. Wrist above hip at y=0.2 (recall y grows downward).
    active_frame = _create_synthetic_frame(pose_present=True, lh_present=True, lh_wrist_y=0.2, hip_y=0.8)
    inactive_frame = _create_synthetic_frame(pose_present=True, lh_present=True, lh_wrist_y=1.0, hip_y=0.8)
    no_pose_frame = _create_synthetic_frame(pose_present=False, lh_present=True, lh_wrist_y=0.2, hip_y=0.8)
    no_hand_frame = _create_synthetic_frame(pose_present=True, lh_present=False, hip_y=0.8)

    # 30 fps = 30 frames per second
    fps = 30.0

    # Build a sequence: 10 inactive, 30 active (1.0s), 10 inactive
    frames = np.stack([inactive_frame] * 10 + [active_frame] * 30 + [inactive_frame] * 10)
    act = activity(frames, fps=fps)

    # Should detect the active segment and pad ~6 frames (0.2s * 30fps) on each side
    assert act[0] == 0.0
    # Inside the active region
    assert np.all(act[10:40] == 1.0)
    # Padded before and after
    assert act[9] == 1.0  # padded into inactive
    assert act[41] == 1.0  # padded into inactive


def test_activity_drops_short_segments():
    fps = 30.0
    # 0.2s segment = 6 frames (< 0.35s / 10.5 frames)
    active_frame = _create_synthetic_frame(pose_present=True, rh_present=True, rh_wrist_y=0.0, hip_y=0.8)
    inactive_frame = _create_synthetic_frame(pose_present=True, rh_present=False, hip_y=0.8)

    frames = np.stack([inactive_frame] * 20 + [active_frame] * 6 + [inactive_frame] * 20)
    act = activity(frames, fps=fps)
    assert np.all(act == 0.0), "Segment under 0.35s should have been dropped"


def test_activity_merges_short_gaps():
    fps = 30.0
    active_frame = _create_synthetic_frame(pose_present=True, rh_present=True, rh_wrist_y=0.0, hip_y=0.8)
    inactive_frame = _create_synthetic_frame(pose_present=True, rh_present=False, hip_y=0.8)

    # Active for 20 frames (0.67s), inactive gap for 5 frames (~0.16s < 0.3s), active for 20 frames
    frames = np.stack([active_frame] * 20 + [inactive_frame] * 5 + [active_frame] * 20)
    act = activity(frames, fps=fps)
    # The gap should be filled
    assert np.all(act == 1.0), "Gap under 0.3s between active regions should have been merged"


def test_segments_timestamps():
    fps = 30.0
    active_frame = _create_synthetic_frame(pose_present=True, rh_present=True, rh_wrist_y=0.0, hip_y=0.8)
    inactive_frame = _create_synthetic_frame(pose_present=True, rh_present=False, hip_y=0.8)

    # 30 inactive frames (1.0s), 30 active frames (1.0s), 30 inactive frames (1.0s)
    frames = np.stack([inactive_frame] * 30 + [active_frame] * 30 + [inactive_frame] * 30)
    ts_ms = np.arange(90) * (1000.0 / fps)

    segs = segments(frames, ts_ms, fps=fps)
    assert len(segs) == 1
    t_start, t_end = segs[0]
    # Active started at frame 30 (1000ms) with 0.2s padding (starts ~800ms)
    # Active ended at frame 59 (~1966ms) with 0.2s padding (ends ~2166ms)
    assert t_start <= 1000.0
    assert t_end >= 1966.0
