"""Feature extraction, caching, and sign segmentation.

Extracts normalized 258-d landmark features in VIDEO mode, caches results
to disk, computes signing activity, and segments videos into active signing spans.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from . import config as C
from .landmarks import HolisticExtractor

EXTRACTOR_VERSION = "holistic_video_v1"


def _cache_path(video_path: Path) -> Path:
    stat = video_path.stat()
    raw = f"{video_path.resolve()}_{stat.st_size}_{stat.st_mtime_ns}_{EXTRACTOR_VERSION}"
    digest = hashlib.sha1(raw.encode()).hexdigest()
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return C.CACHE_DIR / f"{digest}.npz"


def extract_video(
    path: Path | str,
    cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Extract (frames, ts_ms, fps) from a video.

    Uses HolisticExtractor(running_mode="VIDEO"), fresh extractor per file,
    no flip, and preserves no-pose frames as zero vectors.
    """
    path = Path(path)
    cpath = _cache_path(path)
    if cache and cpath.exists():
        data = np.load(cpath)
        return data["frames"], data["ts_ms"], float(data["fps"])

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return np.zeros((0, C.FEATURE_DIM), dtype=np.float32), np.array([], dtype=np.int64), 30.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120 or np.isnan(fps):
        fps = 30.0

    frame_idx = 0
    last_ts_ms = -1
    vectors = []
    timestamps = []

    with HolisticExtractor(running_mode="VIDEO") as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Strictly increasing integer-ms timestamps
            raw_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            if raw_ts > last_ts_ms and not np.isnan(raw_ts):
                ts_ms = int(round(raw_ts))
            else:
                ts_ms = int(round(frame_idx * 1000.0 / fps))

            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + max(1, int(round(1000.0 / fps)))
            last_ts_ms = ts_ms

            # No flip! Raw unmirrored frame
            vec, _result, _pose_present = extractor(frame, timestamp_ms=ts_ms)
            vectors.append(vec)
            timestamps.append(ts_ms)
            frame_idx += 1

    cap.release()

    if vectors:
        frames = np.stack(vectors).astype(np.float32)
        ts_ms = np.array(timestamps, dtype=np.int64)
    else:
        frames = np.zeros((0, C.FEATURE_DIM), dtype=np.float32)
        ts_ms = np.array([], dtype=np.int64)

    if cache:
        np.savez_compressed(cpath, frames=frames, ts_ms=ts_ms, fps=fps)

    return frames, ts_ms, fps


def activity(frames: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Mark each frame as signing-active (1.0) or inactive (0.0).

    Active when:
      - at least one hand is present, AND
      - that hand's wrist is above the hip line (wrist_y < hip_y in normalized coords).
    Smoothed: gaps < 0.3s merged, segments < 0.35s dropped, padded 0.2s on each side.
    """
    frames = np.asarray(frames, dtype=np.float32)
    n = len(frames)
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    # Presence checks
    pose_present = np.any(frames[:, C.POSE_SLICE] != 0.0, axis=1)
    lh_present = np.any(frames[:, C.LH_SLICE] != 0.0, axis=1)
    rh_present = np.any(frames[:, C.RH_SLICE] != 0.0, axis=1)

    # Hip line: average of left hip (lm 23) and right hip (lm 24) y-coordinates
    # In POSE_SLICE: landmark k has (x, y, z, vis) -> y is at k*4 + 1
    l_hip_y = frames[:, 23 * 4 + 1]
    r_hip_y = frames[:, 24 * 4 + 1]
    hip_y = (l_hip_y + r_hip_y) / 2.0

    # Hand wrists: landmark 0 (x, y, z) -> y is at slice_start + 1
    lh_wrist_y = frames[:, C.LH_SLICE.start + 1]
    rh_wrist_y = frames[:, C.RH_SLICE.start + 1]

    # In normalized coords, y grows downward: "above hip" means wrist_y < hip_y
    lh_above = lh_present & (lh_wrist_y < hip_y)
    rh_above = rh_present & (rh_wrist_y < hip_y)

    raw_active = pose_present & (lh_above | rh_above)

    # Convert to runs and apply temporal filters
    gap_frames = max(1, int(round(0.3 * fps)))
    min_frames = max(1, int(round(0.35 * fps)))
    pad_frames = int(round(0.2 * fps))

    # 1. Merge gaps under 0.3s between active regions
    merged = raw_active.copy()
    in_gap = False
    gap_start = 0
    for i in range(n):
        if not merged[i]:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                in_gap = False
                # If bounded by active at gap_start - 1 and current i
                if gap_start > 0 and (i - gap_start) < gap_frames:
                    merged[gap_start:i] = True

    # 2. Drop active segments under 0.35s
    filtered = merged.copy()
    in_seg = False
    seg_start = 0
    for i in range(n):
        if filtered[i]:
            if not in_seg:
                in_seg = True
                seg_start = i
        else:
            if in_seg:
                in_seg = False
                if (i - seg_start) < min_frames:
                    filtered[seg_start:i] = False
    if in_seg and (n - seg_start) < min_frames:
        filtered[seg_start:n] = False

    # 3. Pad 0.2s on each side of active segments
    padded = np.zeros(n, dtype=bool)
    in_seg = False
    seg_start = 0
    for i in range(n):
        if filtered[i]:
            if not in_seg:
                in_seg = True
                seg_start = i
        else:
            if in_seg:
                in_seg = False
                p_start = max(0, seg_start - pad_frames)
                p_end = min(n, i + pad_frames)
                padded[p_start:p_end] = True
    if in_seg:
        p_start = max(0, seg_start - pad_frames)
        p_end = min(n, n + pad_frames)
        padded[p_start:p_end] = True

    return padded.astype(np.float32)


def segments(frames: np.ndarray, ts_ms: np.ndarray, fps: float | None = None) -> List[Tuple[float, float]]:
    """Return a list of (t_start_ms, t_end_ms) for all signing segments."""
    n = len(frames)
    if n == 0 or len(ts_ms) == 0:
        return []

    if fps is None:
        duration_s = (ts_ms[-1] - ts_ms[0]) / 1000.0 if n > 1 else 0.0
        fps = (n / duration_s) if duration_s > 0 else 30.0

    act = activity(frames, fps=fps)
    seg_list = []
    in_seg = False
    seg_start = 0

    for i in range(n):
        if act[i] > 0.5:
            if not in_seg:
                in_seg = True
                seg_start = i
        else:
            if in_seg:
                in_seg = False
                t_start = float(ts_ms[seg_start])
                t_end = float(ts_ms[i - 1])
                seg_list.append((t_start, t_end))
    if in_seg:
        t_start = float(ts_ms[seg_start])
        t_end = float(ts_ms[n - 1])
        seg_list.append((t_start, t_end))

    return seg_list
