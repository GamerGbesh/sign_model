"""Custom dataset builder for SignSpeak.

Extracts fixed-duration causal windows from custom video files, performs video-level
splits (stratified or chronological), extracts sign windows with jitter and idle
windows with overlap gating, applies frame-rate augmentation, and saves train/val/test
arrays and metadata.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import config as C
from .video_features import extract_video, segments
from .windowing import sample_window

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def parse_label(path: Path, base_dir: Path) -> str:
    rel = path.relative_to(base_dir)
    if len(rel.parts) > 1:
        return rel.parts[0].lower()
    import re
    stem = path.stem.lower()
    return re.sub(r"[_\-\s]*\d+$", "", stem)


def find_dataset_videos(video_dir: Path) -> Dict[str, List[Path]]:
    grouped = defaultdict(list)
    if not video_dir.exists():
        return grouped
    for p in sorted(video_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            label = parse_label(p, video_dir)
            grouped[label].append(p)
    return dict(grouped)


def compute_overlap(w_start: float, w_end: float, s_start: float, s_end: float) -> float:
    """Calculate overlap fraction of segment S covered by window W."""
    s_dur = s_end - s_start
    if s_dur <= 0:
        return 0.0
    inter = max(0.0, min(w_end, s_end) - max(w_start, s_start))
    return inter / s_dur


def thin_frames(frames: np.ndarray, ts_ms: np.ndarray, target_fps: float,
                rng: np.random.Generator, jitter_ms: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
    """Frame-rate augmentation: thin source frames and jitter timestamps."""
    if len(frames) == 0 or target_fps <= 0:
        return frames, ts_ms
    step_ms = 1000.0 / target_fps
    grid = np.arange(ts_ms[0], ts_ms[-1] + 1e-3, step_ms)
    # Causal nearest index in ts_ms
    idx = np.searchsorted(ts_ms, grid, side="right") - 1
    idx = np.clip(idx, 0, len(frames) - 1)
    # Remove duplicate consecutive indices
    unique_mask = np.concatenate([[True], idx[1:] != idx[:-1]])
    idx = idx[unique_mask]

    aug_frames = frames[idx].copy()
    jitter = rng.uniform(-jitter_ms, jitter_ms, size=len(idx))
    aug_ts = ts_ms[idx].astype(np.float64) + jitter

    # Enforce strictly non-decreasing
    for i in range(1, len(aug_ts)):
        if aug_ts[i] <= aug_ts[i - 1]:
            aug_ts[i] = aug_ts[i - 1] + 1.0

    return aug_frames, aug_ts


def split_videos(
    video_groups: Dict[str, List[Path]],
    holdout_dir: Path | None = None,
    seed: int = 0,
) -> Tuple[Dict[str, Dict[str, List[Path]]], Dict[str, str]]:
    """Split videos into train/val/test per label.

    Returns:
      splits: {label: {"train": [...], "val": [...], "test": [...]}}
      strategies: {label: "video_split" | "chronological" | "holdout_test"}
    """
    rng = np.random.default_rng(seed)
    splits = {}
    strategies = {}

    holdout_videos = find_dataset_videos(holdout_dir) if holdout_dir and holdout_dir.exists() else {}

    for label, paths in sorted(video_groups.items()):
        paths = list(paths)
        if holdout_videos.get(label):
            # Dedicated test set exists
            rng.shuffle(paths)
            n = len(paths)
            n_val = max(1, int(round(0.20 * n)))
            splits[label] = {
                "train": paths[n_val:],
                "val": paths[:n_val],
                "test": holdout_videos[label],
            }
            strategies[label] = "holdout_test"
        elif len(paths) >= 4:
            # Video-level split (70 / 15 / 15)
            rng.shuffle(paths)
            n = len(paths)
            n_test = max(1, int(round(0.15 * n)))
            n_val = max(1, int(round(0.15 * n)))
            n_train = n - n_val - n_test
            if n_train <= 0:
                n_train = 1
                n_val = max(1, (n - 1) // 2)
                n_test = n - n_train - n_val

            train_p = paths[:n_train]
            val_p = paths[n_train : n_train + n_val]
            test_p = paths[n_train + n_val :]
            splits[label] = {"train": train_p, "val": val_p, "test": test_p}
            strategies[label] = "video_split"
        else:
            # Chronological split for 1-3 videos
            splits[label] = {"train": paths, "val": paths, "test": paths}
            strategies[label] = "chronological"

    return splits, strategies


def build_dataset(
    video_dir: Path = C.CUSTOM_VIDEO_DIR,
    holdout_dir: Path = C.HOLDOUT_VIDEO_DIR,
    out_dir: Path = C.CUSTOM_PROCESSED,
    window_s: float = C.WINDOW_S,
    include_wlasl: bool = False,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    video_groups = find_dataset_videos(video_dir)

    if not video_groups:
        raise ValueError(f"No videos found in {video_dir}")

    # Discover labels (sign labels + idle)
    sign_labels = [lbl for lbl in sorted(video_groups.keys()) if lbl != C.IDLE_LABEL]
    labels = sorted(sign_labels + [C.IDLE_LABEL])
    label_to_idx = {l: i for i, l in enumerate(labels)}

    splits, strategies = split_videos(video_groups, holdout_dir=holdout_dir, seed=seed)

    # Validate split isolation for video_split strategy
    for label, s_dict in splits.items():
        if strategies[label] == "video_split":
            tr_names = {p.name for p in s_dict["train"]}
            va_names = {p.name for p in s_dict["val"]}
            te_names = {p.name for p in s_dict["test"]}
            assert tr_names.isdisjoint(va_names), f"Leakage between train and val in {label}"
            assert tr_names.isdisjoint(te_names), f"Leakage between train and test in {label}"
            assert va_names.isdisjoint(te_names), f"Leakage between val and test in {label}"

    print("\nDataset split strategies:")
    for label, strat in sorted(strategies.items()):
        msg = f"  - {label:<12s}: {strat} ({len(video_groups[label])} source videos)"
        if strat == "chronological":
            msg += " [WARN: same-session split is optimistic]"
        print(msg)

    X_splits = {"train": [], "val": [], "test": []}
    y_splits = {"train": [], "val": [], "test": []}

    # For 15 fps evaluation
    X_val_15, y_val_15 = [], []
    X_test_15, y_test_15 = [], []

    train_sign_counts = defaultdict(int)

    # Process each label
    for label in sorted(video_groups.keys()):
        strat = strategies[label]
        target_label_idx = label_to_idx[label]

        for split_name in ("train", "val", "test"):
            paths = splits[label][split_name]

            for path in paths:
                frames, ts_ms, fps = extract_video(path, cache=True)
                if len(frames) == 0:
                    continue

                t_min = ts_ms[0]
                t_max = ts_ms[-1]
                t_dur = t_max - t_min

                segs = segments(frames, ts_ms, fps=fps)

                # --- 1. IDLE WINDOWS ---
                # Slide with 300ms stride over the video
                stride_ms = 300.0
                curr_tend = t_min + 300.0
                while curr_tend <= t_max:
                    w_start = curr_tend - window_s * 1000.0
                    w_end = curr_tend
                    max_ov = max([compute_overlap(w_start, w_end, s0, s1) for s0, s1 in segs], default=0.0)

                    # Clean idle window: overlap < 10%
                    if max_ov < 0.10:
                        is_split_match = False
                        if strat == "chronological":
                            b1 = t_min + 0.70 * t_dur
                            b2 = t_min + 0.85 * t_dur
                            if split_name == "train" and curr_tend < b1:
                                is_split_match = True
                            elif split_name == "val" and b1 <= curr_tend < b2:
                                is_split_match = True
                            elif split_name == "test" and curr_tend >= b2:
                                is_split_match = True
                        else:
                            is_split_match = True

                        if is_split_match:
                            w = sample_window(frames, ts_ms, curr_tend, window_s=window_s, n=C.SEQ_LEN)
                            X_splits[split_name].append(w)
                            y_splits[split_name].append(label_to_idx[C.IDLE_LABEL])

                            if split_name != "train":
                                f_15, ts_15 = thin_frames(frames, ts_ms, 15.0, rng, jitter_ms=0.0)
                                w_15 = sample_window(f_15, ts_15, curr_tend, window_s=window_s, n=C.SEQ_LEN)
                                if split_name == "val":
                                    X_val_15.append(w_15)
                                    y_val_15.append(label_to_idx[C.IDLE_LABEL])
                                else:
                                    X_test_15.append(w_15)
                                    y_test_15.append(label_to_idx[C.IDLE_LABEL])

                    curr_tend += stride_ms

                # --- 2. SIGN WINDOWS ---
                for s_start, s_end in segs:
                    s_dur = s_end - s_start
                    # Generate t_end candidates with >= 70% overlap of S
                    t_end_min = s_start + 0.70 * s_dur
                    t_end_max = min(t_max, s_end - 0.70 * s_dur + window_s * 1000.0)
                    if t_end_max < t_end_min:
                        t_end_min = s_end
                        t_end_max = min(t_max, s_end + max(0.0, window_s * 1000.0 - s_dur))

                    if strat == "chronological":
                        b1 = t_min + 0.70 * t_dur
                        b2 = t_min + 0.85 * t_dur
                        gap = 1000.0 if t_dur > 8000.0 else max(50.0, 0.05 * t_dur)

                        if split_name == "train":
                            r_min = t_end_min
                            r_max = max(r_min, min(t_end_max, b1 - gap / 2.0))
                            t_ends = rng.uniform(r_min, r_max, size=6)
                        elif split_name == "val":
                            r_min = min(t_end_max, max(t_end_min, b1 + gap / 2.0))
                            r_max = max(r_min, min(t_end_max, b2 - gap / 2.0))
                            t_ends = np.linspace(r_min, r_max, 3)
                        else:  # test
                            r_min = min(t_end_max, max(t_end_min, b2 + gap / 2.0))
                            r_max = t_end_max
                            t_ends = np.linspace(r_min, r_max, 3)
                    else:
                        if split_name == "train":
                            t_ends = rng.uniform(t_end_min, t_end_max, size=6)
                        else:
                            t_ends = np.linspace(t_end_min, t_end_max, 3)

                    for te in t_ends:
                        if split_name == "train":
                            # Frame-rate augmentation (p=0.5)
                            if rng.random() < 0.5:
                                rand_fps = float(rng.uniform(10.0, 30.0))
                                aug_f, aug_ts = thin_frames(frames, ts_ms, rand_fps, rng, jitter_ms=15.0)
                                w = sample_window(aug_f, aug_ts, te, window_s=window_s, n=C.SEQ_LEN)
                            else:
                                w = sample_window(frames, ts_ms, te, window_s=window_s, n=C.SEQ_LEN)
                            X_splits["train"].append(w)
                            y_splits["train"].append(target_label_idx)
                            train_sign_counts[label] += 1
                        else:
                            w = sample_window(frames, ts_ms, te, window_s=window_s, n=C.SEQ_LEN)
                            X_splits[split_name].append(w)
                            y_splits[split_name].append(target_label_idx)

                            # 15 fps simulated evaluation
                            f_15, ts_15 = thin_frames(frames, ts_ms, 15.0, rng, jitter_ms=0.0)
                            w_15 = sample_window(f_15, ts_15, te, window_s=window_s, n=C.SEQ_LEN)
                            if split_name == "val":
                                X_val_15.append(w_15)
                                y_val_15.append(target_label_idx)
                            else:
                                X_test_15.append(w_15)
                                y_test_15.append(target_label_idx)

    # Optional WLASL merge (train only)
    if include_wlasl and C.WLASL_JSON.exists():
        print("Merging WLASL videos into train split...")
        from .dataset import _select_instances, _video_path
        selected = _select_instances()
        for gloss, insts in selected.items():
            if gloss in label_to_idx:
                g_idx = label_to_idx[gloss]
                for inst in insts:
                    vp = _video_path(inst["video_id"])
                    if vp:
                        w_frames, w_ts, w_fps = extract_video(vp, cache=True)
                        if len(w_frames) > 0:
                            # Use middle window
                            te = w_ts[-1]
                            w = sample_window(w_frames, w_ts, te, window_s=window_s, n=C.SEQ_LEN)
                            X_splits["train"].append(w)
                            y_splits["train"].append(g_idx)
                            train_sign_counts[gloss] += 1

    # Cap train idle windows at 1.5x mean per-sign train count
    mean_sign_train = np.mean(list(train_sign_counts.values())) if train_sign_counts else 30.0
    max_train_idle = max(10, int(round(1.5 * mean_sign_train)))

    tr_X = np.array(X_splits["train"])
    tr_y = np.array(y_splits["train"])
    idle_mask = (tr_y == label_to_idx[C.IDLE_LABEL])
    sign_mask = ~idle_mask

    idle_indices = np.where(idle_mask)[0]
    if len(idle_indices) > max_train_idle:
        keep_idle_idx = rng.choice(idle_indices, size=max_train_idle, replace=False)
        keep_all_idx = np.concatenate([np.where(sign_mask)[0], keep_idle_idx])
        rng.shuffle(keep_all_idx)
        X_train = tr_X[keep_all_idx]
        y_train = tr_y[keep_all_idx]
    else:
        X_train = tr_X
        y_train = tr_y

    X_val = np.array(X_splits["val"], dtype=np.float32)
    y_val = np.array(y_splits["val"], dtype=np.int64)
    X_test = np.array(X_splits["test"], dtype=np.float32)
    y_test = np.array(y_splits["test"], dtype=np.int64)

    X_val_15 = np.array(X_val_15, dtype=np.float32)
    y_val_15 = np.array(y_val_15, dtype=np.int64)
    X_test_15 = np.array(X_test_15, dtype=np.float32)
    y_test_15 = np.array(y_test_15, dtype=np.int64)

    # Save to disk
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)
    np.save(out_dir / "X_val_15fps.npy", X_val_15)
    np.save(out_dir / "y_val_15fps.npy", y_val_15)
    np.save(out_dir / "X_test_15fps.npy", X_test_15)
    np.save(out_dir / "y_test_15fps.npy", y_test_15)

    # Metadata
    counts = {
        "train": {l: int(np.sum(y_train == i)) for l, i in label_to_idx.items()},
        "val": {l: int(np.sum(y_val == i)) for l, i in label_to_idx.items()},
        "test": {l: int(np.sum(y_test == i)) for l, i in label_to_idx.items()},
    }

    meta = {
        "labels": labels,
        "idle_label": C.IDLE_LABEL,
        "window_s": window_s,
        "seq_len": C.SEQ_LEN,
        "feature_dim": C.FEATURE_DIM,
        "counts": counts,
        "splits": {
            "strategy_per_label": strategies,
            "train_videos": {l: [p.name for p in s["train"]] for l, s in splits.items()},
            "val_videos": {l: [p.name for p in s["val"]] for l, s in splits.items()},
            "test_videos": {l: [p.name for p in s["test"]] for l, s in splits.items()},
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Print summary table
    print("\n================ DATASET BUILD SUMMARY ================")
    print(f"{'Class':<12s} {'Train':<8s} {'Val':<8s} {'Test':<8s} {'Total':<8s} {'Warnings'}")
    print("-" * 65)
    for l, i in label_to_idx.items():
        tr_c = counts["train"][l]
        va_c = counts["val"][l]
        te_c = counts["test"][l]
        tot = tr_c + va_c + te_c
        warns = []
        if l != C.IDLE_LABEL:
            if tr_c < 30:
                warns.append(f"<30 train ({tr_c})")
            n_vids = len(video_groups.get(l, []))
            if n_vids < 3:
                warns.append(f"<3 vids ({n_vids})")
        w_str = ", ".join(warns) if warns else "OK"
        print(f"{l:<12s} {tr_c:<8d} {va_c:<8d} {te_c:<8d} {tot:<8d} {w_str}")
    print("-" * 65)
    print(f"{'TOTAL':<12s} {len(X_train):<8d} {len(X_val):<8d} {len(X_test):<8d} {len(X_train)+len(X_val)+len(X_test):<8d}")
    print(f"\nArtifacts saved to {out_dir}/")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Build custom windowed dataset for SignSpeak")
    parser.add_argument("--include-wlasl", action="store_true", help="Merge matching WLASL clips into train split")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for splits and jitter")
    args = parser.parse_args()

    build_dataset(include_wlasl=args.include_wlasl, seed=args.seed)


if __name__ == "__main__":
    main()
