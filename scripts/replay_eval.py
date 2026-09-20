"""Replay evaluation: evaluates StreamingRecognizer on test videos as if live.

Feeds test video frames sequentially with real-time timestamps, measures word recall,
commit precision, false commits per minute, and median sign-onset-to-commit latency.
Saves results to models/report/replay.json.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from asl import config as C
from asl.dataset_custom import thin_frames
from asl.realtime import StreamingRecognizer
from asl.video_features import extract_video, segments


def run_replay_eval(
    meta_path: Path = C.CUSTOM_PROCESSED / "meta.json",
    video_dir: Path = C.CUSTOM_VIDEO_DIR,
    target_fps: float | None = None,
    out_path: Path = C.MODELS_DIR / "report" / "replay.json",
) -> dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path} - run asl.dataset_custom first.")

    meta = json.loads(meta_path.read_text())
    test_videos_by_label = meta["splits"]["test_videos"]

    print("================ RUNNING REPLAY EVALUATION ================")
    if target_fps:
        print(f"Frame rate: simulated {target_fps:.1f} fps")
    else:
        print("Frame rate: native video fps")

    recognizer = StreamingRecognizer(model_dir=C.MODELS_DIR, device="cpu")

    per_label_stats = defaultdict(lambda: {"total_segments": 0, "hits": 0, "errors": 0, "latencies_ms": []})
    total_idle_duration_s = 0.0
    total_idle_false_commits = 0
    total_all_commits = 0
    total_true_commits = 0

    rng = np.random.default_rng(42)

    for label, file_names in sorted(test_videos_by_label.items()):
        if label == C.IDLE_LABEL:
            continue

        for fname in file_names:
            vpath = video_dir / label / fname
            if not vpath.exists():
                vpath = video_dir / fname
            if not vpath.exists():
                continue

            cap = cv2.VideoCapture(str(vpath))
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 120 or np.isnan(fps):
                fps = 30.0

            raw_frames = []
            raw_ts = []
            f_idx = 0
            last_t = -1
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos > last_t and not np.isnan(pos):
                    t_ms = int(round(pos))
                else:
                    t_ms = int(round(f_idx * 1000.0 / fps))
                if t_ms <= last_t:
                    t_ms = last_t + max(1, int(round(1000.0 / fps)))
                last_t = t_ms
                raw_frames.append(fr)
                raw_ts.append(t_ms)
                f_idx += 1
            cap.release()

            if not raw_frames:
                continue

            frames_arr = np.stack(raw_frames)
            ts_arr = np.array(raw_ts, dtype=np.int64)

            # Get reference segments for this video
            cached_f, cached_t, cached_fps = extract_video(vpath, cache=True)
            video_segs = segments(cached_f, cached_t, fps=cached_fps)

            # Optional frame thinning
            if target_fps:
                step = 1000.0 / target_fps
                grid = np.arange(ts_arr[0], ts_arr[-1] + 1e-3, step)
                indices = np.searchsorted(ts_arr, grid, side="right") - 1
                indices = np.clip(indices, 0, len(raw_frames) - 1)
                unique_mask = np.concatenate([[True], indices[1:] != indices[:-1]])
                indices = indices[unique_mask]
                eval_frames = [raw_frames[i] for i in indices]
                eval_ts = ts_arr[indices]
            else:
                eval_frames = raw_frames
                eval_ts = ts_arr

            recognizer.reset()
            commits = []

            for fr, t_ms in zip(eval_frames, eval_ts):
                res = recognizer.push(fr, int(t_ms))
                if res["commit"]:
                    commits.append((t_ms, res["commit"]))
                    total_all_commits += 1

            # Total duration
            v_dur_s = (eval_ts[-1] - eval_ts[0]) / 1000.0 if len(eval_ts) > 1 else 0.0

            # Evaluate against segments
            matched_commits = set()
            for s_start, s_end in video_segs:
                per_label_stats[label]["total_segments"] += 1
                win_start = s_start - 750.0
                win_end = s_end + 750.0

                # Commits falling in segment window
                in_window = [(t, w, i) for i, (t, w) in enumerate(commits) if win_start <= t <= win_end]
                correct_in_win = [c for c in in_window if c[1].lower() == label.lower()]

                if correct_in_win:
                    per_label_stats[label]["hits"] += 1
                    total_true_commits += 1
                    first_hit_t = correct_in_win[0][0]
                    latency = max(0.0, first_hit_t - s_start)
                    per_label_stats[label]["latencies_ms"].append(latency)
                    matched_commits.add(correct_in_win[0][2])
                else:
                    per_label_stats[label]["errors"] += 1

            # Unmatched commits are false positives
            for i, (t, w) in enumerate(commits):
                if i not in matched_commits:
                    total_idle_false_commits += 1

            # Calculate idle duration (total video duration minus segment durations)
            seg_total_s = sum((s1 - s0) / 1000.0 for s0, s1 in video_segs)
            idle_s = max(0.0, v_dur_s - seg_total_s)
            total_idle_duration_s += idle_s

    # Aggregate metrics
    print("\n================ REPLAY EVALUATION SUMMARY ================")
    print(f"{'Label':<12s} {'Segments':<10s} {'Hits':<6s} {'Errors':<8s} {'Recall %':<10s} {'Median Latency (ms)'}")
    print("-" * 75)

    all_latencies = []
    total_segs = 0
    total_hits = 0

    per_label_report = {}
    for label in sorted(test_videos_by_label.keys()):
        if label == C.IDLE_LABEL:
            continue
        stats = per_label_stats[label]
        n_segs = stats["total_segments"]
        hits = stats["hits"]
        errs = stats["errors"]
        recall = hits / n_segs if n_segs > 0 else 0.0
        med_lat = float(np.median(stats["latencies_ms"])) if stats["latencies_ms"] else 0.0

        all_latencies.extend(stats["latencies_ms"])
        total_segs += n_segs
        total_hits += hits

        per_label_report[label] = {
            "segments": n_segs,
            "hits": hits,
            "errors": errs,
            "recall": recall,
            "median_latency_ms": med_lat,
        }
        print(f"{label:<12s} {n_segs:<10d} {hits:<6d} {errs:<8d} {recall:<10.1%} {med_lat:<10.0f}")

    overall_recall = total_hits / total_segs if total_segs > 0 else 0.0
    overall_precision = total_true_commits / total_all_commits if total_all_commits > 0 else 0.0
    median_latency_ms = float(np.median(all_latencies)) if all_latencies else 0.0
    false_commits_per_min = (total_idle_false_commits / (total_idle_duration_s / 60.0)) if total_idle_duration_s > 0 else 0.0

    print("-" * 75)
    print(f"{'OVERALL':<12s} {total_segs:<10d} {total_hits:<6d} {total_segs - total_hits:<8d} {overall_recall:<10.1%} {median_latency_ms:<10.0f}")
    print(f"\nOverall Commit Precision:    {overall_precision:.1%}")
    print(f"Overall Word Recall:         {overall_recall:.1%}")
    print(f"False Commits per Minute:    {false_commits_per_min:.2f} / min")
    print(f"Median Sign-to-Commit Delay: {median_latency_ms:.0f} ms ({median_latency_ms / 1000.0:.2f} s)")

    result = {
        "overall_recall": overall_recall,
        "commit_precision": overall_precision,
        "false_commits_per_min": false_commits_per_min,
        "median_latency_ms": median_latency_ms,
        "total_segments": total_segs,
        "total_hits": total_hits,
        "per_label": per_label_report,
        "target_fps": target_fps,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved replay results -> {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Replay evaluation of StreamingRecognizer on test split")
    parser.add_argument("--fps", type=float, default=None, help="Simulate specific FPS (e.g. 15)")
    args = parser.parse_args()

    run_replay_eval(target_fps=args.fps)


if __name__ == "__main__":
    main()
