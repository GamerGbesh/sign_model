"""Inspect video dataset for ASL sign recognition.

Reports file count, duration, fps, resolution, pose and hand detection rates per file
and per label. Saves skeleton-overlay thumbnails to data/debug/ for visual inspection.
Optionally plots activity timelines when --plot is specified.
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from asl import config as C
from asl.landmarks import HolisticExtractor, arrays_from_result, draw_overlay

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def parse_label_from_path(path: Path, base_dir: Path) -> str:
    """Extract label from folder name or flat filename."""
    rel = path.relative_to(base_dir)
    if len(rel.parts) > 1:
        # e.g. videos/hello/123.mp4 -> hello
        return rel.parts[0].lower()
    # Flat file: e.g. videos/hello_1.mp4 or videos/thank_you-02.avi
    stem = path.stem.lower()
    # Strip trailing digits and separators (_ or -)
    label = re.sub(r"[_\-\s]*\d+$", "", stem)
    return label


def find_videos(video_dir: Path) -> dict[str, list[Path]]:
    """Group video paths by label."""
    grouped = defaultdict(list)
    if not video_dir.exists():
        return grouped

    for p in sorted(video_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            label = parse_label_from_path(p, video_dir)
            grouped[label].append(p)
    return dict(grouped)


def inspect_video(video_path: Path, extractor: HolisticExtractor):
    """Inspect a single video file, returning detection stats and sample frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120 or np.isnan(fps):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    pose_detected = 0
    hand_detected = 0
    last_ts_ms = -1

    annotated_frames = []
    frames_features = []
    timestamps_ms = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Compute strictly increasing integer ms timestamp
        raw_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if raw_ts > last_ts_ms and not np.isnan(raw_ts):
            ts_ms = int(round(raw_ts))
        else:
            ts_ms = int(round(frame_idx * 1000.0 / fps))

        if ts_ms <= last_ts_ms:
            ts_ms = last_ts_ms + max(1, int(round(1000.0 / fps)))
        last_ts_ms = ts_ms
        timestamps_ms.append(ts_ms)

        vec, result, pose_present = extractor(frame, timestamp_ms=ts_ms)
        frames_features.append(vec)

        if pose_present:
            pose_detected += 1

        _, _, _, _, lh_present, rh_present = arrays_from_result(result)
        has_hand = lh_present or rh_present
        if has_hand:
            hand_detected += 1

        # Keep candidates for thumbnail (draw skeleton overlay)
        if pose_present and has_hand:
            overlay_frame = draw_overlay(frame.copy(), result)
            annotated_frames.append((frame_idx, overlay_frame))
        elif pose_present and len(annotated_frames) < 3:
            overlay_frame = draw_overlay(frame.copy(), result)
            annotated_frames.append((frame_idx, overlay_frame))

        frame_idx += 1

    cap.release()

    duration_s = frame_idx / fps if fps > 0 else 0.0
    pose_rate = pose_detected / frame_idx if frame_idx > 0 else 0.0
    hand_rate = hand_detected / frame_idx if frame_idx > 0 else 0.0

    return {
        "path": video_path,
        "frames": frame_idx,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": duration_s,
        "pose_detected": pose_detected,
        "hand_detected": hand_detected,
        "pose_rate": pose_rate,
        "hand_rate": hand_rate,
        "annotated_frames": annotated_frames,
        "features": np.stack(frames_features) if frames_features else np.zeros((0, C.FEATURE_DIM), dtype=np.float32),
        "timestamps_ms": np.array(timestamps_ms, dtype=np.int64),
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect video dataset for ASL recognition")
    parser.add_argument("--video-dir", type=Path, default=C.BASE_DIR / "videos")
    parser.add_argument("--debug-dir", type=Path, default=C.BASE_DIR / "data" / "debug")
    parser.add_argument("--plot", action="store_true", help="Plot timeline PNGs with activity and segments")
    args = parser.parse_args()

    video_groups = find_videos(args.video_dir)
    if not video_groups:
        print(f"No video files found in {args.video_dir}")
        return

    args.debug_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = args.debug_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {sum(len(v) for v in video_groups.values())} videos across {len(video_groups)} labels: {sorted(video_groups.keys())}\n")

    all_flagged = []
    label_summaries = {}

    all_segments_by_label = defaultdict(list)

    for label, paths in sorted(video_groups.items()):
            print(f"--- Label: {label} ({len(paths)} videos) ---")
            total_dur = 0.0
            resolutions = set()
            fps_list = []
            pose_rates = []
            hand_rates = []
            saved_thumbs = 0

            for p in paths:
                # Use cached extraction
                from asl import video_features
                frames, ts_ms, fps = video_features.extract_video(p)

                # Compute detection stats from extracted frames
                pose_present = np.any(frames[:, C.POSE_SLICE] != 0.0, axis=1)
                lh_present = np.any(frames[:, C.LH_SLICE] != 0.0, axis=1)
                rh_present = np.any(frames[:, C.RH_SLICE] != 0.0, axis=1)
                has_hand = lh_present | rh_present

                n_frames = len(frames)
                dur_s = (ts_ms[-1] - ts_ms[0]) / 1000.0 if n_frames > 1 else (n_frames / fps)
                pose_rate = np.mean(pose_present) if n_frames > 0 else 0.0
                hand_rate = np.mean(has_hand) if n_frames > 0 else 0.0

                cap = cv2.VideoCapture(str(p))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                total_dur += dur_s
                resolutions.add(f"{w}x{h}")
                fps_list.append(fps)
                pose_rates.append(pose_rate)
                hand_rates.append(hand_rate)

                # Compute segments
                segs = video_features.segments(frames, ts_ms, fps=fps)
                for s_start, s_end in segs:
                    seg_dur = (s_end - s_start) / 1000.0
                    all_segments_by_label[label].append(seg_dur)

                flag = ""
                if hand_rate < 0.70:
                    flag = " [FLAG: hand < 70%]"
                    all_flagged.append((label, p.name, hand_rate))

                print(f"  {p.name:<15s} {dur_s:4.1f}s | {w}x{h} @ {fps:4.1f}fps | "
                      f"pose: {pose_rate:4.0%} | hand: {hand_rate:4.0%} | segs: {len(segs)}{flag}")

                # Save thumbnails if needed
                if saved_thumbs < 3:
                    cap = cv2.VideoCapture(str(p))
                    cands = []
                    f_idx = 0
                    with HolisticExtractor(running_mode="VIDEO") as ext:
                        while len(cands) < 3:
                            ok, fr = cap.read()
                            if not ok:
                                break
                            if f_idx < len(ts_ms):
                                t_ms = int(ts_ms[f_idx])
                                _, res, pp = ext(fr, timestamp_ms=t_ms)
                                if pp:
                                    cands.append((f_idx, draw_overlay(fr, res)))
                            f_idx += 1
                    cap.release()
                    for f_no, img in cands:
                        if saved_thumbs < 3:
                            thumb_name = f"{label}_thumb_{saved_thumbs + 1}_{p.stem}_f{f_no}.jpg"
                            cv2.imwrite(str(thumbs_dir / thumb_name), img)
                            saved_thumbs += 1

                if args.plot:
                    act = video_features.activity(frames, fps=fps)
                    plot_path = args.debug_dir / f"{label}_{p.stem}_timeline.png"
                    _save_timeline_plot(ts_ms, act, segs, label, p.stem, plot_path)

            avg_pose = np.mean(pose_rates) if pose_rates else 0.0
            avg_hand = np.mean(hand_rates) if hand_rates else 0.0
            mean_fps = np.mean(fps_list) if fps_list else 0.0
            res_str = ", ".join(sorted(resolutions))

            label_summaries[label] = {
                "count": len(paths),
                "duration_s": total_dur,
                "fps": mean_fps,
                "resolutions": res_str,
                "pose_rate": avg_pose,
                "hand_rate": avg_hand,
            }

            print(f"  Summary: {len(paths)} files, {total_dur:.1f}s total | Res: {res_str} | FPS: {mean_fps:.1f} | "
                  f"Avg Pose: {avg_pose:.0%} | Avg Hand: {avg_hand:.0%}\n")

    print("\n================ DATASET INSPECTION SUMMARY ================")
    print(f"{'Label':<12s} {'Count':<6s} {'Total (s)':<10s} {'FPS':<6s} {'Pose %':<8s} {'Hand %':<8s} {'Resolutions'}")
    print("-" * 75)
    for label, s in sorted(label_summaries.items()):
        print(f"{label:<12s} {s['count']:<6d} {s['duration_s']:<10.1f} {s['fps']:<6.1f} {s['pose_rate']:<8.0%} {s['hand_rate']:<8.0%} {s['resolutions']}")

    # Segment duration percentiles
    print("\n================ SIGN SEGMENT DURATION PERCENTILES ================")
    print(f"{'Label':<12s} {'Segments':<10s} {'p50 (s)':<8s} {'p75 (s)':<8s} {'p90 (s)':<8s} {'p95 (s)':<8s} {'Max (s)'}")
    print("-" * 75)
    all_durs = []
    for label, durs in sorted(all_segments_by_label.items()):
        all_durs.extend(durs)
        if durs:
            p50, p75, p90, p95, mx = np.percentile(durs, [50, 75, 90, 95, 100])
            print(f"{label:<12s} {len(durs):<10d} {p50:<8.2f} {p75:<8.2f} {p90:<8.2f} {p95:<8.2f} {mx:<8.2f}")
        else:
            print(f"{label:<12s} {0:<10d} {'-':<8s} {'-':<8s} {'-':<8s} {'-':<8s} {'-'}")

    if all_durs:
        overall_p50, overall_p75, overall_p90, overall_p95, overall_max = np.percentile(all_durs, [50, 75, 90, 95, 100])
        print("-" * 75)
        print(f"{'OVERALL':<12s} {len(all_durs):<10d} {overall_p50:<8.2f} {overall_p75:<8.2f} {overall_p90:<8.2f} {overall_p95:<8.2f} {overall_max:<8.2f}")
        calc_window_s = float(np.clip(overall_p90 + 0.4, 1.5, 3.0))
        print(f"\nCalculated WINDOW_S = clip(p90 + 0.4, 1.5, 3.0) = clip({overall_p90:.2f} + 0.4, 1.5, 3.0) = {calc_window_s:.2f}s")
        # Check segments > 1.5x WINDOW_S
        long_segs = [d for d in all_durs if d > 1.5 * calc_window_s]
        print(f"Segments longer than 1.5 * WINDOW_S ({1.5 * calc_window_s:.2f}s): {len(long_segs)} of {len(all_durs)}")

    if all_flagged:
        print(f"\n[!] Flagged {len(all_flagged)} files with <70% hand detection across full clip:")
        for lbl, fn, hr in all_flagged:
            print(f"    - {lbl}/{fn}: {hr:.1%} hand detection")
    else:
        print("\nAll files met the >=70% hand detection threshold.")
    print(f"\nThumbnails saved to: {thumbs_dir}")


def _save_timeline_plot(timestamps_ms, activity, segments, label, stem, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_sec = timestamps_ms / 1000.0
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_sec, activity, label="Activity (wrist above hip & hand present)", color="#1f77b4", lw=1.5)
    ax.fill_between(t_sec, 0, activity, alpha=0.2, color="#1f77b4")

    for start_ms, end_ms in segments:
        ax.axvspan(start_ms / 1000.0, end_ms / 1000.0, color="#2ca02c", alpha=0.3, label="Segment")

    # Remove duplicate labels
    handles, labels_list = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_list, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Active")
    ax.set_title(f"{label} - {stem}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
