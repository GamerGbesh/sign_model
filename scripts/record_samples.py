"""Webcam clip recorder for custom sign vocabulary.

Records clips to videos/<label>/<label>_<timestamp>.mp4.
Shows a 3-2-1 countdown, mirrored preview for display only, and saves
raw, unmirrored frames so ingestion is identical to the pipeline.
"""
from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import cv2

from asl import config as C


def record_clip(
    cap: cv2.VideoCapture,
    label: str,
    duration_s: float,
    output_dir: Path,
    countdown_s: int = 3,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120 or fps != fps:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Countdown
    start_t = time.time()
    while True:
        elapsed = time.time() - start_t
        remain = countdown_s - elapsed
        if remain <= 0:
            break
        ok, frame = cap.read()
        if not ok:
            break
        preview = cv2.flip(frame, 1)
        txt = f"Get ready: {int(remain) + 1}"
        cv2.putText(preview, txt, (width // 4, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
        cv2.imshow("SignSpeak Recorder (q to cancel)", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    # Recording
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"{label}_{ts_str}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    rec_start = time.time()
    frames_recorded = 0
    while True:
        rec_elapsed = time.time() - rec_start
        if rec_elapsed >= duration_s:
            break
        ok, frame = cap.read()
        if not ok:
            break

        # Save RAW unmirrored frame
        writer.write(frame)
        frames_recorded += 1

        # Mirrored preview for display
        preview = cv2.flip(frame, 1)
        remain_s = max(0.0, duration_s - rec_elapsed)
        cv2.putText(preview, f"RECORDING: {label.upper()} ({remain_s:.1f}s)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("SignSpeak Recorder (q to cancel)", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            writer.release()
            out_file.unlink(missing_ok=True)
            return False

    writer.release()
    print(f"Recorded {frames_recorded} frames ({duration_s:.1f}s) -> {out_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Record webcam clips for ASL recognition")
    parser.add_argument("--label", required=True, help="Word to sign (or 'idle')")
    parser.add_argument("--n", type=int, default=20, help="Number of repetitions to record")
    parser.add_argument("--duration", type=float, default=2.5, help="Clip duration in seconds")
    parser.add_argument("--video-dir", type=Path, default=C.CUSTOM_VIDEO_DIR)
    args = parser.parse_args()

    label = args.label.strip().lower()
    target_dir = args.video_dir / label

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"\nRecording {args.n} clips for '{label}' ({args.duration}s each).")
    print("Press Space to record next clip, or 'q' to quit.")

    count = 0
    try:
        while count < args.n:
            ok, frame = cap.read()
            if not ok:
                break
            preview = cv2.flip(frame, 1)
            cv2.putText(preview, f"Clip {count + 1}/{args.n}: '{label}' - Press SPACE to record", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("SignSpeak Recorder", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                success = record_clip(cap, label, args.duration, target_dir)
                if success:
                    count += 1
            elif key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
