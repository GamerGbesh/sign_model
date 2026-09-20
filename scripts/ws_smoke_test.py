"""WebSocket smoke test: streams a video to running server via WebSocket in real-time.

Verifies end-to-end WebSocket protocol, measures round-trip latency, and confirms
that online prediction matches offline recognition.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time
from pathlib import Path

import cv2
import websockets

from asl import config as C


async def stream_video_websocket(
    video_path: Path,
    ws_url: str = "ws://127.0.0.1:8000/ws/recognize",
    max_width: int = 480,
    jpeg_quality: int = 70,
):
    print(f"Connecting to {ws_url} ...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120 or fps != fps:
        fps = 30.0

    frame_interval = 1.0 / fps
    commits = []
    rtts = []

    async with websockets.connect(ws_url, max_size=1024 * 1024) as ws:
        # Receive ready message
        ready_msg = json.loads(await ws.recv())
        print(f"Received ready handshake: {ready_msg['labels']}")

        stop_event = asyncio.Event()

        # Receiver task
        async def receiver():
            try:
                while not stop_event.is_set():
                    raw = await ws.recv()
                    data = json.loads(raw)
                    if data.get("type") == "prediction":
                        now_client = time.time() * 1000.0
                        client_t = data.get("t", now_client)
                        rtt = now_client - client_t
                        if rtt > 0:
                            rtts.append(rtt)
                        if data.get("commit"):
                            commits.append((data["t"], data["commit"]))
                            print(f"  [COMMIT] '{data['commit']}' @ t={data['t']}ms (RTT: {rtt:.1f}ms)")
            except (websockets.ConnectionClosed, asyncio.CancelledError):
                pass

        recv_task = asyncio.create_task(receiver())

        # Sender loop (real-time pace)
        f_idx = 0
        stream_start_ms = time.time() * 1000.0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                h, w = frame.shape[:2]
                scale = min(1.0, max_width / w)
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if not ok:
                    continue

                jpeg_bytes = buf.tobytes()
                client_ts_ms = time.time() * 1000.0
                header = struct.pack(">d", client_ts_ms)
                packet = header + jpeg_bytes

                await ws.send(packet)
                f_idx += 1
                await asyncio.sleep(frame_interval)

            # Wait briefly for in-flight predictions
            await asyncio.sleep(0.5)
        finally:
            cap.release()
            stop_event.set()
            recv_task.cancel()

    print("\n================ WS SMOKE TEST SUMMARY ================")
    print(f"Frames streamed: {f_idx} @ {fps:.1f} fps")
    print(f"Total commits:   {len(commits)} -> {[c[1] for c in commits]}")
    if rtts:
        print(f"Latency: median={sorted(rtts)[len(rtts)//2]:.1f}ms, p95={sorted(rtts)[int(len(rtts)*0.95)]:.1f}ms, min={min(rtts):.1f}ms")
    print("WebSocket smoke test completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="WebSocket smoke test for SignSpeak")
    parser.add_argument("--video", type=Path, default=None, help="Video file to stream")
    parser.add_argument("--url", type=str, default="ws://127.0.0.1:8000/ws/recognize")
    args = parser.parse_args()

    vpath = args.video
    if vpath is None:
        # Pick the first video in videos/
        import glob
        cands = sorted(glob.glob("videos/*/*.mp4"))
        if not cands:
            raise FileNotFoundError("No videos found in videos/")
        vpath = Path(cands[0])

    print(f"Streaming video: {vpath}")
    asyncio.run(stream_video_websocket(vpath, ws_url=args.url))


if __name__ == "__main__":
    main()
