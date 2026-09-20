"""FastAPI WebSocket server for real-time SignSpeak recognition.

Exposes REST endpoints (/health, /meta, /demo) and a high-performance binary
WebSocket endpoint (/ws/recognize) with single-slot frame queue, backpressure handling,
session limiting, origin validation, and strictly in-memory processing.
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config as C
from .model import SignLSTM, load_model
from .realtime import StreamingRecognizer

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_FRAME_BYTES = 512 * 1024  # 512 KB
SESSION_TIMEOUT_S = 60.0
DEFAULT_MAX_SESSIONS = 4

_shared_model: Optional[SignLSTM] = None
_labels: list[str] = []
_model_meta: dict = {}
_active_sessions: Set[int] = set()


def get_shared_model():
    global _shared_model, _labels, _model_meta
    if _shared_model is None:
        meta_path = C.MODELS_DIR / "model_meta.json"
        if meta_path.exists():
            _model_meta = json.loads(meta_path.read_text())
        _shared_model, _labels = load_model(
            weights_path=C.MODEL_WEIGHTS, labels_path=C.LABELS_JSON, device="cpu"
        )
    return _shared_model, _labels, _model_meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    if C.MODEL_WEIGHTS.exists() and C.LABELS_JSON.exists():
        get_shared_model()
    yield


app = FastAPI(title="SignSpeak Real-Time API", lifespan=lifespan)

# Configure CORS
dev_mode = os.environ.get("DEV", "0") == "1"
if dev_mode:
    allowed_origins = ["*"]
else:
    origins_env = os.environ.get("ALLOWED_ORIGINS", "")
    if origins_env:
        allowed_origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    else:
        allowed_origins = [
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    _, labels, meta = get_shared_model()
    return {
        "status": "ok",
        "labels": labels,
        "window_s": meta.get("window_s", C.WINDOW_S),
    }


@app.get("/meta")
def meta():
    _, _, m = get_shared_model()
    # Return public parts of model_meta.json
    public_fields = [
        "labels", "idle_label", "seq_len", "feature_dim", "window_s",
        "infer_stride_ms", "conf_threshold", "debounce_n", "metrics", "created_at"
    ]
    return {k: m[k] for k in public_fields if k in m}


@app.get("/demo")
def demo():
    demo_file = STATIC_DIR / "live.html"
    if not demo_file.exists():
        raise HTTPException(status_code=404, detail="Demo page not found")
    return FileResponse(demo_file)


# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    # Origin verification
    origin = websocket.headers.get("origin")
    if not dev_mode and "*" not in allowed_origins and origin:
        if origin not in allowed_origins:
            await websocket.close(code=1008, reason="Origin not allowed")
            return

    # Session limit check
    max_sessions = int(os.environ.get("MAX_SESSIONS", DEFAULT_MAX_SESSIONS))
    if len(_active_sessions) >= max_sessions:
        await websocket.close(code=1013, reason="Max concurrent sessions reached")
        return

    session_id = id(websocket)
    _active_sessions.add(session_id)
    await websocket.accept()

    shared_model, labels, meta_data = get_shared_model()
    recognizer = StreamingRecognizer(
        model=shared_model,
        labels=labels,
        window_s=meta_data.get("window_s", C.WINDOW_S),
        conf_threshold=meta_data.get("conf_threshold", 0.40),
        debounce_n=meta_data.get("debounce_n", 4),
        infer_stride_ms=meta_data.get("infer_stride_ms", C.INFER_STRIDE_MS),
    )

    # Initial ready handshake
    await websocket.send_json({
        "type": "ready",
        "labels": labels,
        "window_s": recognizer.window_s,
        "recommended_fps": 15,
        "max_width": 480,
    })

    # Single-slot backpressure queue: keeps only the newest unconsumed frame
    latest_frame: Optional[Tuple[float, bytes]] = None
    frame_ready = asyncio.Event()
    stop_event = asyncio.Event()

    async def reader_task():
        nonlocal latest_frame
        try:
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(websocket.receive(), timeout=SESSION_TIMEOUT_S)
                except asyncio.TimeoutError:
                    await websocket.close(code=1000, reason="Session idle timeout")
                    break

                if "text" in msg and msg["text"]:
                    try:
                        data = json.loads(msg["text"])
                        if data.get("cmd") == "reset":
                            recognizer.reset()
                            await websocket.send_json({"type": "reset", "phrase": []})
                    except Exception:
                        pass
                elif "bytes" in msg and msg["bytes"]:
                    bdata = msg["bytes"]
                    if len(bdata) > MAX_FRAME_BYTES:
                        await websocket.close(code=1009, reason="Frame exceeds 512KB limit")
                        break
                    if len(bdata) >= 8:
                        # Big-endian float64 timestamp in ms
                        t_ms = struct.unpack(">d", bdata[:8])[0]
                        jpeg_bytes = bdata[8:]
                        # Single-slot queue: overwrite with newest frame
                        latest_frame = (t_ms, jpeg_bytes)
                        frame_ready.set()
                elif msg.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            stop_event.set()
            frame_ready.set()

    async def worker_task():
        nonlocal latest_frame
        last_sent_ms = 0.0
        try:
            while not stop_event.is_set():
                await frame_ready.wait()
                frame_ready.clear()
                if stop_event.is_set():
                    break

                item = latest_frame
                if item is None:
                    continue

                t_ms, jpeg_bytes = item
                # Decode JPEG in thread pool (never write to disk - privacy strictly preserved)
                def decode_and_push():
                    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        return None
                    return recognizer.push(frame_bgr, int(round(t_ms)))

                res = await asyncio.to_thread(decode_and_push)
                if res is None:
                    continue

                now_ms = time.time() * 1000.0
                is_commit = res["commit"] is not None
                time_to_send = (now_ms - last_sent_ms) >= 100.0

                # Send immediately on commit, or at most every 100 ms
                if is_commit or time_to_send:
                    last_sent_ms = now_ms
                    payload = {
                        "type": "prediction",
                        "server_ms": now_ms,
                        **res,
                    }
                    await websocket.send_json(payload)
        except Exception:
            pass
        finally:
            stop_event.set()

    reader = asyncio.create_task(reader_task())
    worker = asyncio.create_task(worker_task())

    try:
        await asyncio.gather(reader, worker, return_exceptions=True)
    finally:
        _active_sessions.discard(session_id)
        recognizer.close()
