"""FastAPI WebSocket server for real-time Amegbe recognition and Twi speech.

Exposes REST endpoints (/health, /meta, /demo, /audio/{key}.wav, /vocab, /vocab/reload,
/tts/voices, /vocab/{gloss}/audio) and a high-performance binary WebSocket endpoint
(/ws/recognize) with single-slot frame queue, backpressure handling, session limiting,
and SpeechOrchestrator integration.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
from .speech import SpeechOrchestrator
from .tts import TwiTTS
from .vocab import VocabularyStore

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_FRAME_BYTES = 512 * 1024  # 512 KB
SESSION_TIMEOUT_S = 60.0
DEFAULT_MAX_SESSIONS = 4
AUDIO_KEY_RE = re.compile(r"^[0-9a-f]{40}$")

_shared_model: Optional[SignLSTM] = None
_labels: list[str] = []
_model_meta: dict = {}
_active_sessions: Set[int] = set()

_shared_vocab_store: Optional[VocabularyStore] = None
_shared_tts: Optional[TwiTTS] = None


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


def get_vocab_store() -> VocabularyStore:
    global _shared_vocab_store
    if _shared_vocab_store is None:
        _shared_vocab_store = VocabularyStore()
    return _shared_vocab_store


def get_tts() -> TwiTTS:
    global _shared_tts
    if _shared_tts is None:
        _shared_tts = TwiTTS()
    return _shared_tts


@asynccontextmanager
async def lifespan(app: FastAPI):
    if C.MODEL_WEIGHTS.exists() and C.LABELS_JSON.exists():
        get_shared_model()
    get_vocab_store()
    yield
    if _shared_tts:
        _shared_tts.stop()


app = FastAPI(title="Amegbe Real-Time Sign & Speech API", lifespan=lifespan)

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
        "app": "amegbe",
        "labels": labels,
        "window_s": meta.get("window_s", C.WINDOW_S),
    }


@app.get("/meta")
def meta():
    _, _, m = get_shared_model()
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
    return FileResponse(str(demo_file))


@app.get("/audio/{key}.wav")
def get_audio(key: str):
    if not AUDIO_KEY_RE.match(key):
        raise HTTPException(status_code=400, detail="Invalid audio key format")
    audio_path = C.TTS_CACHE_DIR / f"{key}.wav"
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        str(audio_path),
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/vocab")
def get_vocab():
    snap = get_vocab_store().get_snapshot()
    return {
        "version": snap.version,
        "dialect": snap.dialect,
        "defaults": snap.defaults,
        "words": snap.enabled_words,
        "phrases": snap.enabled_phrases,
    }


@app.post("/vocab/reload")
def reload_vocab():
    store = get_vocab_store()
    ok, errors = store.reload(force=True)
    snap = store.get_snapshot()
    return {
        "ok": ok,
        "errors": errors,
        "count": len(snap.words),
    }


@app.get("/tts/voices")
def get_voices():
    return get_tts().get_voices()


@app.get("/vocab/{gloss}/audio")
def get_vocab_audio(gloss: str):
    store = get_vocab_store()
    snap = store.get_snapshot()
    entry = snap.words.get(gloss)
    if not entry or not entry.get("twi"):
        raise HTTPException(status_code=404, detail=f"No Twi vocabulary entry for gloss '{gloss}'")
    tts = get_tts()
    res = tts.speak(entry["twi"], voice=snap.default_voice, language="twi", audio_override=entry.get("audio"))
    if not res or not res.path.exists():
        raise HTTPException(status_code=500, detail="Audio generation failed")
    return FileResponse(str(res.path), media_type="audio/wav")


# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if not dev_mode and "*" not in allowed_origins and origin:
        if origin not in allowed_origins:
            await websocket.close(code=1008, reason="Origin not allowed")
            return

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
        conf_threshold=meta_data.get("conf_threshold", C.CONF_THRESHOLD),
        debounce_n=meta_data.get("debounce_n", 4),
        infer_stride_ms=meta_data.get("infer_stride_ms", C.INFER_STRIDE_MS),
    )

    orchestrator = SpeechOrchestrator(
        vocab_store=get_vocab_store(),
        tts=get_tts(),
        mode=C.SPEECH_MODE,
        voice=C.TTS_VOICE,
    )

    # Initial ready handshake
    await websocket.send_json({
        "type": "ready",
        "app": "amegbe",
        "labels": labels,
        "window_s": recognizer.window_s,
        "recommended_fps": 15,
        "max_width": 480,
        "speech_mode": orchestrator.mode,
        "voice": orchestrator.voice,
    })

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
                        cmd = data.get("cmd") or data.get("type")
                        if cmd == "reset":
                            recognizer.reset()
                            orchestrator.clear()
                            await websocket.send_json({"type": "reset", "phrase": []})
                        elif cmd == "config":
                            if "speech_mode" in data:
                                orchestrator.set_mode(data["speech_mode"])
                            if "voice" in data:
                                orchestrator.set_voice(data["voice"])
                            await websocket.send_json({
                                "type": "config_ack",
                                "speech_mode": orchestrator.mode,
                                "voice": orchestrator.voice,
                            })
                        elif cmd == "speak":
                            sp_events = await asyncio.to_thread(orchestrator.speak_now)
                            for ev in sp_events:
                                await websocket.send_json(ev)
                        elif cmd == "clear":
                            orchestrator.clear()
                            await websocket.send_json({"type": "clear_ack"})
                        elif cmd == "ack_audio":
                            if "key" in data:
                                orchestrator.ack_audio(data["key"])
                    except Exception as e:
                        logger.warning("Error processing text ws message: %s", e)
                elif "bytes" in msg and msg["bytes"]:
                    bdata = msg["bytes"]
                    if len(bdata) > MAX_FRAME_BYTES:
                        await websocket.close(code=1009, reason="Frame exceeds 512KB limit")
                        break
                    if len(bdata) >= 8:
                        t_ms = struct.unpack(">d", bdata[:8])[0]
                        jpeg_bytes = bdata[8:]
                        latest_frame = (t_ms, jpeg_bytes)
                        frame_ready.set()
                elif msg.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug("WS reader error: %s", e)
        finally:
            stop_event.set()
            frame_ready.set()

    async def worker_task():
        nonlocal latest_frame
        last_sent_ms = 0.0
        try:
            while not stop_event.is_set():
                try:
                    await asyncio.wait_for(frame_ready.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    # Timeout check: tick sentence orchestrator on idle pause
                    tick_events = await asyncio.to_thread(orchestrator.tick, time.time())
                    for ev in tick_events:
                        await websocket.send_json(ev)
                    continue

                frame_ready.clear()
                if stop_event.is_set():
                    break

                item = latest_frame
                if item is None:
                    continue

                t_ms, jpeg_bytes = item

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

                # Check sentence mode tick
                tick_events = await asyncio.to_thread(orchestrator.tick, now_ms / 1000.0)
                for ev in tick_events:
                    await websocket.send_json(ev)

                # Process commit speech
                if is_commit:
                    commit_events = await asyncio.to_thread(orchestrator.on_commit, res["commit"], now_ms / 1000.0)
                    for ev in commit_events:
                        await websocket.send_json(ev)

                if is_commit or time_to_send:
                    last_sent_ms = now_ms
                    phrase_data = res.get("phrase", [])
                    enriched_phrase = orchestrator.enrich_phrase_event({"words": phrase_data}) if phrase_data else None
                    payload = {
                        "type": "prediction",
                        "server_ms": now_ms,
                        **res,
                        "twi_phrase": enriched_phrase["twi"] if enriched_phrase else "",
                    }
                    await websocket.send_json(payload)
        except Exception as e:
            logger.debug("WS worker error: %s", e)
        finally:
            stop_event.set()

    reader = asyncio.create_task(reader_task())
    worker = asyncio.create_task(worker_task())

    try:
        await asyncio.gather(reader, worker, return_exceptions=True)
    finally:
        _active_sessions.discard(session_id)
        recognizer.close()
