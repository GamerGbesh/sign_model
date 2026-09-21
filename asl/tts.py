"""Text-to-Speech synthesis and audio caching for Amegbe.

Wraps ghananlpcommunity/stable-twi-tts (ONNX/Piper) with:
- Abstract SynthBackend protocol with PiperStableTwiBackend and FakeBackend
- SHA1 audio caching in data/tts_cache/
- Threaded worker with bounded queue (drops oldest on overflow)
- Support for custom audio overrides from vocabulary.json
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import queue
import tempfile
import threading
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from . import config as C

logger = logging.getLogger(__name__)

DEFAULT_MODEL_VERSION = "v0.1.0"


def get_cache_key(
    text: str,
    voice: str = "twi-6",
    language: str = "twi",
    model_version: str = DEFAULT_MODEL_VERSION,
) -> str:
    """Computes deterministic SHA1 cache key from normalized text, voice, language, and model."""
    norm_voice = "twi-6" if not voice or voice == "auto" else voice
    norm_text = unicodedata.normalize("NFC", text.strip())
    raw = f"{norm_text}|{norm_voice}|{language}|{model_version}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


@dataclass(frozen=True)
class AudioResult:
    key: str
    path: Path
    duration_s: float
    sample_rate: int
    voice: str
    cached: bool

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "path": str(self.path),
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "voice": self.voice,
            "cached": self.cached,
        }


@runtime_checkable
class SynthBackend(Protocol):
    def synthesize(self, text: str, voice: str, language: str) -> Tuple[bytes, int, float]:
        """Synthesizes text into audio. Returns (wav_bytes, sample_rate, duration_s)."""
        ...

    def get_voices(self) -> Dict[str, Any]:
        """Returns voices metadata."""
        ...


class FakeBackend:
    """In-memory fake synthesizer for fast, offline unit tests."""

    def __init__(self, sample_rate: int = 22050, duration_s: float = 0.5):
        self.sample_rate = sample_rate
        self.duration_s = duration_s
        self.calls: List[Tuple[str, str, str]] = []

    def synthesize(self, text: str, voice: str, language: str) -> Tuple[bytes, int, float]:
        self.calls.append((text, voice, language))
        num_frames = int(self.sample_rate * self.duration_s)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(b"\x00\x00" * num_frames)
        return buf.getvalue(), self.sample_rate, self.duration_s

    def get_voices(self) -> Dict[str, Any]:
        return {
            "tiers": {
                "twi_only": ["twi-6"],
                "codeswitch": ["twi-1"],
            },
            "voices": [
                {"name": "twi-6", "language": "twi", "twi_only_uer": 0.268},
                {"name": "twi-1", "language": "twi", "codeswitch_uer_avg": 0.598},
            ],
            "default": "twi-6",
        }


class PiperStableTwiBackend:
    """Real synthesis backend wrapping stable_twi_tts.StableTwiTTS."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or C.TTS_MODEL_DIR
        self._tts = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._tts is None:
            with self._lock:
                if self._tts is None:
                    from stable_twi_tts import StableTwiTTS
                    logger.info("Loading StableTwiTTS model...")
                    self._tts = StableTwiTTS.from_pretrained(
                        str(self.model_dir) if self.model_dir and self.model_dir.exists() else None
                    )

    def synthesize(self, text: str, voice: str, language: str) -> Tuple[bytes, int, float]:
        self._ensure_loaded()
        try:
            norm_text = unicodedata.normalize("NFC", text.strip())
            synthesis = self._tts.synthesize(norm_text, voice=voice, language=language)
            # Write out to in-memory WAV
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                synthesis.save(tmp_path)
                with open(tmp_path, "rb") as f:
                    wav_bytes = f.read()
                return wav_bytes, synthesis.sample_rate, synthesis.duration
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        except Exception as e:
            logger.warning("TTS synthesis failed for %r (voice=%s, lang=%s): %s", text, voice, language, e)
            return b"", 22050, 0.0

    def get_voices(self) -> Dict[str, Any]:
        self._ensure_loaded()
        voices_file = self._tts.dir / "voices.json"
        if voices_file.exists():
            try:
                return json.loads(voices_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Failed to parse voices.json: %s", e)
        return {
            "tiers": {"twi_only": ["twi-6"], "codeswitch": ["twi-1"]},
            "default": "twi-6",
        }


@dataclass
class _QueueItem:
    text: str
    voice: Optional[str]
    language: Optional[str]
    audio_override: Optional[str]
    callback: Optional[Callable[[Optional[AudioResult]], None]]
    event: Optional[threading.Event]
    result: Optional[AudioResult] = None


class TwiTTS:
    """Manages audio synthesis, caching, audio overrides, and background worker queue."""

    def __init__(
        self,
        backend: Optional[SynthBackend] = None,
        cache_dir: Path = C.TTS_CACHE_DIR,
        voice: Optional[str] = None,
        default_voice: Optional[str] = None,
        language: str = C.TTS_LANGUAGE,
        queue_max: int = C.WORD_QUEUE_MAX,
        start_worker: bool = True,
    ):
        self.backend = backend or PiperStableTwiBackend()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_voice = voice or default_voice or C.TTS_VOICE
        self.language = language
        self.queue_max = queue_max

        self._queue: queue.Queue[Optional[_QueueItem]] = queue.Queue(maxsize=queue_max)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        if start_worker:
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="TwiTTS-Worker")
            self._worker_thread.start()

    @property
    def voice(self) -> str:
        return self.default_voice

    def get_voices(self) -> Dict[str, Any]:
        return self.backend.get_voices()

    def _resolve_voice(self, voice: Optional[str]) -> str:
        v = voice or self.default_voice
        if not v or v == "auto":
            return "twi-6"
        return v

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        audio_override: Optional[str] = None,
    ) -> Optional[AudioResult]:
        """Synthesizes text or returns cached/override WAV.
        
        Returns None if text is empty and no override is given.
        """
        # 1. Custom audio override
        if audio_override:
            override_path = Path(audio_override)
            if not override_path.is_absolute():
                override_path = C.BASE_DIR / override_path
            if override_path.exists():
                try:
                    with wave.open(str(override_path), "rb") as w:
                        duration_s = w.getnframes() / float(w.getframerate())
                        sample_rate = w.getframerate()
                    return AudioResult(
                        key=f"override_{override_path.stem}",
                        path=override_path,
                        duration_s=duration_s,
                        sample_rate=sample_rate,
                        voice="override",
                        cached=True,
                    )
                except Exception as e:
                    logger.warning("Failed to open audio override %s: %s", override_path, e)

        # 2. Text validation
        norm_text = unicodedata.normalize("NFC", text.strip()) if text else ""
        if not norm_text:
            return None

        chosen_voice = self._resolve_voice(voice)
        chosen_lang = language or self.language
        key = get_cache_key(norm_text, chosen_voice, chosen_lang)
        cache_path = self.cache_dir / f"{key}.wav"

        # 3. Cache hit
        if cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                with wave.open(str(cache_path), "rb") as w:
                    duration_s = w.getnframes() / float(w.getframerate())
                    sample_rate = w.getframerate()
                return AudioResult(
                    key=key,
                    path=cache_path,
                    duration_s=duration_s,
                    sample_rate=sample_rate,
                    voice=chosen_voice,
                    cached=True,
                )
            except Exception as e:
                logger.warning("Cache file %s corrupted: %s", cache_path, e)
                try:
                    cache_path.unlink()
                except OSError:
                    pass

        # 4. Cache miss: synthesize
        wav_bytes, sample_rate, duration_s = self.backend.synthesize(norm_text, chosen_voice, chosen_lang)
        if not wav_bytes:
            return None

        # Write atomically
        tmp_file = tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix=".tmp", delete=False)
        try:
            tmp_file.write(wav_bytes)
            tmp_file.flush()
            tmp_file.close()
            os.replace(tmp_file.name, cache_path)
        except Exception as e:
            logger.error("Failed to write cache file %s: %s", cache_path, e)
            if os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)
            return None

        return AudioResult(
            key=key,
            path=cache_path,
            duration_s=duration_s,
            sample_rate=sample_rate,
            voice=chosen_voice,
            cached=False,
        )

    def enqueue(
        self,
        text: str,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        audio_override: Optional[str] = None,
        callback: Optional[Callable[[Optional[AudioResult]], None]] = None,
    ) -> threading.Event:
        """Enqueues synthesis request. Drops oldest item if queue is full. Returns Event signaled on completion."""
        event = threading.Event()
        item = _QueueItem(
            text=text,
            voice=voice,
            language=language,
            audio_override=audio_override,
            callback=callback,
            event=event,
        )

        if self._queue.full():
            try:
                dropped = self._queue.get_nowait()
                if dropped:
                    logger.warning("TTS queue full (max=%d). Dropping oldest pending item: %r", self.queue_max, dropped.text)
                    if dropped.callback:
                        try:
                            dropped.callback(None)
                        except Exception:
                            pass
                    if dropped.event:
                        dropped.event.set()
                self._queue.task_done()
            except (queue.Empty, ValueError):
                pass

        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning("TTS queue still full after drop; skipping item %r", text)
            event.set()

        return event

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                self._queue.task_done()
                break

            try:
                res = self.speak(
                    text=item.text,
                    voice=item.voice,
                    language=item.language,
                    audio_override=item.audio_override,
                )
                item.result = res
                if item.callback:
                    try:
                        item.callback(res)
                    except Exception as e:
                        logger.exception("Error in TTS callback: %s", e)
            except Exception as e:
                logger.exception("TTS worker error on %r: %s", item.text, e)
            finally:
                if item.event:
                    item.event.set()
                self._queue.task_done()

    def clear_queue(self):
        """Clears all pending items from queue."""
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item and item.event:
                    item.event.set()
                self._queue.task_done()
            except (queue.Empty, ValueError):
                break

    def stop(self):
        """Stops the worker thread cleanly."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            self._worker_thread.join(timeout=1.0)
