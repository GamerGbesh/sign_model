"""Unit tests for asl.tts module using FakeBackend."""
import threading
import time
import wave
from pathlib import Path

import pytest

from asl.tts import AudioResult, FakeBackend, TwiTTS, get_cache_key


@pytest.fixture
def fake_backend():
    return FakeBackend()


@pytest.fixture
def tts_instance(fake_backend, tmp_path):
    cache_dir = tmp_path / "tts_cache"
    tts = TwiTTS(backend=fake_backend, cache_dir=cache_dir, default_voice="twi-6", queue_max=3, start_worker=True)
    yield tts
    tts.stop()


def test_cache_hit_avoids_backend_invocation(tts_instance, fake_backend):
    text = "Akwaaba"
    # First call: cache miss
    res1 = tts_instance.speak(text)
    assert res1 is not None
    assert res1.cached is False
    assert len(fake_backend.calls) == 1
    assert res1.path.exists()

    # Second call: cache hit
    res2 = tts_instance.speak(text)
    assert res2 is not None
    assert res2.cached is True
    assert res2.key == res1.key
    assert len(fake_backend.calls) == 1  # Not called again


def test_empty_or_whitespace_text_returns_none(tts_instance, fake_backend):
    assert tts_instance.speak("") is None
    assert tts_instance.speak("   ") is None
    assert len(fake_backend.calls) == 0


def test_audio_override_bypasses_backend(tts_instance, fake_backend, tmp_path):
    override_wav = tmp_path / "custom.wav"
    with wave.open(str(override_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 22050)  # 1.0s

    res = tts_instance.speak("anything", audio_override=str(override_wav))
    assert res is not None
    assert res.cached is True
    assert res.voice == "override"
    assert res.path == override_wav
    assert pytest.approx(res.duration_s, 0.01) == 1.0
    assert len(fake_backend.calls) == 0


def test_queue_drops_oldest_on_overflow(fake_backend, tmp_path):
    cache_dir = tmp_path / "tts_cache_q"
    # Create with worker stopped so queue fills
    tts = TwiTTS(backend=fake_backend, cache_dir=cache_dir, queue_max=2, start_worker=False)

    tts.enqueue("first")
    tts.enqueue("second")
    # Queue is now full (2 items)
    # Enqueue third -> should drop "first"
    tts.enqueue("third")

    items = []
    while not tts._queue.empty():
        item = tts._queue.get_nowait()
        if item:
            items.append(item.text)
    assert items == ["second", "third"]


def test_async_enqueue_and_worker_processing(tts_instance):
    results = []
    done_ev = threading.Event()

    def on_done(res):
        if res:
            results.append(res)
        done_ev.set()

    ev = tts_instance.enqueue("wo ho te sɛn", callback=on_done)
    ev.wait(timeout=2.0)
    done_ev.wait(timeout=2.0)

    assert len(results) == 1
    assert results[0].key == get_cache_key("wo ho te sɛn", "twi-6", "twi")


def test_audio_result_dict_and_item_access():
    res = AudioResult(
        key="abc123",
        path=Path("/tmp/audio.wav"),
        duration_s=1.25,
        sample_rate=22050,
        voice="twi-6",
        cached=False,
    )
    assert res["key"] == "abc123"
    assert res.get("duration_s") == 1.25
    assert res.to_dict()["sample_rate"] == 22050
