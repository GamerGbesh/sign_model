"""Integration tests exercising real ONNX synthesis via StableTwiTTS."""
import wave
from pathlib import Path

import numpy as np
import pytest

from asl.tts import PiperStableTwiBackend, TwiTTS


@pytest.mark.tts
def test_real_tts_synthesis_properties(tmp_path):
    backend = PiperStableTwiBackend()
    cache_dir = tmp_path / "tts_cache_real"
    tts = TwiTTS(backend=backend, cache_dir=cache_dir, start_worker=False)

    word_text = "nsuo"
    res = tts.speak(word_text, voice="twi-6", language="twi")

    assert res is not None
    assert res.path.exists()
    assert res.cached is False

    # Inspect WAV file
    with wave.open(str(res.path), "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        raw_bytes = w.readframes(nframes)

    assert channels == 1, f"Expected mono, got {channels} channels"
    assert sampwidth == 2, f"Expected 16-bit (2 bytes), got {sampwidth}"
    assert framerate == 22050, f"Expected 22050 Hz, got {framerate}"
    assert nframes > 0, "Audio frames must be non-zero"

    duration = nframes / float(framerate)
    assert 0.1 <= duration <= 3.0, f"Duration {duration:.2f}s out of expected bounds"

    # Verify audio is non-silent (RMS amplitude)
    audio_samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(audio_samples**2))
    assert rms > 200.0, f"Audio appears silent: RMS={rms:.2f}"

    # Sentence synthesis duration test
    sentence_text = "Akwaaba, wo ho te sɛn paa?"
    res_sent = tts.speak(sentence_text, voice="twi-6", language="twi")
    assert res_sent is not None
    assert res_sent.duration_s > res.duration_s, "Sentence duration should exceed single-word duration"
