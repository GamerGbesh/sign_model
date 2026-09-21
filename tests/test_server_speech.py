"""Integration tests for Amegbe speech REST and WebSocket endpoints."""
import os
import wave
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from asl import config as C
from asl.server import app, get_vocab_store, get_tts
from asl.tts import FakeBackend, get_cache_key


@pytest.fixture
def client(tmp_path):
    os.environ["DEV"] = "1"
    # Ensure cache dir exists
    C.TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with TestClient(app) as c:
        yield c


def test_audio_endpoint_valid_key(client, tmp_path):
    key = "a" * 40
    test_wav = C.TTS_CACHE_DIR / f"{key}.wav"
    with wave.open(str(test_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 2205)

    try:
        res = client.get(f"/audio/{key}.wav")
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/wav"
        assert "immutable" in res.headers["cache-control"]
    finally:
        if test_wav.exists():
            test_wav.unlink()


def test_audio_endpoint_traversal_rejected(client):
    # Directory traversal attempt
    res = client.get("/audio/../../etc/passwd.wav")
    assert res.status_code in (400, 404)

    # Invalid key length
    res2 = client.get("/audio/shortkey.wav")
    assert res2.status_code == 400


def test_audio_endpoint_missing_404(client):
    key = "b" * 40
    res = client.get(f"/audio/{key}.wav")
    assert res.status_code == 404


def test_vocab_endpoints(client):
    res = client.get("/vocab")
    assert res.status_code == 200
    data = res.json()
    assert "words" in data
    assert "phrases" in data

    # Test reload
    res_reload = client.post("/vocab/reload")
    assert res_reload.status_code == 200
    reload_data = res_reload.json()
    assert reload_data["ok"] is True
    assert "count" in reload_data


def test_tts_voices_endpoint(client):
    res = client.get("/tts/voices")
    assert res.status_code == 200
    data = res.json()
    assert "tiers" in data or "voices" in data


def test_websocket_speech_commands(client):
    with client.websocket_connect("/ws/recognize") as ws:
        ready_msg = ws.receive_json()
        assert ready_msg["type"] == "ready"
        assert ready_msg.get("app") == "amegbe"

        # 1. Config command
        ws.send_json({"type": "config", "speech_mode": "sentence", "voice": "twi-6"})
        config_ack = ws.receive_json()
        assert config_ack["type"] == "config_ack"
        assert config_ack["speech_mode"] == "sentence"
        assert config_ack["voice"] == "twi-6"

        # 2. Clear command
        ws.send_json({"type": "clear"})
        clear_ack = ws.receive_json()
        assert clear_ack["type"] == "clear_ack"

        # 3. Speak command (empty buffer -> no speech output)
        ws.send_json({"type": "speak"})
