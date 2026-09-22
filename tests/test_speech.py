"""Unit tests for asl.speech module using FakeBackend."""
import json
import pytest

from asl.speech import SpeechOrchestrator
from asl.tts import FakeBackend, TwiTTS, get_cache_key
from asl.vocab import VocabularyStore
from asl import config as C


@pytest.fixture
def mock_vocab_store(tmp_path):
    vpath = tmp_path / "vocabulary.json"
    data = {
        "$schema": str(C.VOCAB_SCHEMA),
        "version": 1,
        "dialect": "asante",
        "defaults": {"voice": "auto", "unknown_word_policy": "skip"},
        "words": {
            "hello": {
                "enabled": True,
                "english": "hello",
                "twi": "Akwaaba",
                "status": "verified",
                "verified_by": "Dr. Mensah",
            },
            "friend": {
                "enabled": True,
                "english": "friend",
                "twi": "adamfo",
                "status": "unverified",
            },
            "thank": {
                "enabled": True,
                "english": "thank",
                "twi": "daase",
                "status": "unverified",
            },
            "you": {
                "enabled": True,
                "english": "you",
                "twi": "wo",
                "status": "unverified",
            },
            "disabled_word": {
                "enabled": False,
                "english": "disabled word",
                "twi": "",
                "status": "unverified",
            },
        },
        "phrases": [
            {
                "id": "thank-you",
                "match": ["thank", "you"],
                "twi": "Medaase",
                "enabled": True,
                "status": "verified",
            }
        ],
    }
    vpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return VocabularyStore(path=vpath, schema_path=C.VOCAB_SCHEMA)


@pytest.fixture
def mock_tts(tmp_path):
    cache_dir = tmp_path / "tts_cache"
    fake_backend = FakeBackend()
    return TwiTTS(backend=fake_backend, cache_dir=cache_dir, default_voice="twi-6", start_worker=False)


def test_word_mode_speaks_immediately(mock_vocab_store, mock_tts):
    emitted = []
    orch = SpeechOrchestrator(
        vocab_store=mock_vocab_store,
        tts=mock_tts,
        mode="word",
        on_event=emitted.append,
    )

    events = orch.on_commit("hello", t_s=100.0)
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "speech"
    assert ev["twi"] == "Akwaaba"
    assert ev["mode"] == "word"
    assert ev["is_sentence"] is False
    assert ev["verified"] is True
    expected_key = get_cache_key("Akwaaba", "twi-6", "twi")
    assert ev["audio_url"] == f"/audio/{expected_key}.wav"
    assert len(emitted) == 1


def test_sentence_mode_buffers_until_pause(mock_vocab_store, mock_tts):
    orch = SpeechOrchestrator(
        vocab_store=mock_vocab_store,
        tts=mock_tts,
        mode="sentence",
        pause_s=2.0,
    )

    # First commit: buffered, no speech
    evs1 = orch.on_commit("thank", t_s=100.0)
    assert evs1 == []

    # Second commit within pause window: still buffered
    evs2 = orch.on_commit("you", t_s=101.0)
    assert evs2 == []

    # Tick before pause threshold (1.0s elapsed)
    assert orch.tick(t_s=102.0) == []

    # Tick after pause threshold (2.1s elapsed since last commit)
    evs3 = orch.tick(t_s=103.1)
    assert len(evs3) == 1
    ev = evs3[0]
    assert ev["type"] == "speech"
    # Should use phrase match "Medaase"
    assert ev["twi"] == "Medaase"
    assert ev["mode"] == "sentence"
    assert ev["is_sentence"] is True
    assert ev["verified"] is True


def test_speak_now_forces_flush(mock_vocab_store, mock_tts):
    orch = SpeechOrchestrator(
        vocab_store=mock_vocab_store,
        tts=mock_tts,
        mode="sentence",
    )
    orch.on_commit("hello", t_s=100.0)
    orch.on_commit("friend", t_s=100.5)

    evs = orch.speak_now()
    assert len(evs) == 1
    assert evs[0]["twi"] == "Akwaaba adamfo"
    assert evs[0]["is_sentence"] is True


def test_clear_discards_buffer(mock_vocab_store, mock_tts):
    orch = SpeechOrchestrator(
        vocab_store=mock_vocab_store,
        tts=mock_tts,
        mode="sentence",
    )
    orch.on_commit("hello", t_s=100.0)
    orch.clear()
    assert orch.speak_now() == []


def test_mode_switch_runtime(mock_vocab_store, mock_tts):
    orch = SpeechOrchestrator(
        vocab_store=mock_vocab_store,
        tts=mock_tts,
        mode="word",
    )
    assert len(orch.on_commit("hello")) == 1

    # Switch to off
    orch.set_mode("off")
    assert orch.on_commit("hello") == []

    # Switch to sentence
    orch.set_mode("sentence")
    orch.on_commit("hello")
    evs = orch.speak_now()
    assert len(evs) == 1


def test_disabled_words_ignored(mock_vocab_store, mock_tts):
    orch = SpeechOrchestrator(
        vocab_store=mock_vocab_store,
        tts=mock_tts,
        mode="word",
    )
    evs = orch.on_commit("disabled_word")
    assert evs == []


def test_word_queue_backpressure(mock_vocab_store, mock_tts):
    orch = SpeechOrchestrator(
        vocab_store=mock_vocab_store,
        tts=mock_tts,
        mode="word",
        word_queue_max=2,
    )
    # Commit 3 words in succession without acknowledging
    orch.on_commit("hello")
    orch.on_commit("friend")
    assert orch.pending_count == 2

    # Third word triggers drop of oldest ("hello")
    orch.on_commit("hello")
    assert orch.pending_count == 2
    assert [ev["twi"] for ev in orch._pending_word_queue] == ["adamfo", "Akwaaba"]


def test_enrich_phrase_event(mock_vocab_store, mock_tts):
    orch = SpeechOrchestrator(
        vocab_store=mock_vocab_store,
        tts=mock_tts,
        mode="off",
    )
    phrase_ev = {"type": "phrase", "words": ["thank", "you"], "duration_s": 1.2}
    enriched = orch.enrich_phrase_event(phrase_ev)
    assert enriched["twi"] == "Medaase"
    assert enriched["twi_verified"] is True
