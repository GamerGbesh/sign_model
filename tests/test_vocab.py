"""Unit tests for VocabularyStore and VocabSnapshot in asl.vocab."""
import json
import time
import unicodedata
import wave
from pathlib import Path

import pytest

from asl.vocab import VocabSnapshot, VocabularyStore
from asl import config as C


@pytest.fixture
def sample_schema_path():
    return C.VOCAB_SCHEMA


@pytest.fixture
def temp_vocab(tmp_path, sample_schema_path):
    vpath = tmp_path / "vocabulary.json"
    data = {
        "$schema": str(sample_schema_path),
        "version": 1,
        "dialect": "asante",
        "defaults": {
            "voice": "auto",
            "unknown_word_policy": "skip",
        },
        "words": {
            "hello": {
                "enabled": True,
                "english": "hello",
                "twi": "Akwaaba",
                "status": "unverified",
                "verified_by": None,
                "audio": None,
                "notes": "Test greeting (illustrative, unverified)",
            },
            "no": {
                "enabled": False,
                "english": "no",
                "twi": "",
                "status": "unverified",
                "verified_by": None,
                "audio": None,
                "notes": "",
            },
        },
        "phrases": [
            {
                "id": "how-are-you",
                "match": ["how", "are", "you"],
                "twi": "Wo ho te sɛn?",
                "enabled": True,
                "status": "unverified",
                "audio": None,
            }
        ],
    }
    vpath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return vpath


def test_vocab_store_load_valid(temp_vocab, sample_schema_path):
    store = VocabularyStore(path=temp_vocab, schema_path=sample_schema_path)
    snap = store.get_snapshot()

    assert snap.version == 1
    assert snap.dialect == "asante"
    assert snap.default_voice == "auto"
    assert snap.unknown_word_policy == "skip"
    assert len(snap.words) == 2
    assert "hello" in snap.enabled_words
    assert "no" not in snap.enabled_words
    assert snap.enabled_glosses == {"hello"}
    assert len(snap.enabled_phrases) == 1
    assert snap.enabled_phrases[0]["id"] == "how-are-you"


def test_vocab_store_nfc_normalization(temp_vocab, sample_schema_path):
    store = VocabularyStore(path=temp_vocab, schema_path=sample_schema_path)
    data = dict(store.get_snapshot().raw_data)

    accented_nfd = "e\u0301"  # é decomposed
    data["words"]["test"] = {
        "enabled": True,
        "english": "test",
        "twi": f"t{accented_nfd}st",
        "status": "unverified",
    }
    errors = store.validate_data(data)
    assert any("NFC" in e for e in errors)


def test_vocab_store_digits_rejected(temp_vocab, sample_schema_path):
    store = VocabularyStore(path=temp_vocab, schema_path=sample_schema_path)
    data = dict(store.get_snapshot().raw_data)
    data["words"]["test"] = {
        "enabled": True,
        "english": "test",
        "twi": "baako 1",
        "status": "unverified",
    }
    errors = store.validate_data(data)
    assert any("digits" in e for e in errors)


def test_vocab_store_enabled_empty_twi(temp_vocab, sample_schema_path):
    store = VocabularyStore(path=temp_vocab, schema_path=sample_schema_path)
    data = dict(store.get_snapshot().raw_data)
    data["words"]["test"] = {
        "enabled": True,
        "english": "test",
        "twi": "   ",
        "status": "unverified",
    }
    errors = store.validate_data(data)
    assert any("empty Twi translation" in e for e in errors)


def test_vocab_store_duplicate_phrase(temp_vocab, sample_schema_path):
    store = VocabularyStore(path=temp_vocab, schema_path=sample_schema_path)
    data = dict(store.get_snapshot().raw_data)
    # Duplicate ID
    data["phrases"].append({
        "id": "how-are-you",
        "match": ["other", "match"],
        "twi": "Akwaaba",
        "enabled": True,
    })
    errors = store.validate_data(data)
    assert any("Duplicate phrase ID" in e for e in errors)

    # Collision in match tokens
    data["phrases"] = [
        {"id": "p1", "match": ["thank", "you"], "twi": "Medaase", "enabled": True},
        {"id": "p2", "match": ["thank", "you"], "twi": "Mepaakyɛw", "enabled": True},
    ]
    errors2 = store.validate_data(data)
    assert any("Phrase collision" in e for e in errors2)


def test_vocab_store_audio_file_validation(temp_vocab, sample_schema_path, tmp_path):
    store = VocabularyStore(path=temp_vocab, schema_path=sample_schema_path)
    data = dict(store.get_snapshot().raw_data)

    # Non-existent audio
    data["words"]["hello"]["audio"] = str(tmp_path / "nonexistent.wav")
    errors = store.validate_data(data)
    assert any("Audio file not found" in e for e in errors)

    # Valid mono wav
    mono_wav = tmp_path / "mono.wav"
    with wave.open(str(mono_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * 100)

    data["words"]["hello"]["audio"] = str(mono_wav)
    errors_mono = store.validate_data(data)
    assert not any("Audio" in e for e in errors_mono)

    # Stereo wav (rejected)
    stereo_wav = tmp_path / "stereo.wav"
    with wave.open(str(stereo_wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00\x00\x00" * 100)

    data["words"]["hello"]["audio"] = str(stereo_wav)
    errors_stereo = store.validate_data(data)
    assert any("Audio must be mono" in e for e in errors_stereo)


def test_vocab_store_verified_requires_verifier(temp_vocab, sample_schema_path):
    store = VocabularyStore(path=temp_vocab, schema_path=sample_schema_path)
    data = dict(store.get_snapshot().raw_data)
    data["words"]["hello"]["status"] = "verified"
    data["words"]["hello"]["verified_by"] = None

    errors = store.validate_data(data)
    assert any("verified but verified_by is empty" in e for e in errors)

    data["words"]["hello"]["verified_by"] = "Dr. Mensah"
    assert not store.validate_data(data)


def test_vocab_store_hot_reload_success(temp_vocab, sample_schema_path):
    store = VocabularyStore(path=temp_vocab, schema_path=sample_schema_path)
    assert "friend" not in store.get_snapshot().words

    # Edit file directly
    data = json.loads(temp_vocab.read_text(encoding="utf-8"))
    data["words"]["friend"] = {
        "enabled": True,
        "english": "friend",
        "twi": "adamfo",
        "status": "unverified",
    }
    time.sleep(0.01)  # ensure mtime step
    temp_vocab.write_text(json.dumps(data), encoding="utf-8")

    # Access snapshot - should detect mtime change and reload
    snap = store.get_snapshot()
    assert "friend" in snap.words
    assert snap.words["friend"]["twi"] == "adamfo"


def test_vocab_store_hot_reload_failure_retains_snapshot(temp_vocab, sample_schema_path):
    store = VocabularyStore(path=temp_vocab, schema_path=sample_schema_path)
    initial_words = set(store.get_snapshot().words.keys())

    # Write corrupt JSON
    time.sleep(0.01)
    temp_vocab.write_text("{corrupt json", encoding="utf-8")

    ok, errors = store.reload()
    assert not ok
    assert any("Invalid JSON" in e for e in errors)
    # Old snapshot retained
    assert set(store.get_snapshot().words.keys()) == initial_words

    # Write invalid schema (empty Twi on enabled)
    bad_data = {
        "version": 1,
        "words": {
            "broken": {"enabled": True, "twi": ""}
        }
    }
    temp_vocab.write_text(json.dumps(bad_data), encoding="utf-8")
    ok2, errors2 = store.reload()
    assert not ok2
    assert set(store.get_snapshot().words.keys()) == initial_words
