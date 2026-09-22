"""Test verifying the documented end-to-end vocabulary workflow from docs/VOCABULARY.md."""
import csv
import json
from pathlib import Path

import pytest

from asl import config as C
from asl.tts import FakeBackend, TwiTTS
from asl.twi_translate import translate_glosses
from asl.vocab import (
    VocabularyStore,
    cmd_add_word,
    cmd_export_review,
    cmd_validate,
    cmd_verify,
)


def test_documented_add_word_flow(tmp_path):
    """Executes the documented add-a-word steps in an isolated temporary store."""
    vpath = tmp_path / "vocabulary.json"
    store = VocabularyStore(path=vpath, schema_path=C.VOCAB_SCHEMA)

    # Scaffolding initial empty vocabulary
    initial_data = {
        "$schema": str(C.VOCAB_SCHEMA),
        "version": 1,
        "dialect": "asante",
        "defaults": {
            "voice": "auto",
            "unknown_word_policy": "skip",
        },
        "words": {},
        "phrases": [],
    }
    store.save(initial_data)

    # Step 1: Add vocabulary entry
    res_add = cmd_add_word(store, "water", "nsuo", english="water", notes="standard Asante")
    assert res_add == 0
    snap1 = store.get_snapshot()
    assert "water" in snap1.words
    assert snap1.words["water"]["twi"] == "nsuo"
    assert snap1.words["water"]["enabled"] is True
    assert snap1.words["water"]["status"] == "unverified"

    # Step 2: Validate schema & pronounceability
    res_val = cmd_validate(store)
    assert res_val == 0

    # Step 3: Export review sheet
    csv_path = tmp_path / "review.csv"
    res_exp = cmd_export_review(store, out_path=csv_path)
    assert res_exp == 0
    assert csv_path.exists()
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # header + 'water'
    assert rows[1][0] == "water"
    assert rows[1][3] == "nsuo"
    assert rows[1][4] == "unverified"

    # Step 4: Verification by native speaker
    res_ver = cmd_verify(store, "water", by_name="Dr. Mensah")
    assert res_ver == 0
    snap2 = store.get_snapshot()
    assert snap2.words["water"]["status"] == "verified"
    assert snap2.words["water"]["verified_by"] == "Dr. Mensah"

    # Step 5: Translation layer lookup
    trans = translate_glosses(["water"], snap2)
    assert trans.twi_text == "nsuo"
    assert trans.english_text == "water"
    assert len(trans.items) == 1
    assert trans.items[0].verified is True

    # Step 6: Audio pre-rendering / caching
    fake_backend = FakeBackend()
    tts = TwiTTS(backend=fake_backend, cache_dir=tmp_path / "cache", start_worker=False)
    audio_res = tts.speak(trans.twi_text)
    assert audio_res is not None
    assert audio_res.path.exists()
    assert audio_res.cached is False

    # Second lookup: cached
    audio_res2 = tts.speak(trans.twi_text)
    assert audio_res2 is not None
    assert audio_res2.cached is True
