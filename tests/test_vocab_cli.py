"""Unit tests for asl.vocab CLI commands."""
import csv
import json
from pathlib import Path

import pytest

from asl.vocab import (
    VocabularyStore,
    cmd_init,
    cmd_list,
    cmd_add_word,
    cmd_add_phrase,
    cmd_set_twi,
    cmd_set_enabled,
    cmd_remove,
    cmd_verify,
    cmd_export_review,
    cmd_validate,
    cmd_status,
)
from asl import config as C


@pytest.fixture
def cli_store(tmp_path):
    vpath = tmp_path / "vocabulary.json"
    store = VocabularyStore(path=vpath, schema_path=C.VOCAB_SCHEMA)
    # Init empty valid vocab
    data = {
        "$schema": str(C.VOCAB_SCHEMA),
        "version": 1,
        "dialect": "",
        "defaults": {"voice": "auto", "unknown_word_policy": "skip"},
        "words": {},
        "phrases": [],
    }
    store.save(data)
    return store


def test_cli_add_word(cli_store):
    ret = cmd_add_word(cli_store, "hello", "Akwaaba", english="hello", notes="greeting")
    assert ret == 0

    snap = cli_store.get_snapshot()
    assert "hello" in snap.words
    assert snap.words["hello"]["twi"] == "Akwaaba"
    assert snap.words["hello"]["enabled"] is True
    assert snap.words["hello"]["status"] == "unverified"


def test_cli_add_phrase(cli_store):
    ret = cmd_add_phrase(cli_store, "thank you", "Medaase", phrase_id="thank-you")
    assert ret == 0

    snap = cli_store.get_snapshot()
    assert len(snap.phrases) == 1
    p = snap.phrases[0]
    assert p["id"] == "thank-you"
    assert p["match"] == ["thank", "you"]
    assert p["twi"] == "Medaase"
    assert p["enabled"] is True


def test_cli_set_twi(cli_store):
    cmd_add_word(cli_store, "friend", "")
    assert cli_store.get_snapshot().words["friend"]["enabled"] is False

    ret = cmd_set_twi(cli_store, "friend", "adamfo")
    assert ret == 0
    snap = cli_store.get_snapshot()
    assert snap.words["friend"]["twi"] == "adamfo"
    assert snap.words["friend"]["enabled"] is True


def test_cli_enable_disable(cli_store):
    cmd_add_word(cli_store, "friend", "adamfo")
    cmd_add_phrase(cli_store, "thank you", "Medaase", phrase_id="p1")

    # Disable word
    cmd_set_enabled(cli_store, "friend", False)
    assert cli_store.get_snapshot().words["friend"]["enabled"] is False

    # Disable phrase
    cmd_set_enabled(cli_store, "p1", False)
    assert cli_store.get_snapshot().phrases[0]["enabled"] is False

    # Enable word
    cmd_set_enabled(cli_store, "friend", True)
    assert cli_store.get_snapshot().words["friend"]["enabled"] is True


def test_cli_verify(cli_store):
    cmd_add_word(cli_store, "friend", "adamfo")
    ret = cmd_verify(cli_store, "friend", by_name="Dr. Mensah")
    assert ret == 0
    w = cli_store.get_snapshot().words["friend"]
    assert w["status"] == "verified"
    assert w["verified_by"] == "Dr. Mensah"


def test_cli_remove(cli_store):
    cmd_add_word(cli_store, "friend", "adamfo")
    cmd_add_phrase(cli_store, "thank you", "Medaase", phrase_id="p1")

    cmd_remove(cli_store, "friend")
    assert "friend" not in cli_store.get_snapshot().words

    cmd_remove(cli_store, "p1")
    assert len(cli_store.get_snapshot().phrases) == 0


def test_cli_export_review(cli_store, tmp_path):
    cmd_add_word(cli_store, "friend", "adamfo")
    cmd_add_phrase(cli_store, "thank you", "Medaase", phrase_id="p1")

    out_csv = tmp_path / "review.csv"
    ret = cmd_export_review(cli_store, out_path=out_csv)
    assert ret == 0
    assert out_csv.exists()

    with open(out_csv, newline="", encoding="utf-8") as f:
        reader = list(csv.reader(f))
    assert reader[0] == ["id", "type", "english", "twi", "status", "verified_by", "notes"]
    assert len(reader) == 3  # header + 1 word + 1 phrase


def test_cli_validate(cli_store):
    cmd_add_word(cli_store, "friend", "adamfo")
    ret = cmd_validate(cli_store)
    assert ret == 0


def test_cli_status_and_list(cli_store, capsys):
    cmd_add_word(cli_store, "friend", "adamfo")
    assert cmd_list(cli_store) == 0
    assert cmd_status(cli_store) == 0
    out = capsys.readouterr().out
    assert "AMEGBE VOCABULARY" in out
    assert "friend" in out
