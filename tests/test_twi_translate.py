"""Unit tests for asl.twi_translate."""
import time
import unicodedata
import pytest

from asl.vocab import VocabSnapshot
from asl.twi_translate import translate_glosses, TranslationItem, TranslationResult


@pytest.fixture
def sample_snapshot():
    raw_data = {
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
                "status": "verified",
                "verified_by": "Dr. Mensah",
                "audio": "data/audio/hello.wav",
            },
            "friend": {
                "enabled": True,
                "english": "friend",
                "twi": "adamfo",
                "status": "unverified",
                "verified_by": None,
                "audio": None,
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
            "disabled_sign": {
                "enabled": False,
                "english": "disabled sign",
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
                "audio": "data/audio/medaase.wav",
            },
            {
                "id": "how-are-you-doing",
                "match": ["how", "are", "you", "doing"],
                "twi": "Wo ho te sɛn paa?",
                "enabled": True,
                "status": "unverified",
            },
            {
                "id": "how-are-you",
                "match": ["how", "are", "you"],
                "twi": "Wo ho te sɛn?",
                "enabled": True,
                "status": "unverified",
            },
        ],
    }
    return VocabSnapshot(
        raw_data=raw_data,
        version=1,
        dialect="asante",
        defaults=raw_data["defaults"],
        words=raw_data["words"],
        phrases=raw_data["phrases"],
    )


def test_phrase_match_wins_over_individual_words(sample_snapshot):
    res = translate_glosses(["thank", "you"], sample_snapshot)
    assert res.twi_text == "Medaase"
    assert len(res.items) == 1
    assert res.items[0].rule_type == "phrase"
    assert res.items[0].entry_id == "thank-you"
    assert res.items[0].audio == "data/audio/medaase.wav"
    assert res.items[0].verified is True
    assert res.untranslated == []


def test_longest_phrase_match(sample_snapshot):
    # Should match "how-are-you-doing" (4 tokens) rather than "how-are-you" (3 tokens)
    res = translate_glosses(["how", "are", "you", "doing"], sample_snapshot)
    assert res.twi_text == "Wo ho te sɛn paa?"
    assert len(res.items) == 1
    assert res.items[0].entry_id == "how-are-you-doing"

    # Should match "how-are-you" (3 tokens)
    res2 = translate_glosses(["how", "are", "you"], sample_snapshot)
    assert res2.twi_text == "Wo ho te sɛn?"
    assert len(res2.items) == 1
    assert res2.items[0].entry_id == "how-are-you"


def test_single_word_translation(sample_snapshot):
    res = translate_glosses(["hello", "friend"], sample_snapshot)
    assert res.twi_text == "Akwaaba adamfo"
    assert res.english_text == "hello friend"
    assert len(res.items) == 2
    assert res.items[0].entry_id == "hello"
    assert res.items[0].audio == "data/audio/hello.wav"
    assert res.items[0].verified is True
    assert res.items[1].entry_id == "friend"
    assert res.items[1].audio is None
    assert res.items[1].verified is False


def test_disabled_word_skipped_under_skip_policy(sample_snapshot):
    res = translate_glosses(["hello", "disabled_sign", "friend"], sample_snapshot)
    assert res.twi_text == "Akwaaba adamfo"
    assert res.english_text == "hello friend"
    assert res.untranslated == ["disabled_sign"]
    assert len(res.items) == 2


def test_disabled_word_bracketed_under_english_policy(sample_snapshot):
    # Snapshot with english unknown word policy
    raw_data = dict(sample_snapshot.raw_data)
    raw_data["defaults"] = {"voice": "auto", "unknown_word_policy": "english"}
    snap = VocabSnapshot(
        raw_data=raw_data,
        version=1,
        dialect="asante",
        defaults=raw_data["defaults"],
        words=raw_data["words"],
        phrases=raw_data["phrases"],
    )

    res = translate_glosses(["hello", "disabled_sign", "friend"], snap)
    assert res.twi_text == "Akwaaba [disabled_sign] adamfo"
    assert res.english_text == "hello [disabled_sign] friend"
    assert res.untranslated == ["disabled_sign"]
    assert len(res.items) == 3
    assert res.items[1].rule_type == "unknown"
    assert res.items[1].twi == "[disabled_sign]"


def test_digits_handled_safely(sample_snapshot):
    raw_data = dict(sample_snapshot.raw_data)
    raw_data["defaults"] = {"voice": "auto", "unknown_word_policy": "english"}
    snap = VocabSnapshot(
        raw_data=raw_data,
        version=1,
        dialect="asante",
        defaults=raw_data["defaults"],
        words=raw_data["words"],
        phrases=raw_data["phrases"],
    )

    # Pure digits should be skipped entirely
    res = translate_glosses(["12345"], snap)
    assert res.twi_text == ""
    assert res.untranslated == ["12345"]

    # Alphanumeric with digits should have digits stripped
    res2 = translate_glosses(["test123word"], snap)
    assert res2.twi_text == "[testword]"


def test_empty_input(sample_snapshot):
    res = translate_glosses([], sample_snapshot)
    assert res.twi_text == ""
    assert res.english_text == ""
    assert res.items == []
    assert res.untranslated == []


def test_nfc_normalization_preserved(sample_snapshot):
    res = translate_glosses(["how", "are", "you"], sample_snapshot)
    assert unicodedata.is_normalized("NFC", res.twi_text)


def test_performance(sample_snapshot):
    input_glosses = ["hello", "friend", "thank", "you", "hello"]
    start = time.perf_counter()
    for _ in range(500):
        translate_glosses(input_glosses, sample_snapshot)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / 500) * 1000
    assert avg_ms < 1.0, f"Average latency too high: {avg_ms:.3f}ms"
