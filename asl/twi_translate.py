"""Translation layer: converts recognized sign glosses into Twi and English text.

Deterministic, offline, auditable dictionary lookup with longest-match phrase rules,
word fallback, unknown-word policies, and audio override propagation.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .vocab import VocabSnapshot

DIGIT_PATTERN = re.compile(r"\d")


@dataclass(frozen=True)
class TranslationItem:
    glosses: List[str]
    rule_type: str  # "phrase", "word", or "unknown"
    twi: str
    english: str
    entry_id: str
    audio: Optional[str] = None
    verified: bool = False


@dataclass(frozen=True)
class TranslationResult:
    twi_text: str
    english_text: str
    items: List[TranslationItem]
    untranslated: List[str]


def _match_phrase_at(
    glosses: Sequence[str],
    start_idx: int,
    phrases: Sequence[dict],
) -> Optional[tuple[dict, int]]:
    """Finds the longest enabled phrase matching glosses starting at start_idx.
    
    Returns (phrase_dict, match_length) or None.
    """
    best_phrase: Optional[dict] = None
    best_len = 0

    remaining = len(glosses) - start_idx
    for phrase in phrases:
        tokens = [t.lower() for t in phrase.get("match", [])]
        k = len(tokens)
        if 2 <= k <= remaining:
            if [g.lower() for g in glosses[start_idx : start_idx + k]] == tokens:
                if k > best_len:
                    best_phrase = phrase
                    best_len = k

    if best_phrase is not None:
        return best_phrase, best_len
    return None


def translate_glosses(
    glosses: Sequence[str],
    vocab: VocabSnapshot,
) -> TranslationResult:
    """Translates a sequence of sign glosses into Twi and English.
    
    1. Longest-match greedy against enabled phrases in vocab.
    2. Fallback to enabled words in vocab.
    3. Handles unknown / disabled words according to vocab.unknown_word_policy ('skip' or 'english').
    """
    if not glosses:
        return TranslationResult(twi_text="", english_text="", items=[], untranslated=[])

    items: List[TranslationItem] = []
    untranslated: List[str] = []
    enabled_phrases = vocab.enabled_phrases
    enabled_words = vocab.enabled_words
    policy = vocab.unknown_word_policy

    i = 0
    n = len(glosses)

    while i < n:
        raw_gloss = glosses[i]
        norm_gloss = raw_gloss.strip().lower()

        # 1. Check phrase match
        phrase_match = _match_phrase_at(glosses, i, enabled_phrases)
        if phrase_match:
            phrase, length = phrase_match
            matched_glosses = list(glosses[i : i + length])
            twi_str = unicodedata.normalize("NFC", phrase.get("twi", "").strip())
            english_str = " ".join(matched_glosses)
            items.append(
                TranslationItem(
                    glosses=matched_glosses,
                    rule_type="phrase",
                    twi=twi_str,
                    english=english_str,
                    entry_id=phrase.get("id", ""),
                    audio=phrase.get("audio"),
                    verified=(phrase.get("status") == "verified"),
                )
            )
            i += length
            continue

        # 2. Check word match (check key as-is, with spaces replaced by _, and _ replaced by spaces)
        word_entry = None
        matched_key = None
        for candidate_key in (norm_gloss, norm_gloss.replace(" ", "_"), norm_gloss.replace("_", " ")):
            if candidate_key in enabled_words:
                word_entry = enabled_words[candidate_key]
                matched_key = candidate_key
                break

        if word_entry and word_entry.get("twi"):
            twi_str = unicodedata.normalize("NFC", word_entry.get("twi", "").strip())
            english_str = word_entry.get("english") or norm_gloss.replace("_", " ")
            items.append(
                TranslationItem(
                    glosses=[raw_gloss],
                    rule_type="word",
                    twi=twi_str,
                    english=english_str,
                    entry_id=matched_key,
                    audio=word_entry.get("audio"),
                    verified=(word_entry.get("status") == "verified"),
                )
            )
            i += 1
            continue

        # 3. Unknown or disabled word
        untranslated.append(raw_gloss)
        if policy == "english":
            # Strip digits if present, as Twi TTS fails on digits
            clean = DIGIT_PATTERN.sub("", raw_gloss).strip()
            if clean:
                bracketed = f"[{clean}]"
                items.append(
                    TranslationItem(
                        glosses=[raw_gloss],
                        rule_type="unknown",
                        twi=bracketed,
                        english=bracketed,
                        entry_id=raw_gloss,
                        audio=None,
                        verified=False,
                    )
                )
        i += 1

    twi_parts = [it.twi for it in items if it.twi]
    english_parts = [it.english for it in items if it.english]

    twi_text = unicodedata.normalize("NFC", " ".join(twi_parts))
    english_text = " ".join(english_parts)

    return TranslationResult(
        twi_text=twi_text,
        english_text=english_text,
        items=items,
        untranslated=untranslated,
    )
