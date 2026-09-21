"""Vocabulary management, validation, and CLI for SignSpeak / Amegbe.

Maintains vocab/vocabulary.json as the single source of truth linking sign glosses,
English text, Twi translations, and audio overrides. Provides schema validation,
pronounceability checking via ghana-g2p, immutable snapshots, and developer CLI.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonschema

from . import config as C

ALLOWED_KEY_PATTERN = re.compile(r"^[a-z0-9_ -]+$")
DIGIT_PATTERN = re.compile(r"\d")


@dataclass(frozen=True)
class VocabSnapshot:
    raw_data: Dict[str, Any]
    version: int
    dialect: str
    defaults: Dict[str, Any]
    words: Dict[str, Dict[str, Any]]
    phrases: List[Dict[str, Any]]

    @property
    def enabled_words(self) -> Dict[str, Dict[str, Any]]:
        return {k: v for k, v in self.words.items() if v.get("enabled", False)}

    @property
    def enabled_phrases(self) -> List[Dict[str, Any]]:
        return [p for p in self.phrases if p.get("enabled", False)]

    @property
    def enabled_glosses(self) -> Set[str]:
        return {k for k, v in self.words.items() if v.get("enabled", False)}

    @property
    def unknown_word_policy(self) -> str:
        return self.defaults.get("unknown_word_policy", "skip")

    @property
    def default_voice(self) -> str:
        return self.defaults.get("voice", "auto")


class VocabularyStore:
    def __init__(
        self,
        path: Path = C.VOCAB_JSON,
        schema_path: Path = C.VOCAB_SCHEMA,
    ):
        self.path = Path(path)
        self.schema_path = Path(schema_path)
        self._mtime: Optional[float] = None
        self._snapshot: Optional[VocabSnapshot] = None
        self._load_initial()

    def _load_schema(self) -> Dict[str, Any]:
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file missing: {self.schema_path}")
        return json.loads(self.schema_path.read_text(encoding="utf-8"))

    def _load_initial(self) -> None:
        if self.path.exists():
            ok, errors = self.reload(force=True)
            if not ok:
                print(f"[WARN] Failed to load initial vocabulary: {errors}", file=sys.stderr)
        else:
            # Fallback empty snapshot
            self._snapshot = VocabSnapshot(
                raw_data={"version": 1, "dialect": "", "defaults": {"voice": "auto", "unknown_word_policy": "skip"}, "words": {}, "phrases": []},
                version=1,
                dialect="",
                defaults={"voice": "auto", "unknown_word_policy": "skip"},
                words={},
                phrases=[],
            )

    def get_snapshot(self) -> VocabSnapshot:
        """Returns the current immutable snapshot, reloading if file changed on disk."""
        if self.path.exists():
            curr_mtime = self.path.stat().st_mtime
            if self._mtime is None or curr_mtime != self._mtime:
                self.reload()
        return self._snapshot

    def reload(self, force: bool = False) -> Tuple[bool, List[str]]:
        """Reloads vocabulary from disk. Swaps snapshot ONLY if validation passes."""
        if not self.path.exists():
            return False, [f"File does not exist: {self.path}"]

        curr_mtime = self.path.stat().st_mtime
        if not force and self._mtime is not None and curr_mtime == self._mtime:
            return True, []

        try:
            content = self.path.read_text(encoding="utf-8")
            data = json.loads(content)
        except Exception as e:
            return False, [f"Invalid JSON: {e}"]

        errors = self.validate_data(data, check_pronounceable=False)
        if errors:
            # Keep previous snapshot, report errors
            return False, errors

        # Validation passed: update snapshot and mtime
        self._mtime = curr_mtime
        self._snapshot = VocabSnapshot(
            raw_data=data,
            version=data.get("version", 1),
            dialect=data.get("dialect", ""),
            defaults=data.get("defaults", {}),
            words=dict(data.get("words", {})),
            phrases=list(data.get("phrases", [])),
        )
        return True, []

    def save(self, data: Dict[str, Any]) -> None:
        """Validates and writes vocabulary data to disk."""
        errors = self.validate_data(data, check_pronounceable=False)
        if errors:
            raise ValueError(f"Cannot save invalid vocabulary:\n" + "\n".join(f" - {e}" for e in errors))

        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        self.path.write_text(content, encoding="utf-8")
        self._mtime = self.path.stat().st_mtime
        self._snapshot = VocabSnapshot(
            raw_data=data,
            version=data.get("version", 1),
            dialect=data.get("dialect", ""),
            defaults=data.get("defaults", {}),
            words=dict(data.get("words", {})),
            phrases=list(data.get("phrases", [])),
        )

    def validate_data(
        self,
        data: Dict[str, Any],
        check_pronounceable: bool = False,
        model_labels: Optional[List[str]] = None,
    ) -> List[str]:
        """Validates data dictionary against schema, orthography, and pronounceability rules."""
        errors: List[str] = []

        # 1. JSON Schema validation
        schema = self._load_schema()
        validator = jsonschema.Draft202012Validator(schema)
        for err in validator.iter_errors(data):
            path = ".".join(str(p) for p in err.path)
            errors.append(f"Schema error at '{path}': {err.message}")

        words = data.get("words", {})
        phrases = data.get("phrases", [])

        # 2. Word key checks
        for key, entry in words.items():
            if not ALLOWED_KEY_PATTERN.match(key):
                errors.append(f"Word key '{key}' contains invalid characters (allowed: a-z, 0-9, _, -, space)")

            twi = entry.get("twi", "")
            enabled = entry.get("enabled", False)

            # NFC normalization check
            if twi and not unicodedata.is_normalized("NFC", twi):
                errors.append(f"Word '{key}': Twi text is not NFC normalized")

            # Digits not allowed in Twi
            if DIGIT_PATTERN.search(twi):
                errors.append(f"Word '{key}': Twi text '{twi}' contains digits (write numbers as Twi words)")

            # Enabled entry must have non-empty Twi
            if enabled and not twi.strip():
                errors.append(f"Word '{key}' is enabled but has empty Twi translation")

            # Audio override check
            audio_path_str = entry.get("audio")
            if audio_path_str:
                audio_path = Path(audio_path_str)
                if not audio_path.is_absolute():
                    audio_path = C.BASE_DIR / audio_path
                if not audio_path.exists():
                    errors.append(f"Word '{key}': Audio file not found: {audio_path_str}")
                else:
                    try:
                        with wave.open(str(audio_path), "rb") as w:
                            if w.getnchannels() != 1:
                                errors.append(f"Word '{key}': Audio must be mono (found {w.getnchannels()} channels)")
                    except Exception as e:
                        errors.append(f"Word '{key}': Invalid WAV file: {e}")

            # Status check
            status = entry.get("status")
            if status == "verified" and not entry.get("verified_by"):
                errors.append(f"Word '{key}' is marked verified but verified_by is empty")

        # 3. Phrase checks
        seen_phrase_ids: Set[str] = set()
        seen_phrase_matches: Dict[Tuple[str, ...], str] = {}

        for p in phrases:
            pid = p.get("id", "")
            if pid in seen_phrase_ids:
                errors.append(f"Duplicate phrase ID: '{pid}'")
            seen_phrase_ids.add(pid)

            match_tokens = tuple(p.get("match", []))
            if p.get("enabled", False):
                if match_tokens in seen_phrase_matches:
                    errors.append(f"Phrase collision: '{pid}' and '{seen_phrase_matches[match_tokens]}' have identical match tokens {match_tokens}")
                seen_phrase_matches[match_tokens] = pid

            twi = p.get("twi", "")
            if twi and not unicodedata.is_normalized("NFC", twi):
                errors.append(f"Phrase '{pid}': Twi text is not NFC normalized")
            if DIGIT_PATTERN.search(twi):
                errors.append(f"Phrase '{pid}': Twi text '{twi}' contains digits")
            if p.get("enabled", False) and not twi.strip():
                errors.append(f"Phrase '{pid}' is enabled but has empty Twi translation")

            audio_path_str = p.get("audio")
            if audio_path_str:
                audio_path = Path(audio_path_str)
                if not audio_path.is_absolute():
                    audio_path = C.BASE_DIR / audio_path
                if not audio_path.exists():
                    errors.append(f"Phrase '{pid}': Audio file not found: {audio_path_str}")
                else:
                    try:
                        with wave.open(str(audio_path), "rb") as w:
                            if w.getnchannels() != 1:
                                errors.append(f"Phrase '{pid}': Audio must be mono")
                    except Exception as e:
                        errors.append(f"Phrase '{pid}': Invalid WAV file: {e}")

        # 4. Pronounceability check (ghana-g2p)
        if check_pronounceable:
            try:
                from ghana_g2p import g2p
                for key, entry in words.items():
                    if entry.get("enabled", False) and entry.get("twi"):
                        twi_text = entry["twi"]
                        try:
                            phonemes = g2p(twi_text, "twi")
                            if not phonemes or not phonemes.strip():
                                errors.append(f"Word '{key}': Twi '{twi_text}' yielded empty phonemes in ghana-g2p")
                        except Exception as e:
                            errors.append(f"Word '{key}': Pronounceability failure for '{twi_text}': {e}")

                for p in phrases:
                    if p.get("enabled", False) and p.get("twi"):
                        twi_text = p["twi"]
                        try:
                            phonemes = g2p(twi_text, "twi")
                            if not phonemes or not phonemes.strip():
                                errors.append(f"Phrase '{p['id']}': Twi '{twi_text}' yielded empty phonemes in ghana-g2p")
                        except Exception as e:
                            errors.append(f"Phrase '{p['id']}': Pronounceability failure for '{twi_text}': {e}")
            except ImportError:
                errors.append("ghana-g2p is not installed; cannot perform pronounceability check")

        return errors


# --- CLI Commands -----------------------------------------------------------

def cmd_init(store: VocabularyStore, force: bool = False):
    if store.path.exists() and not force:
        print(f"Vocabulary file already exists at {store.path}. Use --force to re-scaffold.")
        return 1

    labels_file = C.LABELS_JSON
    if not labels_file.exists():
        print(f"Error: Labels file {labels_file} does not exist. Train the model first.")
        return 1

    labels = json.loads(labels_file.read_text())
    sign_labels = [l for l in labels if l != C.IDLE_LABEL]

    words = {}
    for label in sorted(sign_labels):
        english_display = label.replace("_", " ")
        words[label] = {
            "enabled": False,
            "english": english_display,
            "twi": "",
            "status": "unverified",
            "verified_by": None,
            "audio": None,
            "notes": "",
        }

    data = {
        "$schema": "./vocabulary.schema.json",
        "version": 1,
        "dialect": "",
        "defaults": {
            "voice": "auto",
            "unknown_word_policy": "skip",
        },
        "words": words,
        "phrases": [],
    }

    store.save(data)
    print(f"Scaffolded {len(words)} entries into {store.path} (all disabled with empty Twi).")
    return 0


def cmd_list(store: VocabularyStore):
    snap = store.get_snapshot()
    print(f"\n================ AMEGBE VOCABULARY ({len(snap.words)} words, {len(snap.phrases)} phrases) ================")
    print(f"Dialect: {snap.dialect or '(unset)'} | Default Voice: {snap.default_voice} | Unknown Policy: {snap.unknown_word_policy}")
    print("-" * 75)
    print(f"{'Type':<6s} {'Key / ID':<15s} {'English':<15s} {'Twi':<18s} {'Status':<10s} {'Enabled'}")
    print("-" * 75)

    for k, w in sorted(snap.words.items()):
        status = w.get("status", "unverified")
        if status == "verified" and w.get("verified_by"):
            status = f"verif ({w['verified_by']})"
        en_str = "YES" if w.get("enabled") else "NO"
        print(f"{'word':<6s} {k:<15s} {w.get('english', ''):<15s} {w.get('twi', ''):<18s} {status:<10s} {en_str}")

    for p in snap.phrases:
        status = p.get("status", "unverified")
        en_str = "YES" if p.get("enabled") else "NO"
        match_str = " ".join(p.get("match", []))
        print(f"{'phrase':<6s} {p.get('id', ''):<15s} {match_str:<15s} {p.get('twi', ''):<18s} {status:<10s} {en_str}")

    return 0


def cmd_add_word(store: VocabularyStore, gloss: str, twi: str, english: Optional[str] = None, notes: str = "", audio: Optional[str] = None):
    gloss = gloss.strip().lower()
    twi = unicodedata.normalize("NFC", twi.strip())
    data = dict(store.get_snapshot().raw_data)
    words = dict(data.get("words", {}))

    english = english or gloss.replace("_", " ")
    is_new = gloss not in words

    words[gloss] = {
        "enabled": bool(twi),
        "english": english,
        "twi": twi,
        "status": "unverified",
        "verified_by": None,
        "audio": audio,
        "notes": notes,
    }
    data["words"] = words
    store.save(data)

    print(f"Successfully {'added' if is_new else 'updated'} word: '{gloss}' -> '{twi}'")

    # Check if gloss is in model
    labels_file = C.LABELS_JSON
    in_model = False
    if labels_file.exists():
        labels = json.loads(labels_file.read_text())
        in_model = gloss in labels

    if not in_model:
        print(f"\n[NEXT STEPS] '{gloss}' is NOT recognized by the current model weights:")
        print(f"  1. Record 10+ webcam clips:  uv run python -m scripts.record_samples --label {gloss} --n 10")
        print(f"  2. Rebuild dataset:          uv run python -m asl.dataset_custom")
        print(f"  3. Retrain model:            uv run python -m asl.train_custom --seeds 3")
        print(f"  4. Restart server:           uv run uvicorn asl.server:app --port 8000")
    else:
        print("Model already knows this sign. Ready to use immediately (hot reload enabled).")
    return 0


def cmd_add_phrase(store: VocabularyStore, match_str: str, twi: str, phrase_id: Optional[str] = None, audio: Optional[str] = None):
    tokens = [t.lower() for t in match_str.strip().split() if t.strip()]
    if not tokens:
        print("Error: Phrase match cannot be empty.")
        return 1

    pid = phrase_id or "-".join(tokens)
    twi = unicodedata.normalize("NFC", twi.strip())
    data = dict(store.get_snapshot().raw_data)
    phrases = list(data.get("phrases", []))

    # Check existing phrase id
    phrases = [p for p in phrases if p.get("id") != pid]
    phrases.append({
        "id": pid,
        "match": tokens,
        "twi": twi,
        "enabled": bool(twi),
        "status": "unverified",
        "audio": audio,
    })
    data["phrases"] = phrases
    store.save(data)
    print(f"Successfully added phrase '{pid}' (match {tokens}) -> '{twi}'")
    return 0


def cmd_set_twi(store: VocabularyStore, gloss: str, twi: str):
    gloss = gloss.strip().lower()
    twi = unicodedata.normalize("NFC", twi.strip())
    data = dict(store.get_snapshot().raw_data)
    words = dict(data.get("words", {}))

    if gloss not in words:
        print(f"Error: Word '{gloss}' not in vocabulary. Use 'add' first.")
        return 1

    words[gloss]["twi"] = twi
    if twi and not words[gloss]["enabled"]:
        words[gloss]["enabled"] = True
    data["words"] = words
    store.save(data)
    print(f"Updated '{gloss}' Twi translation to '{twi}' (enabled={words[gloss]['enabled']})")
    return 0


def cmd_set_enabled(store: VocabularyStore, identifier: str, enabled: bool):
    identifier = identifier.strip().lower()
    data = dict(store.get_snapshot().raw_data)
    words = dict(data.get("words", {}))
    phrases = list(data.get("phrases", []))

    found = False
    if identifier in words:
        words[identifier]["enabled"] = enabled
        found = True
        print(f"Set word '{identifier}' enabled={enabled}")

    for p in phrases:
        if p.get("id") == identifier:
            p["enabled"] = enabled
            found = True
            print(f"Set phrase '{identifier}' enabled={enabled}")

    if not found:
        print(f"Error: Word or phrase '{identifier}' not found.")
        return 1

    data["words"] = words
    data["phrases"] = phrases
    store.save(data)
    return 0


def cmd_remove(store: VocabularyStore, gloss: str, purge_audio: bool = False):
    gloss = gloss.strip().lower()
    data = dict(store.get_snapshot().raw_data)
    words = dict(data.get("words", {}))
    phrases = list(data.get("phrases", []))

    found = False
    if gloss in words:
        del words[gloss]
        found = True
        print(f"Removed word '{gloss}' from vocabulary.")

    new_phrases = [p for p in phrases if p.get("id") != gloss]
    if len(new_phrases) != len(phrases):
        found = True
        print(f"Removed phrase '{gloss}' from vocabulary.")
    phrases = new_phrases

    if not found:
        print(f"Error: '{gloss}' not found in vocabulary.")
        return 1

    data["words"] = words
    data["phrases"] = phrases
    store.save(data)

    # Check if in model
    labels_file = C.LABELS_JSON
    in_model = False
    if labels_file.exists():
        labels = json.loads(labels_file.read_text())
        in_model = gloss in labels

    if in_model:
        print(f"\n[NOTE] '{gloss}' is still present in the trained model weights.")
        print("  Until retrained, the model will output this label, but Amegbe treats it as disabled.")
        print(f"  To permanently remove it from the recognizer:")
        print(f"  1. Delete video recordings:  rm -rf videos/{gloss}")
        print(f"  2. Rebuild dataset:          uv run python -m asl.dataset_custom")
        print(f"  3. Retrain model:            uv run python -m asl.train_custom --seeds 3")

    return 0


def cmd_verify(store: VocabularyStore, gloss: str, by_name: str):
    gloss = gloss.strip().lower()
    by_name = by_name.strip()
    if not by_name:
        print("Error: Must provide verifier name via --by")
        return 1

    data = dict(store.get_snapshot().raw_data)
    words = dict(data.get("words", {}))
    phrases = list(data.get("phrases", []))

    found = False
    if gloss in words:
        words[gloss]["status"] = "verified"
        words[gloss]["verified_by"] = by_name
        found = True
        print(f"Verified word '{gloss}' by {by_name}")

    for p in phrases:
        if p.get("id") == gloss:
            p["status"] = "verified"
            found = True
            print(f"Verified phrase '{gloss}' by {by_name}")

    if not found:
        print(f"Error: '{gloss}' not found.")
        return 1

    data["words"] = words
    data["phrases"] = phrases
    store.save(data)
    return 0


def cmd_export_review(store: VocabularyStore, out_path: Optional[Path] = None):
    snap = store.get_snapshot()
    out_path = out_path or (C.VOCAB_DIR / "review.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "type", "english", "twi", "status", "verified_by", "notes"])
        for k, w in sorted(snap.words.items()):
            writer.writerow([k, "word", w.get("english", ""), w.get("twi", ""), w.get("status", ""), w.get("verified_by", ""), w.get("notes", "")])
        for p in snap.phrases:
            match_str = " ".join(p.get("match", []))
            writer.writerow([p.get("id", ""), "phrase", match_str, p.get("twi", ""), p.get("status", ""), "", ""])

    print(f"Exported review CSV to {out_path} ({len(snap.words)} words, {len(snap.phrases)} phrases)")
    return 0


def cmd_validate(store: VocabularyStore):
    snap = store.get_snapshot()
    model_labels = None
    if C.LABELS_JSON.exists():
        model_labels = json.loads(C.LABELS_JSON.read_text())

    errors = store.validate_data(snap.raw_data, check_pronounceable=True, model_labels=model_labels)
    if errors:
        print(f"Validation FAILED ({len(errors)} errors):")
        for err in errors:
            print(f"  [!] {err}")
        return 1

    print(f"Validation PASSED ({len(snap.words)} words, {len(snap.phrases)} phrases).")
    return 0


def cmd_status(store: VocabularyStore):
    snap = store.get_snapshot()
    model_labels: Set[str] = set()
    if C.LABELS_JSON.exists():
        model_labels = set(json.loads(C.LABELS_JSON.read_text()))
    model_labels.discard(C.IDLE_LABEL)

    all_keys = sorted(set(snap.words.keys()) | model_labels)

    print("\n================ AMEGBE VOCABULARY STATUS ================")
    print(f"{'Gloss':<14s} {'In Model':<10s} {'Videos':<8s} {'Has Twi':<9s} {'Enabled':<9s} {'Verified':<10s} {'Audio':<8s} {'Notes'}")
    print("-" * 80)

    for k in all_keys:
        in_model = "YES" if k in model_labels else "NO"
        v_dir = C.CUSTOM_VIDEO_DIR / k
        v_count = len(list(v_dir.glob("*.mp4"))) if v_dir.exists() else 0

        w = snap.words.get(k)
        if w is None:
            print(f"{k:<14s} {in_model:<10s} {v_count:<8d} {'NO':<9s} {'NO':<9s} {'NO':<10s} {'NO':<8s} [Trained sign with no Twi entry!]")
            continue

        has_twi = "YES" if bool(w.get("twi")) else "NO"
        enabled = "YES" if w.get("enabled") else "NO"
        verified = "YES" if w.get("status") == "verified" else "NO"

        # Check cached audio
        try:
            from .tts import get_cache_key
            key = get_cache_key(w.get("twi", ""), voice=snap.default_voice, language="twi") if w.get("twi") else None
            has_audio = "YES" if (key and (C.TTS_CACHE_DIR / f"{key}.wav").exists()) or bool(w.get("audio")) else "NO"
        except (ImportError, ModuleNotFoundError):
            has_audio = "YES" if bool(w.get("audio")) else "NO"

        note = ""
        if in_model == "NO":
            note = "[Twi entry for sign NOT in model (needs videos + retrain)]"
        elif has_twi == "NO":
            note = "[Missing Twi text]"
        elif enabled == "NO":
            note = "[Disabled]"

        print(f"{k:<14s} {in_model:<10s} {v_count:<8d} {has_twi:<9s} {enabled:<9s} {verified:<10s} {has_audio:<8s} {note}")

    return 0


def cmd_audio_build(store: VocabularyStore, voice: Optional[str] = None, force: bool = False):
    from .tts import TwiTTS
    snap = store.get_snapshot()
    tts = TwiTTS(voice=voice or snap.default_voice)

    count = 0
    skipped = 0
    print(f"Building audio for enabled entries with voice='{tts.voice}'...")

    for k, w in sorted(snap.enabled_words.items()):
        twi = w.get("twi", "")
        if not twi:
            continue
        res = tts.speak(twi, voice=voice, language="twi")
        if res.get("cached") and not force:
            skipped += 1
        else:
            count += 1
        print(f"  word '{k}': {twi} -> {res.get('key', '')[:8]}.wav ({'cached' if res.get('cached') else 'synthesized'})")

    for p in snap.enabled_phrases:
        twi = p.get("twi", "")
        if not twi:
            continue
        res = tts.speak(twi, voice=voice, language="twi")
        if res.get("cached") and not force:
            skipped += 1
        else:
            count += 1
        print(f"  phrase '{p['id']}': {twi} -> {res.get('key', '')[:8]}.wav")

    print(f"\nAudio build complete: {count} generated, {skipped} already cached.")
    return 0


def cmd_audio_clean(store: VocabularyStore):
    try:
        from .tts import get_cache_key, TwiTTS
    except (ImportError, ModuleNotFoundError):
        print("TTS module not available.")
        return 1
    snap = store.get_snapshot()
    if not C.TTS_CACHE_DIR.exists():
        print("No cache directory found.")
        return 0

    # Collect all available voices so alternate-voice caches are preserved
    available_voices = {snap.default_voice, "twi-6", "twi-1"}
    try:
        tts = TwiTTS(start_worker=False)
        voices_info = tts.get_voices()
        if "voices" in voices_info:
            for v in voices_info["voices"]:
                if isinstance(v, dict) and "name" in v:
                    available_voices.add(v["name"])
        if "tiers" in voices_info:
            for tier_list in voices_info["tiers"].values():
                if isinstance(tier_list, list):
                    for vname in tier_list:
                        available_voices.add(vname)
    except Exception:
        pass

    valid_keys = set()
    for vname in available_voices:
        for w in snap.words.values():
            if w.get("twi"):
                valid_keys.add(f"{get_cache_key(w['twi'], voice=vname, language='twi')}.wav")
        for p in snap.phrases:
            if p.get("twi"):
                valid_keys.add(f"{get_cache_key(p['twi'], voice=vname, language='twi')}.wav")

    removed = 0
    for wav_file in C.TTS_CACHE_DIR.glob("*.wav"):
        if wav_file.name not in valid_keys:
            wav_file.unlink()
            removed += 1

    print(f"Cleaned {removed} orphaned audio files from {C.TTS_CACHE_DIR}.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Amegbe Vocabulary Management Tool")
    parser.add_argument("--vocab", type=Path, default=C.VOCAB_JSON, help="Path to vocabulary.json")
    sub = parser.add_subparsers(dest="command")

    # init
    p_init = sub.add_parser("init", help="Scaffold vocabulary.json from model labels")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing vocabulary.json")

    # list
    sub.add_parser("list", help="List all vocabulary entries")

    # add
    p_add = sub.add_parser("add", help="Add or update a word")
    p_add.add_argument("gloss", help="Sign label (e.g. water, thank_you)")
    p_add.add_argument("--twi", required=True, help="Twi translation")
    p_add.add_argument("--english", default=None, help="English display text")
    p_add.add_argument("--notes", default="", help="Notes")
    p_add.add_argument("--audio", default=None, help="Custom WAV audio file path")

    # add-phrase
    p_phrase = sub.add_parser("add-phrase", help="Add a phrase mapping")
    p_phrase.add_argument("match", help="Space-separated gloss tokens to match")
    p_phrase.add_argument("--twi", required=True, help="Twi translation")
    p_phrase.add_argument("--id", default=None, help="Phrase identifier")
    p_phrase.add_argument("--audio", default=None, help="Custom WAV audio file path")

    # set-twi
    p_set = sub.add_parser("set-twi", help="Update Twi translation for a word")
    p_set.add_argument("gloss", help="Sign label")
    p_set.add_argument("twi", help="New Twi translation")

    # enable / disable
    p_en = sub.add_parser("enable", help="Enable a word or phrase")
    p_en.add_argument("identifier")
    p_dis = sub.add_parser("disable", help="Disable a word or phrase")
    p_dis.add_argument("identifier")

    # remove
    p_rem = sub.add_parser("remove", help="Remove word or phrase")
    p_rem.add_argument("gloss")
    p_rem.add_argument("--purge-audio", action="store_true")

    # verify
    p_ver = sub.add_parser("verify", help="Mark entry as verified")
    p_ver.add_argument("gloss")
    p_ver.add_argument("--by", required=True, help="Verifier name")

    # export-review
    p_exp = sub.add_parser("export-review", help="Export review CSV")
    p_exp.add_argument("--out", type=Path, default=None)

    # validate
    sub.add_parser("validate", help="Validate vocabulary against schema and rules")

    # status
    sub.add_parser("status", help="Show vocabulary status matrix against model and video dataset")

    # speak
    p_spk = sub.add_parser("speak", help="Quick TTS synthesis test")
    p_spk.add_argument("text", help="Twi text to speak")
    p_spk.add_argument("--voice", default="auto")
    p_spk.add_argument("--out", type=Path, default=None)

    # audio
    p_audio = sub.add_parser("audio", help="Audio cache management")
    audio_sub = p_audio.add_subparsers(dest="audio_cmd")
    p_abuild = audio_sub.add_parser("build", help="Pre-synthesize audio for all enabled entries")
    p_abuild.add_argument("--voice", default=None)
    p_abuild.add_argument("--force", action="store_true")
    audio_sub.add_parser("clean", help="Clean orphaned audio files")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    store = VocabularyStore(path=args.vocab)

    if args.command == "init":
        return cmd_init(store, force=args.force)
    elif args.command == "list":
        return cmd_list(store)
    elif args.command == "add":
        return cmd_add_word(store, args.gloss, args.twi, english=args.english, notes=args.notes, audio=args.audio)
    elif args.command == "add-phrase":
        return cmd_add_phrase(store, args.match, args.twi, phrase_id=args.id, audio=args.audio)
    elif args.command == "set-twi":
        return cmd_set_twi(store, args.gloss, args.twi)
    elif args.command == "enable":
        return cmd_set_enabled(store, args.identifier, True)
    elif args.command == "disable":
        return cmd_set_enabled(store, args.identifier, False)
    elif args.command == "remove":
        return cmd_remove(store, args.gloss, purge_audio=args.purge_audio)
    elif args.command == "verify":
        return cmd_verify(store, args.gloss, args.by)
    elif args.command == "export-review":
        return cmd_export_review(store, out_path=args.out)
    elif args.command == "validate":
        return cmd_validate(store)
    elif args.command == "status":
        return cmd_status(store)
    elif args.command == "speak":
        from .tts import TwiTTS
        tts = TwiTTS(voice=args.voice)
        res = tts.speak(args.text, voice=args.voice)
        print(f"Synthesized '{args.text}': duration={res.get('duration_s', 0):.2f}s, path={res.get('path')}")
        if args.out:
            import shutil
            shutil.copy2(res["path"], args.out)
            print(f"Saved to {args.out}")
        return 0
    elif args.command == "audio":
        if args.audio_cmd == "build":
            return cmd_audio_build(store, voice=args.voice, force=args.force)
        elif args.audio_cmd == "clean":
            return cmd_audio_clean(store)
        else:
            p_audio.print_help()
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
