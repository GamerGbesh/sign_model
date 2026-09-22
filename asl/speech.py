"""Speech orchestrator coordinating recognizer commits, vocabulary translation, and TTS output.

Supports:
- Runtime modes: 'word', 'sentence', 'off'
- Event emission: 'speech', 'speech_error', 'phrase'
- Pause detection in sentence mode via tick(t_s)
- Queue backpressure in word mode (max pending bound)
- Dynamic voice selection and vocabulary reload awareness
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from . import config as C
from .twi_translate import translate_glosses
from .vocab import VocabularyStore
from .tts import TwiTTS

logger = logging.getLogger(__name__)


class SpeechOrchestrator:
    def __init__(
        self,
        vocab_store: VocabularyStore,
        tts: TwiTTS,
        mode: str = C.SPEECH_MODE,
        pause_s: float = C.SENTENCE_PAUSE_S,
        voice: Optional[str] = None,
        language: str = C.TTS_LANGUAGE,
        word_queue_max: int = C.WORD_QUEUE_MAX,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.vocab_store = vocab_store
        self.tts = tts
        self.mode = mode if mode in ("word", "sentence", "off") else "word"
        self.pause_s = pause_s
        self.voice = voice or C.TTS_VOICE
        self.language = language
        self.word_queue_max = word_queue_max
        self.on_event = on_event

        self._buffer: List[str] = []
        self._last_commit_time_s: Optional[float] = None
        self._pending_word_queue: List[Dict[str, Any]] = []

    def set_mode(self, mode: str) -> None:
        """Switches runtime speech mode ('word', 'sentence', 'off')."""
        if mode not in ("word", "sentence", "off"):
            raise ValueError(f"Invalid speech mode: {mode}. Must be 'word', 'sentence', or 'off'.")
        self.mode = mode
        logger.info("SpeechOrchestrator mode set to %s", mode)

    def set_voice(self, voice: Optional[str]) -> None:
        """Sets active TTS voice."""
        self.voice = voice or C.TTS_VOICE

    def clear(self) -> None:
        """Clears buffered words and pending word events."""
        self._buffer.clear()
        self._last_commit_time_s = None
        self._pending_word_queue.clear()

    def ack_audio(self, key: str) -> None:
        """Client acknowledgement that an audio item finished playing."""
        self._pending_word_queue = [ev for ev in self._pending_word_queue if ev.get("audio_key") != key]

    @property
    def pending_count(self) -> int:
        return len(self._pending_word_queue)

    def _emit(self, event: Dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception as e:
                logger.exception("Error in SpeechOrchestrator on_event: %s", e)

    def on_commit(self, gloss: str, t_s: Optional[float] = None) -> List[Dict[str, Any]]:
        """Handles a newly recognized sign commit.
        
        Returns a list of emitted events.
        """
        now = t_s if t_s is not None else time.time()
        self._last_commit_time_s = now

        snap = self.vocab_store.get_snapshot()

        # Intersect with enabled glosses
        norm_gloss = gloss.strip().lower()
        is_enabled = (
            norm_gloss in snap.enabled_glosses
            or norm_gloss.replace(" ", "_") in snap.enabled_glosses
            or norm_gloss.replace("_", " ") in snap.enabled_glosses
        )
        if not is_enabled:
            logger.debug("Gloss %r not enabled in vocabulary; ignoring", gloss)
            return []

        if self.mode == "off":
            return []

        if self.mode == "word":
            res = translate_glosses([gloss], snap)
            if not res.items or not res.items[0].twi:
                return []

            item = res.items[0]
            try:
                audio_res = self.tts.speak(
                    item.twi,
                    voice=self.voice,
                    language=self.language,
                    audio_override=item.audio,
                )
            except Exception as e:
                err_ev = {
                    "type": "speech_error",
                    "gloss": gloss,
                    "error": str(e),
                }
                self._emit(err_ev)
                return [err_ev]

            if not audio_res:
                err_ev = {
                    "type": "speech_error",
                    "gloss": gloss,
                    "error": f"Synthesis returned no audio for {item.twi!r}",
                }
                self._emit(err_ev)
                return [err_ev]

            event = {
                "type": "speech",
                "audio_url": f"/audio/{audio_res.key}.wav",
                "audio_key": audio_res.key,
                "twi": item.twi,
                "english": item.english,
                "duration_s": audio_res.duration_s,
                "mode": "word",
                "is_sentence": False,
                "verified": item.verified,
            }

            # Backpressure: limit pending to word_queue_max
            if len(self._pending_word_queue) >= self.word_queue_max:
                dropped = self._pending_word_queue.pop(0)
                logger.warning("Word queue full (max=%d). Dropped oldest pending audio: %s", self.word_queue_max, dropped.get("twi"))

            self._pending_word_queue.append(event)
            self._emit(event)
            return [event]

        elif self.mode == "sentence":
            self._buffer.append(gloss)
            return []

        return []

    def tick(self, t_s: Optional[float] = None) -> List[Dict[str, Any]]:
        """Periodic check for sentence finalization on pause threshold."""
        if self.mode != "sentence" or not self._buffer or self._last_commit_time_s is None:
            return []

        now = t_s if t_s is not None else time.time()
        if now - self._last_commit_time_s >= self.pause_s:
            return self.speak_now()

        return []

    def speak_now(self) -> List[Dict[str, Any]]:
        """Forces immediate finalization of buffered sentence."""
        if not self._buffer:
            return []

        glosses = list(self._buffer)
        self._buffer.clear()
        self._last_commit_time_s = None

        if self.mode == "off":
            return []

        snap = self.vocab_store.get_snapshot()
        res = translate_glosses(glosses, snap)
        if not res.twi_text:
            return []

        # Check single-phrase audio override
        audio_override = None
        if len(res.items) == 1 and res.items[0].rule_type == "phrase":
            audio_override = res.items[0].audio

        try:
            audio_res = self.tts.speak(
                res.twi_text,
                voice=self.voice,
                language=self.language,
                audio_override=audio_override,
            )
        except Exception as e:
            err_ev = {
                "type": "speech_error",
                "gloss": " ".join(glosses),
                "error": str(e),
            }
            self._emit(err_ev)
            return [err_ev]

        if not audio_res:
            err_ev = {
                "type": "speech_error",
                "gloss": " ".join(glosses),
                "error": f"Synthesis returned no audio for {res.twi_text!r}",
            }
            self._emit(err_ev)
            return [err_ev]

        all_verified = all(it.verified for it in res.items) if res.items else False
        event = {
            "type": "speech",
            "audio_url": f"/audio/{audio_res.key}.wav",
            "audio_key": audio_res.key,
            "twi": res.twi_text,
            "english": res.english_text,
            "duration_s": audio_res.duration_s,
            "mode": "sentence",
            "is_sentence": True,
            "verified": all_verified,
        }
        self._emit(event)
        return [event]

    def enrich_phrase_event(self, phrase_event: Dict[str, Any]) -> Dict[str, Any]:
        """Adds Twi translation to an existing recognizer phrase event."""
        words = phrase_event.get("words", [])
        snap = self.vocab_store.get_snapshot()
        res = translate_glosses(words, snap)
        enriched = dict(phrase_event)
        enriched["twi"] = res.twi_text
        enriched["twi_verified"] = all(it.verified for it in res.items) if res.items else False
        return enriched
