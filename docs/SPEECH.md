# Amegbe Speech Engine Architecture & Reference

Amegbe integrates `ghananlpcommunity/stable-twi-tts` (Piper ONNX) with real-time ASL sign recognition to generate natural, low-latency Twi speech.

---

## 1. System Architecture

```text
[Webcam Stream (Browser)]
           │  Binary frames (JPEG + timestamp)
           ▼
[FastAPI /ws/recognize]
           │  Raw frames
           ▼
[StreamingRecognizer (SignLSTM)]
           │  on_commit(gloss)
           ▼
[SpeechOrchestrator] ──► [twi_translate] (Greedy phrase & word lookup)
           │
           ▼
     [TwiTTS Engine]
     ├── Audio Override Check (custom .wav)
     ├── File Cache (data/tts_cache/<sha1>.wav)
     └── Piper ONNX Synthesizer (stable-twi-tts)
           │
           ▼
[Audio Stream /audio/<sha1>.wav] ──► [AmegbeClient / AudioQueue]
```

---

## 2. Speech Modes & Tradeoffs

Amegbe supports two primary synthesis modes, switchable at runtime without restarting:

| Mode | Latency | Prosody & Grammar | Use Case |
| :--- | :--- | :--- | :--- |
| **Word-by-Word (`word`)** | **~45 ms** per sign | Mechanical (staccato) | Best for immediate real-time feedback, spellings, or isolated signs. |
| **Sentence (`sentence`)** | **~90 ms** after pause | Natural phrase prosody | Best for multi-sign concepts (e.g. `thank` + `you` $\implies$ `Medaase`). |
| **Silent (`off`)** | N/A | None | Displays sign and Twi text visually without audio output. |

### Sentence Mode Dynamics
In sentence mode, glosses accumulate in a buffer. The sentence is synthesized when:
1. **Pause Detection**: No new sign is committed for `SENTENCE_PAUSE_S` (default: 2.5 seconds).
2. **Explicit Trigger**: The user or UI clicks the **"Speak Now"** button (`{"type": "speak"}`).

---

## 3. WebSocket Protocol Additions

The `/ws/recognize` WebSocket endpoint accepts and emits the following messages:

### Client -> Server Messages

#### 1. Configure Speech Mode & Voice
```json
{
  "type": "config",
  "speech_mode": "sentence", // "word" | "sentence" | "off"
  "voice": "twi-6"           // voice name or "auto"
}
```

#### 2. Speak Buffered Sentence Now
```json
{
  "type": "speak"
}
```

#### 3. Clear Sentence Buffer
```json
{
  "type": "clear"
}
```

#### 4. Audio Playback Acknowledgment (Backpressure)
```json
{
  "type": "ack_audio",
  "key": "5a4df0688a220b3cb86e80b2a752251a89b8849b"
}
```

### Server -> Client Messages

#### 1. Ready Handshake
```json
{
  "type": "ready",
  "app": "amegbe",
  "labels": ["friend", "hello", ...],
  "speech_mode": "word",
  "voice": "auto"
}
```

#### 2. Speech Audio Event
Emitted whenever audio has been synthesized or retrieved from cache:
```json
{
  "type": "speech",
  "audio_url": "/audio/5a4df0688a220b3cb86e80b2a752251a89b8849b.wav",
  "audio_key": "5a4df0688a220b3cb86e80b2a752251a89b8849b",
  "twi": "Akwaaba",
  "english": "hello",
  "duration_s": 0.37,
  "mode": "word",
  "is_sentence": false,
  "verified": true
}
```

#### 3. Speech Error Event
Emitted if synthesis fails on an unpronounceable token:
```json
{
  "type": "speech_error",
  "gloss": "123",
  "error": "ValueError: no pronounceable phonemes in '123'"
}
```

---

## 4. Fully Offline Operation

Amegbe requires zero internet access during inference:
1. **Pre-Downloaded ONNX Weights**: The 80.28 MB model is stored locally in `~/.cache/stable-twi-tts/model-v0.1.0`.
2. **Deterministic File Cache**: Pre-rendered or dynamically rendered audio files are saved as 16-bit mono 22050Hz WAV in `data/tts_cache/<sha1>.wav`.
3. **Sub-Millisecond Cache Hits**: Average cache lookup latency is **0.028 ms**, eliminating all model compute on repeated signs.

To warm the entire vocabulary cache ahead of time:
```bash
uv run python -m asl.vocab audio build
```

---

## 5. Voice Tiers & Quality

The model package includes 12 speaker voices evaluated in `voices.json`:

| Voice | Tier | Language | Twi-Only UER | Codeswitch UER | Recommended For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`twi-6`** *(default)* | `twi_only` | Twi | **26.8%** | 62.4% | Pure Twi vocabulary and phrases. |
| **`twi-1`** | `codeswitch` | Twi/English | 33.5% | **59.8%** | Mixed sentences or English fallback borrowings. |
| `twi-2` | `codeswitch` | Twi/English | 29.1% | 60.7% | Alternative speaker tone. |
| `twi-11` | `twi_only` | Twi | 28.5% | 63.1% | Fast, clear pronunciation. |

---

## 6. Licensing & Open-Source Attribution

- **Stable-Twi-TTS Engine Code**: Licensed under the **MIT License** by Ghana NLP Community.
- **Published Voice Weights (12 voices)**: Licensed under **CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial 4.0 International). Suitable for academic, personal, research, and non-commercial prototyping.
- **Phonemizer & Text Processing**:
  - Twi G2P uses `ghana-g2p` (MIT License).
  - The optional `eng` extra requires `espeak-ng` (GPL-3.0); Amegbe defaults strictly to `stable-twi-tts[twi]` (MIT) to maintain a permissive, clean license surface.
