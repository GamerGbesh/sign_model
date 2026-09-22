# Amegbe: Real-Time Sign Language Recognition & Twi Speech Output

**Amegbe** connects real-time ASL sign recognition with deterministic Twi translation and speech synthesis using `ghananlpcommunity/stable-twi-tts`.

Based on the [SignSpeak](https://github.com/UnmannedArchive/signspeak) project by UnmannedArchive and contributors, Amegbe extends the custom-vocabulary sign recognizer into an accessible, bilingual assistive system for Ghanaian Twi.

---

## Key Features

- **Live Webcam Inference**: Sub-100ms real-time classification using MediaPipe Holistic landmarks and an optimized 2-layer LSTM.
- **Twi Speech Output**: Pure offline speech synthesis powered by Piper ONNX (`stable-twi-tts`), delivering Real-Time Factors (RTF) $< 0.05$ on standard CPU.
- **Dual Speech Modes**:
  - **Word Mode**: Speaks immediately as each sign is recognized (~45 ms latency).
  - **Sentence Mode**: Buffers signs and synthesizes fluent phrases upon a natural sign pause.
- **Auditable Vocabulary System**: Single-source-of-truth JSON dictionary (`vocab/vocabulary.json`) strictly validated with JSON Schema, supporting word definitions, greedy phrase matching, and human audio overrides.
- **Zero-Code Vocabulary CLI**: Tool-assisted workflows for adding words, validating orthography, exporting review sheets, and pre-rendering audio.
- **Native Browser Client**: Modular JavaScript client (`asl/static/amegbe-client.js`) featuring an `AudioQueue` with backpressure bounding and seamless autoplay handling.

---

## Quickstart

### 1. Requirements & Setup
Ensure you have Python 3.12 and [uv](https://github.com/astral-sh/uv) installed:
```bash
# Clone and enter the repository
cd sign_model

# Install dependencies and sync environment
uv sync
```

### 2. Launch the Application
Start the FastAPI server:
```bash
uv run uvicorn asl.server:app --host 0.0.0.0 --port 8000
```
Open your browser to:
```
http://localhost:8000/demo
```
Grant webcam permissions, click **Start Camera**, and begin signing.

---

## Vocabulary Management CLI

Amegbe provides a comprehensive CLI for managing vocabulary entries without touching Python code:

```bash
# List current vocabulary
uv run python -m asl.vocab list

# Show status matrix across model, recordings, and Twi translations
uv run python -m asl.vocab status

# Add a new word (unverified)
uv run python -m asl.vocab add water --twi "nsuo" --english "water"

# Add a multi-sign phrase rule
uv run python -m asl.vocab add-phrase "thank you" --twi "Medaase"

# Validate schema conformance, NFC normalization, and pronounceability
uv run python -m asl.vocab validate

# Export review CSV for native speaker verification
uv run python -m asl.vocab export-review --out review.csv

# Verify an entry after native speaker review
uv run python -m asl.vocab verify water --by "Dr. Mensah"

# Pre-synthesize and cache audio for all enabled entries
uv run python -m asl.vocab audio build

# Clean orphaned audio files from the cache
uv run python -m asl.vocab audio clean
```

For complete instructions, see the [Vocabulary Guide](docs/VOCABULARY.md).

---

## Configuration

Amegbe settings can be customized via environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `AMEGBE_SPEECH_MODE` | `word` | Initial speech mode (`word`, `sentence`, `off`). |
| `AMEGBE_TTS_VOICE` | `auto` | Default voice (`auto` resolves to `twi-6`). |
| `AMEGBE_TTS_LANGUAGE`| `twi` | Target synthesis language. |
| `AMEGBE_TTS_CACHE_DIR`| `data/tts_cache` | Path to cached WAV files. |
| `AMEGBE_SENTENCE_PAUSE_S` | `2.5` | Pause duration (seconds) triggering sentence synthesis. |
| `AMEGBE_WORD_QUEUE_MAX` | `3` | Maximum pending audio clips before dropping oldest. |
| `AMEGBE_UNKNOWN_WORD_POLICY` | `skip` | Handling of signs lacking Twi (`skip` or `english`). |

For speech architecture, latency benchmarks, and voice tiers, see the [Speech Documentation](docs/SPEECH.md).

---

## Verification & Testing

Run the automated test suite:
```bash
uv run pytest -q
```

Run the TTS benchmark:
```bash
uv run python -m scripts.bench_tts
```

---

## Attribution & License

- **SignSpeak Recognizer**: Based on the real-time sign recognition architecture developed by [UnmannedArchive/signspeak](https://github.com/UnmannedArchive/signspeak).
- **TTS Engine**: Built using [stable-twi-tts](https://github.com/Ghana-NLP/stable-twi-tts) by the Ghana NLP Community (MIT License).
- **Voice Weights**: CC-BY-NC-4.0 (Creative Commons NonCommercial).
- **Amegbe**: Open-source assistive research prototype.
