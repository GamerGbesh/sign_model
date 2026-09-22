# Amegbe Vocabulary Management Guide

Amegbe maintains `vocab/vocabulary.json` as the single, auditable source of truth linking recognized sign glosses to English text, Twi translations, and pre-recorded audio overrides. This ensures developers and linguists can manage vocabulary without editing Python code.

---

## 1. End-to-End: Adding a New Sign to Amegbe

Follow these 7 steps to introduce a brand-new sign from physical demonstration to spoken Twi:

### Step 1: Record Video Clips
Capture 10+ clips of yourself or signers performing the sign in front of a webcam:
```bash
uv run python -m scripts.record_samples --label water --n 10
```
Clips are saved to `videos/water/*.mp4`.

### Step 2: Extract Landmarks & Rebuild Dataset
Run the MediaPipe holistic landmarker across all recorded videos:
```bash
uv run python -m asl.dataset_custom
```
This processes landmark sequences into `data/processed_custom/`.

### Step 3: Retrain the Recognizer
Train the LSTM model across 3 seeds to update `models/sign_lstm.pt` and `models/labels.json`:
```bash
uv run python -m asl.train_custom --seeds 3
```

### Step 4: Add Vocabulary Entry via CLI
Register the gloss with its verified Twi translation:
```bash
uv run python -m asl.vocab add water --twi "nsuo" --english "water"
```
*(Note: Per Rule Zero, do not invent Twi translations. Only enter verified words or leave empty with `status: "unverified"` until reviewed).*

### Step 5: Validate Schema & Pronounceability
Verify JSON schema conformance, NFC normalization, absence of raw digits, and `ghana-g2p` pronounceability:
```bash
uv run python -m asl.vocab validate
```

### Step 6: Pre-Render Audio (Optional Cache Warming)
Synthesize and cache the native WAV audio:
```bash
uv run python -m asl.vocab audio build
```

### Step 7: Test Live in Browser
Start the server and test with your webcam:
```bash
uv run uvicorn asl.server:app --port 8000
```
Open `http://localhost:8000/demo` in your browser.

---

## 2. Schema Reference (`vocab/vocabulary.schema.json`)

The vocabulary file must conform to JSON Schema Draft 2020-12:

### Root Properties
- `$schema`: Path or URI of the schema file.
- `version`: Integer schema version (`1`).
- `dialect`: Target Twi dialect (e.g. `"asante"`, `"akuapem"`).
- `defaults`:
  - `voice`: Default TTS voice (e.g. `"auto"`, `"twi-6"`).
  - `unknown_word_policy`: How to handle words without Twi: `"skip"` or `"english"`.
- `words`: Mapping of sign glosses to word objects.
- `phrases`: Array of multi-token phrase objects.

### Word Entry Properties (`words.<gloss>`)
- `english` *(string, required)*: Display name in English.
- `twi` *(string, required)*: Twi translation. Must be NFC Unicode normalized. Cannot contain digits. If `enabled` is `true`, `twi` must not be empty.
- `enabled` *(boolean, required)*: Whether this word is active for translation and speech.
- `status` *(string, required)*: Either `"unverified"` or `"verified"`.
- `verified_by` *(string | null, optional)*: Name/identifier of the native speaker or linguist who verified the translation. Required if `status` is `"verified"`.
- `audio` *(string | null, optional)*: Path to a custom 16-bit mono 22050Hz WAV override.
- `notes` *(string, optional)*: Contextual, dialectal, or grammatical notes.

### Phrase Entry Properties (`phrases[]`)
- `id` *(string, required)*: Unique slug identifier (e.g. `"thank-you"`).
- `match` *(array of strings, required, minItems 2)*: Sequence of sign glosses to match greedily.
- `twi` *(string, required)*: Twi phrase translation. Must be NFC normalized without digits.
- `enabled` *(boolean, required)*: Whether this phrase rule is active.
- `status` *(string, optional)*: `"unverified"` or `"verified"`.
- `audio` *(string | null, optional)*: Custom audio override WAV path.

---

## 3. Native Speaker Verification Workflow

To guarantee translation accuracy and avoid invented words:

1. **Scaffold or Add Entries as Unverified**:
   New words default to `status: "unverified"`. The UI displays a clear `[unverified]` badge next to these words.
2. **Export Review Sheet**:
   Export a CSV for native speakers or linguists:
   ```bash
   uv run python -m asl.vocab export-review --out review.csv
   ```
3. **Review and Verify**:
   Once confirmed by a native speaker:
   ```bash
   uv run python -m asl.vocab verify water --by "Kofi Mensah"
   ```
   This marks `status: "verified"` and records `verified_by: "Kofi Mensah"`.

---

## 4. Human Audio Overrides

If a native speaker records natural spoken audio, you can bypass TTS synthesis entirely:

1. Record a 16-bit mono 22050 Hz PCM WAV file (e.g. `data/audio/water_native.wav`).
2. Attach it to the vocabulary entry:
   ```bash
   uv run python -m asl.vocab add water --twi "nsuo" --audio data/audio/water_native.wav
   ```
3. Amegbe will stream this WAV file directly on sign recognition instead of synthesizing audio.

---

## 5. Troubleshooting & FAQ

### Model recognizes a sign, but no speech is heard
- Run `uv run python -m asl.vocab status` to inspect the alignment matrix.
- If the sign has `Has Twi: NO` or `Enabled: NO`, add the Twi text and enable it:
  ```bash
  uv run python -m asl.vocab set-twi <gloss> "<twi_text>"
  uv run python -m asl.vocab enable <gloss>
  ```

### `ghana-g2p` reports unpronounceable characters
- Ensure the text contains no Arabic numerals (`0-9`). Write numbers as words (e.g. "baako" instead of "1").
- Ensure characters are standard Twi orthography (including `ɛ` and `ɔ`).
- Check that the string is NFC normalized (`unicodedata.normalize('NFC', text)`).

### Unknown Word Policies
Configured in `vocabulary.json` under `defaults.unknown_word_policy`:
- `"skip"` *(default)*: If a recognized sign has no enabled Twi translation, it is omitted from the spoken output and recorded in `untranslated`.
- `"english"`: The sign gloss is spoken/displayed in brackets `[like_this]` so the user knows a sign was performed.
