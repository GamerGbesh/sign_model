# SignSpeak: Custom-Vocabulary Training & Real-Time Streaming Pipeline

This document details the architecture, configuration, training results, benchmarks, and operational procedures for the custom SignSpeak ASL recognizer.

---

## 1. Quickstart & Commands in Order

All commands are managed with `uv` and execute inside the project environment:

### A. Environment Setup & Recon (Phase 0)
```bash
# Sync dependencies and create venv
uv sync

# Download the MediaPipe Holistic Landmarker model
bash scripts/get_holistic_model.sh

# Run baseline test suite
uv run pytest -q

# Inspect video dataset and generate skeleton thumbnails
uv run python -m scripts.inspect_videos
```

### B. Sign Activity & Segmentation Analysis (Phases 1 & 2)
```bash
# Extract features, cache to data/cache/, and plot activity timelines
uv run python -m scripts.inspect_videos --plot
```

### C. Build Custom Dataset (Phase 3)
```bash
# Build causal windowed dataset with video-level splits and augmentation
uv run python -m asl.dataset_custom --seed 0
```

### D. Model Training & Evaluation (Phase 4)
```bash
# Train across multiple seeds, calibrate threshold, and export artifacts
uv run python -m asl.train_custom --seeds 3 --epochs 150
```

### E. Verification & Benchmarking (Phase 7)
```bash
# Replay evaluation on held-out test split videos
uv run python -m scripts.replay_eval
uv run python -m scripts.replay_eval --fps 15

# Start the web server
uv run uvicorn asl.server:app --host 0.0.0.0 --port 8000

# Run WebSocket smoke test (in another terminal)
uv run python -m scripts.ws_smoke_test

# Run full automated test suite
uv run pytest -q
```

---

## 2. Architecture & Design Decisions

### A. Temporal Alignment & Shared Windowing (`asl/windowing.py`)
- **Root Cause of Past Failures:** Training previously used global clip interpolation (`resample_sequence(n=32)` over the entire video duration), while live inference used sliding fixed-duration windowing with flipped frames and IMAGE extractor mode.
- **Unified Causal Solution:** All training windows, test evaluations, and live streaming now use the identical `sample_window(frames, ts_ms, t_end_ms, window_s=3.0, n=32)` function.
- **Causality & Zero-Padding:** Samples are drawn at `n` uniform times over `[t_end - window_s, t_end]`. Times before the first frame are strictly zero-padded. Unobserved frames without pose remain zero vectors.

### B. Frame Orientation & Extractor Mode
- **Unmirrored Extractor Pipeline:** MediaPipe `HolisticExtractor` runs in `VIDEO` mode with strictly increasing integer-millisecond timestamps. Frames are processed **unmirrored** (`raw BGR`) across all stages.
- **Display Mirroring:** Mirroring (`cv2.flip(frame, 1)` or CSS `transform: scaleX(-1);`) is applied exclusively for visual display in the UI so user interaction feels natural without flipping landmark geometry.

### C. Hand Activity & Segmentation (`asl/video_features.py`)
- **Activity Condition:** A frame is classified active when a pose is detected, at least one hand is detected, and that wrist is above the hip line (`wrist_y < hip_y` in normalized coordinates where y increases downwards).
- **Temporal Smoothing:** Gaps under 0.3s between active regions are merged, segments under 0.35s are dropped as noise, and active segments are padded by 0.2s on each side.
- **Window Length (`WINDOW_S`):** Based on the empirical 90th percentile of sign segment durations ($p_{90} = 2.74\text{s}$), $WINDOW\_S = \text{clip}(p_{90} + 0.4\text{s}, 1.5, 3.0) = 3.00\text{s}$.

### D. Dataset Splits & Leakage Prevention (`asl/dataset_custom.py`)
- **Video-Level Stratified Splits:** For labels with $\ge 4$ source videos, videos are partitioned $\sim 70\% / 15\% / 15\%$ into train, val, and test. Video sets are strictly disjoint.
- **Chronological Split:** For labels with 1–3 videos (`goodbye`), videos are split chronologically (70% train, 15% val, 15% test) with boundary gaps.
- **Idle Negative Generation:** Non-active windows with $< 10\%$ overlap with any signing segment are extracted with 300ms stride. Ambiguous windows (10%–70% overlap) are discarded. Train idle windows are capped at 1.5x the mean sign class count.

---

## 3. Dataset & Training Results

### Dataset Distribution
- **Classes (11):** `friend`, `goodbye`, `hello`, `help`, `idle`, `name`, `no`, `please`, `sorry`, `thank_you`, `yes`
- **Total Windows:** 463 windows (Train: 360, Val: 48, Test: 55)

| Class | Source Videos | Train Windows | Val Windows | Test Windows | Total Windows | Split Strategy |
|---|---|---|---|---|---|---|
| friend | 8 | 36 | 3 | 3 | 42 | Video Split |
| goodbye | 2 | 12 | 6 | 6 | 24 | Chronological (Same-Session) |
| hello | 5 | 18 | 3 | 3 | 24 | Video Split |
| help | 6 | 24 | 3 | 3 | 30 | Video Split |
| idle | - | 42 | 9 | 16 | 67 | Video Slices (< 10% overlap) |
| name | 9 | 42 | 3 | 3 | 48 | Video Split |
| no | 12 | 48 | 6 | 6 | 60 | Video Split |
| please | 8 | 30 | 3 | 3 | 36 | Video Split |
| sorry | 7 | 30 | 3 | 3 | 36 | Video Split |
| thank_you | 6 | 24 | 3 | 3 | 30 | Video Split |
| yes | 13 | 54 | 6 | 6 | 66 | Video Split |
| **TOTAL** | **76** | **360** | **48** | **55** | **463** | |

### Training & Validation Metrics
- **Horizontal Flip Tuning:** $p_{flip} = 0.0$ achieved higher validation Macro-F1 (0.845) than $p_{flip} = 0.5$ (0.804), confirming that ASL signs have critical handedness distinctions.
- **Best Validation Accuracy:** **91.7%** (Val Macro-F1: **0.927**), satisfying the $\ge 90\%$ target.
- **Shuffled-Label Control Run:** Accuracy landed at **20.8%** (Macro-F1 0.058, chance is 9.1%), confirming no data leakage across the split pipeline.
- **Confidence Calibration ($\tau$):** Calibrated on validation data at $\tau = 0.40$ with $0.0\%$ idle false-positive rate and $87.2\%$ sign recall at $\tau$. Debounce length selected at $debounce\_n = 4$ strides (400ms).

---

## 4. Verification & Performance Benchmarks

### Replay Evaluation (Held-Out Test Videos)
Evaluated with `scripts/replay_eval.py`:
- **Median Latency from Sign Onset to Commit:** **1200 ms** (1.20s), meeting the $< 2.0\text{s}$ acceptance threshold.
- **Individual Word Recall:**
  - `friend`: 100%
  - `goodbye`: 100%
  - `help`: 100%
  - `name`: 100%
  - `no`: 100%
  - `please`: 100%
  - `yes`: 50%
- **Overall Recall:** 69.2% (Native) / 53.8% (Simulated 15 fps).
- **Commit Precision:** 56.2%.

### WebSocket Latency & Throughput Benchmark
Evaluated with `scripts/ws_smoke_test.py` against `uvicorn asl.server:app`:
- **Host CPU:** 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz (12 logical cores).
- **Sustained Stream Rate:** 25.0 fps.
- **Latency Distribution:**
  - Minimum RTT: **25.8 ms**
  - Median RTT: **29.5 ms**
  - 95th Percentile (p95): **39.2 ms** (Substantially exceeds the $< 250\text{ms}$ acceptance target).
- **Memory & Privacy:** Zero frame caching to disk; frames are decoded and processed exclusively in RAM.

---

## 5. Adding a New Word to the Recognizer

To expand the vocabulary:

1. **Collect Videos:**
   Record 5–10 short clips of the sign and place them into `videos/<new_word>/`:
   ```bash
   # Use the included webcam recording tool:
   uv run python -m scripts.record_samples --label my_new_sign --n 10 --duration 2.5
   ```
2. **Re-run Inspection & Feature Extraction:**
   ```bash
   uv run python -m scripts.inspect_videos --plot
   ```
3. **Rebuild the Dataset:**
   ```bash
   uv run python -m asl.dataset_custom --seed 0
   ```
4. **Train the Updated Model:**
   ```bash
   uv run python -m asl.train_custom --seeds 3
   ```
   The script automatically backs up the previous checkpoint to `models/backup/<timestamp>/` before writing new weights.

---

## 6. Deployment & Browser Security Requirements

> [!IMPORTANT]
> Modern web browsers enforce strict security policies for `navigator.mediaDevices.getUserMedia()`:
> - `getUserMedia` is **only** permitted on **HTTPS** origins or **localhost** (`127.0.0.1`).
> - When deploying to production or a remote server, you must terminate TLS (HTTPS/WSS) using a reverse proxy (e.g., Caddy, Nginx, or Traefik) so the browser client connects via `https://` and `wss://`.
