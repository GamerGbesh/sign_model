"""Shared constants and paths.

Everything that the data-prep, training, and live-inference stages need to agree
on lives here, so the three stages can never drift apart.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

WLASL_DIR = DATA_DIR / "wlasl"
WLASL_JSON = WLASL_DIR / "WLASL_v0.3.json"
WLASL_VIDEO_DIR = WLASL_DIR / "videos"
PROCESSED_DIR = DATA_DIR / "processed"

HOLISTIC_TASK = MODELS_DIR / "holistic_landmarker.task"
MODEL_WEIGHTS = MODELS_DIR / "sign_lstm.pt"
LABELS_JSON = MODELS_DIR / "labels.json"

CUSTOM_VIDEO_DIR = BASE_DIR / "videos"
HOLDOUT_VIDEO_DIR = BASE_DIR / "videos_test"
CACHE_DIR = DATA_DIR / "cache"
CUSTOM_PROCESSED = DATA_DIR / "processed_custom"
IDLE_LABEL = "idle"
WINDOW_S = 3.0  # seconds of signal per model input; finalized from p90 (2.74s + 0.4s = 3.0s)
INFER_STRIDE_MS = 100  # classify at most this often (stream time)

# --- Speech & Vocabulary (Amegbe) ------------------------------------------
import os

VOCAB_DIR = BASE_DIR / "vocab"
VOCAB_JSON = VOCAB_DIR / "vocabulary.json"
VOCAB_SCHEMA = VOCAB_DIR / "vocabulary.schema.json"
TTS_CACHE_DIR = Path(os.environ.get("AMEGBE_TTS_CACHE_DIR", str(DATA_DIR / "tts_cache")))

SPEECH_MODE = os.environ.get("AMEGBE_SPEECH_MODE", "word")  # "word" | "sentence" | "off"
TTS_VOICE = os.environ.get("AMEGBE_TTS_VOICE", "auto")
TTS_LANGUAGE = os.environ.get("AMEGBE_TTS_LANGUAGE", "twi")
TTS_MODEL_DIR = os.environ.get("AMEGBE_TTS_MODEL_DIR", None)
UNKNOWN_WORD_POLICY = os.environ.get("AMEGBE_UNKNOWN_WORD_POLICY", "skip")
WORD_QUEUE_MAX = int(os.environ.get("AMEGBE_WORD_QUEUE_MAX", "3"))
SENTENCE_PAUSE_S = float(os.environ.get("AMEGBE_SENTENCE_PAUSE_S", "2.5"))

# --- Feature layout --------------------------------------------------------
# Per frame: pose (33 x [x,y,z,visibility]) + left hand (21 x [x,y,z]) +
# right hand (21 x [x,y,z]). Face landmarks are omitted in v1.
NUM_POSE = 33
NUM_HAND = 21
POSE_DIM = NUM_POSE * 4          # 132
LH_DIM = NUM_HAND * 3            # 63
RH_DIM = NUM_HAND * 3            # 63
FEATURE_DIM = POSE_DIM + LH_DIM + RH_DIM  # 258

# Feature-vector slices (handy for tests and debugging).
POSE_SLICE = slice(0, POSE_DIM)
LH_SLICE = slice(POSE_DIM, POSE_DIM + LH_DIM)
RH_SLICE = slice(POSE_DIM + LH_DIM, FEATURE_DIM)

# Pose landmark indices we anchor normalization on (MediaPipe pose model).
L_SHOULDER = 11
R_SHOULDER = 12

# --- Sequence / model ------------------------------------------------------
SEQ_LEN = 32            # frames per clip fed to the model
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3

# --- Live inference --------------------------------------------------------
CONF_THRESHOLD = 0.6    # minimum softmax prob to show a prediction
DEBOUNCE_FRAMES = 6     # consecutive consistent predictions before committing
MIN_SAMPLES_PER_GLOSS = 5  # drop glosses with fewer available videos than this

# --- Phase 2: gloss -> English (Claude) ------------------------------------
# Default to Opus 4.8. For lower latency on this short task you can switch to
# "claude-haiku-4-5". The translator falls back to a plain word-join when no
# ANTHROPIC_API_KEY is set, so the demo still runs offline.
LLM_MODEL = "claude-opus-4-8"
FINALIZE_PAUSE_S = 2.5  # seconds of no new sign before a sentence is finalized

# --- Vocabulary ------------------------------------------------------------
# A curated set of common, visually distinct WLASL glosses for v1. dataset.py
# intersects this with what actually exists in WLASL_v0.3.json and what videos
# are present on disk, so unavailable words are skipped automatically.
GLOSSES = [
    "computer", "drink", "deaf", "hearing", "who", "candy", "finish", "wrong",
    "before", "no", "hot", "like", "now", "orange", "kiss", "white", "fine", "yes",
]
