"""Transport-agnostic real-time streaming sign recognition engine.

Maintains a ring buffer of landmark vectors and timestamps, performs causal
window sampling, evaluates SignLSTM, handles debounce streaks, gates on hand presence,
manages repetition cooldowns, and yields recognized word commits.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from . import config as C
from .landmarks import HolisticExtractor, arrays_from_result
from .model import SignLSTM, load_model
from .windowing import sample_window


class StreamingRecognizer:
    def __init__(
        self,
        model_dir: Path = C.MODELS_DIR,
        device: str = "cpu",
        extractor: Optional[Any] = None,
        model: Optional[SignLSTM] = None,
        labels: Optional[List[str]] = None,
        conf_threshold: Optional[float] = None,
        debounce_n: Optional[int] = None,
        window_s: Optional[float] = None,
        infer_stride_ms: Optional[int] = None,
    ):
        self.model_dir = Path(model_dir)
        self.device = device

        # Limit PyTorch inference threads for responsive streaming
        torch.set_num_threads(2)

        # Load metadata if present
        meta_file = self.model_dir / "model_meta.json"
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                pass

        self.window_s = float(window_s or meta.get("window_s", C.WINDOW_S))
        self.infer_stride_ms = int(infer_stride_ms or meta.get("infer_stride_ms", C.INFER_STRIDE_MS))
        self.conf_threshold = float(conf_threshold or meta.get("conf_threshold", 0.40))
        self.debounce_n = int(debounce_n or meta.get("debounce_n", 4))
        self.idle_label = str(meta.get("idle_label", C.IDLE_LABEL))

        # Model and labels
        if model is not None:
            self.model = model
            self.labels = list(labels or meta.get("labels", []))
        else:
            weights_path = self.model_dir / "sign_lstm.pt"
            labels_path = self.model_dir / "labels.json"
            self.model, self.labels = load_model(weights_path=weights_path, labels_path=labels_path, device=device)

        self.model.to(device).eval()

        # Extractor
        if extractor is not None:
            self.extractor = extractor
            self._owns_extractor = False
        else:
            self.extractor = HolisticExtractor(running_mode="VIDEO")
            self._owns_extractor = True

        # Buffer: entries are (t_ms, vec, hand_present)
        # Ring buffer spans window_s + 0.5s
        self.buffer_duration_ms = (self.window_s + 0.5) * 1000.0
        self.buffer: deque[Tuple[int, np.ndarray, bool]] = deque()

        # State tracking
        self.last_t_ms: Optional[int] = None
        self.last_infer_t_ms: Optional[int] = None

        # Debounce & Cooldown state
        self.streak_label: Optional[str] = None
        self.streak_count: int = 0
        self.last_committed_word: Optional[str] = None
        self.cooldown_satisfied: bool = True
        self.idle_stretch_ms: float = 0.0

        # Latest state for non-stride frames
        self.latest_result: Dict[str, Any] = {
            "t": 0,
            "label": self.idle_label,
            "conf": 1.0,
            "topk": [(self.idle_label, 1.0)],
            "hands": False,
            "warming_up": True,
            "commit": None,
            "phrase": [],
        }
        self.phrase: List[str] = []

    def reset(self) -> None:
        """Reset internal buffer and streaming state."""
        self.buffer.clear()
        self.last_t_ms = None
        self.last_infer_t_ms = None
        self.streak_label = None
        self.streak_count = 0
        self.last_committed_word = None
        self.cooldown_satisfied = True
        self.idle_stretch_ms = 0.0
        self.phrase.clear()
        self.latest_result = {
            "t": 0,
            "label": self.idle_label,
            "conf": 1.0,
            "topk": [(self.idle_label, 1.0)],
            "hands": False,
            "warming_up": True,
            "commit": None,
            "phrase": [],
        }

    def close(self) -> None:
        """Release extractor resources."""
        if self._owns_extractor and hasattr(self.extractor, "close"):
            self.extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def push(self, frame_bgr: np.ndarray, t_ms: int) -> Dict[str, Any]:
        """Process incoming video frame.

        Args:
            frame_bgr: raw unmirrored BGR frame (H, W, 3).
            t_ms: frame timestamp in milliseconds.

        Returns:
            Dict containing {t, label, conf, topk, hands, warming_up, commit, phrase}
        """
        # Force strictly increasing timestamps: drop out-of-order, bump equal
        if self.last_t_ms is not None:
            if t_ms < self.last_t_ms:
                # Out-of-order frame: drop
                res = dict(self.latest_result)
                res["t"] = t_ms
                res["commit"] = None
                return res
            elif t_ms == self.last_t_ms:
                t_ms = self.last_t_ms + 1

        dt_ms = (t_ms - self.last_t_ms) if self.last_t_ms is not None else self.infer_stride_ms
        self.last_t_ms = t_ms

        # Extract landmark features
        vec, result, pose_present = self.extractor(frame_bgr, timestamp_ms=t_ms)

        # Check hand presence
        if result is not None and hasattr(result, "left_hand_landmarks"):
            _, _, _, _, lh_p, rh_p = arrays_from_result(result)
            hand_present = bool(lh_p or rh_p)
        else:
            lh_p = np.any(vec[C.LH_SLICE] != 0.0)
            rh_p = np.any(vec[C.RH_SLICE] != 0.0)
            hand_present = bool(lh_p or rh_p)

        # Update ring buffer
        self.buffer.append((t_ms, vec, hand_present))
        while self.buffer and (t_ms - self.buffer[0][0]) > self.buffer_duration_ms:
            self.buffer.popleft()

        # Check stride cadence
        should_classify = False
        if self.last_infer_t_ms is None or (t_ms - self.last_infer_t_ms) >= self.infer_stride_ms:
            should_classify = True
            self.last_infer_t_ms = t_ms

        if not should_classify:
            res = dict(self.latest_result)
            res["t"] = t_ms
            res["commit"] = None
            return res

        # Classification stride
        buf_duration = (t_ms - self.buffer[0][0]) if self.buffer else 0.0
        warming_up = buf_duration < 750.0

        # Hand presence in last 0.5s
        t_cutoff_hand = t_ms - 500
        recent_hand = any(elem[2] for elem in self.buffer if elem[0] >= t_cutoff_hand)

        commit = None

        if warming_up or not recent_hand:
            # Idle / no-hand: reset streak, accumulate idle stretch
            self.streak_label = None
            self.streak_count = 0
            self.idle_stretch_ms += dt_ms
            if self.idle_stretch_ms >= 400.0:
                self.cooldown_satisfied = True

            pred_label = self.idle_label
            pred_conf = 1.0
            topk = [(self.idle_label, 1.0)]
        else:
            # Sample causal window
            frames_arr = np.stack([elem[1] for elem in self.buffer])
            ts_arr = np.array([elem[0] for elem in self.buffer], dtype=np.int64)
            window = sample_window(frames_arr, ts_arr, t_end_ms=float(t_ms), window_s=self.window_s, n=C.SEQ_LEN)

            # Model inference
            x = torch.tensor(window[None], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            top_k_idx = probs.argsort()[::-1][:3]
            topk = [(self.labels[i], float(probs[i])) for i in top_k_idx]
            pred_idx = top_k_idx[0]
            pred_label = self.labels[pred_idx]
            pred_conf = float(probs[pred_idx])

            # Debounce and cooldown logic
            if pred_label == self.idle_label or pred_conf < self.conf_threshold:
                self.streak_label = None
                self.streak_count = 0
                self.idle_stretch_ms += dt_ms
                if self.idle_stretch_ms >= 400.0:
                    self.cooldown_satisfied = True
            else:
                self.idle_stretch_ms = 0.0
                if pred_label == self.streak_label:
                    self.streak_count += 1
                else:
                    self.streak_label = pred_label
                    self.streak_count = 1

                # Commit after exactly debounce_n consecutive strides
                if self.streak_count == self.debounce_n:
                    word = self.streak_label
                    # Cooldown check: same word cannot commit again without >= 0.4s idle stretch or different word
                    if word != self.last_committed_word or self.cooldown_satisfied:
                        commit = word
                        self.phrase.append(word)
                        self.last_committed_word = word
                        self.cooldown_satisfied = False

        self.latest_result = {
            "t": t_ms,
            "label": pred_label,
            "conf": pred_conf,
            "topk": topk,
            "hands": recent_hand,
            "warming_up": warming_up,
            "commit": commit,
            "phrase": list(self.phrase),
        }
        return dict(self.latest_result)
