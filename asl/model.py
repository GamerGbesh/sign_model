"""The sign classifier: an LSTM over landmark sequences.

Kept deliberately small and swappable -- the rest of the pipeline only depends
on the (batch, SEQ_LEN, FEATURE_DIM) -> (batch, num_classes) contract, so an MLP
or transformer could drop in here unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from . import config as C


class SignLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_dim: int = C.FEATURE_DIM,
        hidden: int = C.LSTM_HIDDEN,
        layers: int = C.LSTM_LAYERS,
        dropout: float = C.LSTM_DROPOUT,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        feat = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(feat),
            nn.Dropout(dropout),
            nn.Linear(feat, num_classes),
        )

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)   # average over time — robust for short clips
        return self.head(pooled)


def save_model(model: SignLSTM, labels: list[str], weights_path: Path = C.MODEL_WEIGHTS,
               labels_path: Path = C.LABELS_JSON) -> None:
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_classes": len(labels)}, weights_path)
    Path(labels_path).write_text(json.dumps(labels, indent=2))


def load_model(
    weights_path: Path = C.MODEL_WEIGHTS,
    labels_path: Path = C.LABELS_JSON,
    device: str = "cpu",
    expected_num_classes: int | None = None,
    expected_feature_dim: int = C.FEATURE_DIM,
) -> tuple[SignLSTM, list[str]]:
    labels = json.loads(Path(labels_path).read_text())
    ckpt = torch.load(weights_path, map_location=device)
    num_classes = ckpt.get("num_classes", len(labels))
    if num_classes != len(labels):
        raise ValueError(
            f"Model checkpoint num_classes ({num_classes}) does not match labels.json count ({len(labels)})"
        )
    if expected_num_classes is not None and num_classes != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} classes, but model checkpoint has {num_classes}"
        )
    model = SignLSTM(num_classes=num_classes, input_dim=expected_feature_dim)
    model.load_state_dict(ckpt["state_dict"])
    if model.lstm.input_size != expected_feature_dim:
        raise ValueError(
            f"Model feature dim ({model.lstm.input_size}) does not match expected ({expected_feature_dim})"
        )
    model.to(device).eval()
    return model, labels
