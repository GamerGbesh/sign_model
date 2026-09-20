"""Custom training pipeline for SignSpeak.

Trains SignLSTM on windowed custom dataset with class weighting, label smoothing,
data augmentation, confidence calibration, shuffled-label control, and evaluation.
Saves model artifacts, backups, metadata, and performance reports.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import config as C
from .model import SignLSTM, load_model, save_model
from .train import _accuracy, _save_confusion_png, train_model


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    weights = total / (num_classes * np.maximum(counts, 1).astype(np.float32))
    return (weights / np.mean(weights)).astype(np.float32)


def evaluate_model(
    model: SignLSTM,
    X: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    device: str = "cpu",
) -> Tuple[float, float, dict, np.ndarray, np.ndarray]:
    """Returns (accuracy, macro_f1, per_class_report, confusion_matrix, probs)."""
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(Xt)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)

    acc = float((preds == y).mean())
    macro_f1 = float(f1_score(y, preds, average="macro", zero_division=0))
    report = classification_report(
        y, preds, target_names=labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y, preds, labels=list(range(len(labels))))
    return acc, macro_f1, report, cm, probs


def calibrate_threshold(
    val_probs: np.ndarray,
    y_val: np.ndarray,
    idle_idx: int,
    tau_range: np.ndarray = np.arange(0.40, 0.96, 0.05),
) -> Tuple[float, float, float]:
    """Sweep confidence threshold tau in [0.4, 0.95].

    Picks tau with idle false-positive rate <= 2% and highest sign recall.
    Returns (best_tau, idle_fpr, sign_recall).
    """
    best_tau = 0.60
    best_sign_recall = -1.0
    best_idle_fpr = 1.0

    idle_mask = (y_val == idle_idx)
    sign_mask = ~idle_mask

    for tau in tau_range:
        max_prob = val_probs.max(axis=1)
        pred_label = val_probs.argmax(axis=1)

        # A prediction is counted as a sign prediction if max_prob >= tau and pred_label != idle_idx
        pred_sign = (max_prob >= tau) & (pred_label != idle_idx)

        # Idle False Positive: idle ground truth predicted as a sign
        if idle_mask.any():
            idle_fpr = float(pred_sign[idle_mask].mean())
        else:
            idle_fpr = 0.0

        # Sign recall: sign ground truth predicted correctly with conf >= tau
        if sign_mask.any():
            correct_sign = (pred_label[sign_mask] == y_val[sign_mask]) & (max_prob[sign_mask] >= tau)
            sign_recall = float(correct_sign.mean())
        else:
            sign_recall = 0.0

        # Satisfy idle FPR <= 2% (or closest if none <= 2%)
        if idle_fpr <= 0.02:
            if sign_recall > best_sign_recall:
                best_sign_recall = sign_recall
                best_tau = float(tau)
                best_idle_fpr = idle_fpr
        elif best_sign_recall < 0:
            # Fallback if no tau achieves <= 2% FPR
            best_tau = float(tau)
            best_idle_fpr = idle_fpr
            best_sign_recall = sign_recall

    return best_tau, best_idle_fpr, max(0.0, best_sign_recall)


def backup_existing_model():
    if C.MODEL_WEIGHTS.exists():
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = C.MODELS_DIR / "backup" / ts_str
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(C.MODEL_WEIGHTS, backup_dir / C.MODEL_WEIGHTS.name)
        if C.LABELS_JSON.exists():
            shutil.copy2(C.LABELS_JSON, backup_dir / C.LABELS_JSON.name)
        print(f"Backed up existing model to {backup_dir}")


def train_custom(
    data_dir: Path = C.CUSTOM_PROCESSED,
    seeds: int = 3,
    epochs: int = 150,
    batch_size: int = 16,
    lr: float = 1e-3,
    label_smoothing: float = 0.05,
    device: str = "cpu",
):
    print("================ LOADING DATASET ================")
    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")
    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    meta = json.loads((data_dir / "meta.json").read_text())
    labels = meta["labels"]
    num_classes = len(labels)
    idle_idx = labels.index(C.IDLE_LABEL)

    print(f"Loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples across {num_classes} classes.")
    class_weights = compute_class_weights(y_train, num_classes)
    print(f"Computed class weights: min={class_weights.min():.2f}, max={class_weights.max():.2f}")

    # 1. Flip comparison: train with p_flip=0.0 vs p_flip=0.5
    print("\n--- Comparing flip_horizontal probability (p_flip = 0.0 vs 0.5) ---")
    best_p_flip = 0.5
    best_flip_f1 = -1.0
    for pf in (0.0, 0.5):
        res = train_model(
            X=X_train, y=y_train, X_val=X_val, y_val=y_val,
            num_classes=num_classes, epochs=min(epochs, 80), batch_size=batch_size,
            lr=lr, device=device, seed=0, augment=True,
            class_weights=class_weights, label_smoothing=label_smoothing, p_flip=pf,
        )
        _, f1, _, _, _ = evaluate_model(res["model"], X_val, y_val, labels, device=device)
        print(f"  p_flip = {pf:.1f} -> Val Acc: {res['val_acc']:.3f}, Val Macro-F1: {f1:.3f}")
        if f1 > best_flip_f1:
            best_flip_f1 = f1
            best_p_flip = pf
    print(f"Selected best p_flip = {best_p_flip} (Val Macro-F1: {best_flip_f1:.3f})\n")

    # 2. Multi-seed training
    print(f"--- Training across {seeds} seeds with p_flip = {best_p_flip} ---")
    best_model = None
    best_val_f1 = -1.0
    best_seed = 0
    all_seed_results = []

    for s in range(seeds):
        res = train_model(
            X=X_train, y=y_train, X_val=X_val, y_val=y_val,
            num_classes=num_classes, epochs=epochs, batch_size=batch_size,
            lr=lr, device=device, seed=s, augment=True,
            class_weights=class_weights, label_smoothing=label_smoothing, p_flip=best_p_flip,
        )
        acc, f1, report, cm, probs = evaluate_model(res["model"], X_val, y_val, labels, device=device)
        print(f"  Seed {s:d}: Train Acc: {res['train_acc']:.3f} | Val Acc: {acc:.3f} | Val Macro-F1: {f1:.3f}")
        all_seed_results.append({"seed": s, "train_acc": res["train_acc"], "val_acc": acc, "val_f1": f1})
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model = res["model"]
            best_seed = s

    print(f"\nBest seed: {best_seed} with Val Macro-F1: {best_val_f1:.3f}")

    # 3. Shuffled-label control run (leakage check)
    print("\n--- Running shuffled-label control run (leakage check) ---")
    rng_ctrl = np.random.default_rng(999)
    y_shuffled = rng_ctrl.permutation(y_train)
    ctrl_res = train_model(
        X=X_train, y=y_shuffled, X_val=X_val, y_val=y_val,
        num_classes=num_classes, epochs=min(epochs, 60), batch_size=batch_size,
        lr=lr, device=device, seed=999, augment=True,
        class_weights=class_weights, label_smoothing=label_smoothing, p_flip=best_p_flip,
    )
    ctrl_acc, ctrl_f1, _, _, _ = evaluate_model(ctrl_res["model"], X_val, y_val, labels, device=device)
    chance_acc = 1.0 / num_classes
    print(f"Control Run (shuffled labels) -> Val Acc: {ctrl_acc:.3f} (Chance: {chance_acc:.3f}, F1: {ctrl_f1:.3f})")
    if ctrl_acc > chance_acc + 0.15:
        print("[WARN] Shuffled label accuracy significantly above chance - potential data leakage!")
    else:
        print("[PASS] Shuffled label accuracy is near chance. No data leakage detected.")

    # 4. Calibration on val set only
    val_acc, val_f1, val_rep, val_cm, val_probs = evaluate_model(
        best_model, X_val, y_val, labels, device=device
    )
    best_tau, idle_fpr, sign_recall = calibrate_threshold(val_probs, y_val, idle_idx)
    # Require at least 300-400ms consistent predictions at INFER_STRIDE_MS (100ms)
    debounce_n = max(3, int(round(350.0 / C.INFER_STRIDE_MS)))
    print(f"\n--- Calibration (Val Set) ---")
    print(f"Optimal confidence threshold (tau): {best_tau:.2f}")
    print(f"Idle false-positive rate: {idle_fpr:.1%}")
    print(f"Sign recall @ tau: {sign_recall:.1%}")
    print(f"Selected debounce_n: {debounce_n} strides ({debounce_n * C.INFER_STRIDE_MS} ms)")

    # 5. Evaluate on Test split (at native and 15 fps)
    test_acc, test_f1, test_rep, test_cm, test_probs = evaluate_model(
        best_model, X_test, y_test, labels, device=device
    )
    print(f"\n================ TEST SET PERFORMANCE ================")
    print(f"Test Accuracy: {test_acc:.3f} | Test Macro-F1: {test_f1:.3f}")

    # Check 15 fps test performance if available
    test_15_acc = None
    if (data_dir / "X_test_15fps.npy").exists():
        X_test_15 = np.load(data_dir / "X_test_15fps.npy")
        y_test_15 = np.load(data_dir / "y_test_15fps.npy")
        test_15_acc, test_15_f1, _, _, _ = evaluate_model(best_model, X_test_15, y_test_15, labels, device=device)
        print(f"Test (simulated 15 fps) -> Accuracy: {test_15_acc:.3f} | Macro-F1: {test_15_f1:.3f}")

    # 6. Save reports & confusion matrices
    report_dir = C.MODELS_DIR / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    _save_confusion_png(val_cm, labels, val_acc, out_path=report_dir / "confusion_matrix_val.png")
    _save_confusion_png(test_cm, labels, test_acc, out_path=report_dir / "confusion_matrix_test.png")

    report_data = {
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "test_15fps_accuracy": test_15_acc,
        "idle_fpr": idle_fpr,
        "sign_recall_at_tau": sign_recall,
        "conf_threshold_tau": best_tau,
        "debounce_n": debounce_n,
        "best_seed": best_seed,
        "best_p_flip": best_p_flip,
        "control_shuffled_val_acc": ctrl_acc,
        "val_classification_report": val_rep,
        "test_classification_report": test_rep,
        "all_seeds": all_seed_results,
    }
    (report_dir / "metrics.json").write_text(json.dumps(report_data, indent=2))
    print(f"\nSaved metrics and confusion plots to {report_dir}/")

    # 7. Backup and Save Model Artifacts
    backup_existing_model()
    save_model(best_model, labels, weights_path=C.MODEL_WEIGHTS, labels_path=C.LABELS_JSON)

    # Save model_meta.json
    model_meta = {
        "labels": labels,
        "idle_label": C.IDLE_LABEL,
        "seq_len": C.SEQ_LEN,
        "feature_dim": C.FEATURE_DIM,
        "window_s": meta["window_s"],
        "infer_stride_ms": C.INFER_STRIDE_MS,
        "conf_threshold": best_tau,
        "debounce_n": debounce_n,
        "extractor": "holistic-video",
        "flip": False,
        "metrics": {
            "val_acc": val_acc,
            "val_f1": val_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
    }
    (C.MODELS_DIR / "model_meta.json").write_text(json.dumps(model_meta, indent=2))
    print(f"Saved weights to {C.MODEL_WEIGHTS}")
    print(f"Saved labels to {C.LABELS_JSON}")
    print(f"Saved metadata to {C.MODELS_DIR / 'model_meta.json'}")

    return report_data


def main():
    parser = argparse.ArgumentParser(description="Train SignSpeak custom-vocabulary recognizer")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds to train")
    parser.add_argument("--epochs", type=int, default=160, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_custom(seeds=args.seeds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
