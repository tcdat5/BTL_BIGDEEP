"""Đánh giá dùng CHUNG cho baseline, DL và streaming."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.data import BENIGN

METRIC_COLUMNS = ["model", "group", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc",
                  "threshold", "train_time_s", "infer_ms_per_1k"]


def confusion_metrics(y, y_pred) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(y_pred).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def evaluate(y, score, thr) -> dict:
    """Dự đoán tấn công khi score >= thr. pr_auc = average precision. sklearn import lười (cụm không có)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    m = confusion_metrics(y, score >= thr)
    out = {k: float(m[k]) for k in ("precision", "recall", "f1", "fpr")}
    both = len(np.unique(y)) == 2
    out["roc_auc"] = float(roc_auc_score(y, score)) if both else float("nan")
    out["pr_auc"] = float(average_precision_score(y, score)) if both else float("nan")
    out["threshold"] = float(thr)
    return out


def per_attack_recall(model: str, labels, y_pred) -> pd.DataFrame:
    labels = np.asarray(labels).astype(str)
    y_pred = np.asarray(y_pred).astype(bool)
    rows = []
    for att in sorted(set(labels) - {BENIGN}):
        mask = labels == att
        rows.append({"model": model, "attack": att,
                     "recall": float(y_pred[mask].mean()), "n": int(mask.sum())})
    return pd.DataFrame(rows, columns=["model", "attack", "recall", "n"])


def save_scores(path, y, score, thr) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, y=np.asarray(y), score=np.asarray(score), thr=np.asarray(thr))
