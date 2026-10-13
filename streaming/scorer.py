"""Chấm điểm ONNX (dùng chung cho Spark Streaming và dashboard replay).

Graph ONNX trả thẳng score (MSE tái tạo với AE/VAE/DAE, xác suất với MLP); chuẩn hoá qua common.data.transform.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from common.data import load_scaler, transform

PASSTHROUGH_COLS = ["flow_id", "orig_ts", "label", "produce_ts", "kafka_ts"]
_CACHE = {}


class OnnxScorer:
    def __init__(self, model_path, scaler_path, threshold_path=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.scaler = load_scaler(scaler_path)
        self.threshold = None
        if threshold_path is not None:
            with open(threshold_path, encoding="utf-8") as f:
                self.threshold = float(json.load(f)["threshold"])

    def score_matrix(self, raw) -> np.ndarray:
        """raw: [n, d] theo thứ tự scaler.features. Feature thiếu (NaN) -> 0 sau chuẩn hoá (= giá trị trung bình)."""
        if len(raw) == 0:
            return np.zeros(0, dtype=np.float64)
        Z = np.nan_to_num(transform(raw, self.scaler), nan=0.0)
        return self.sess.run(["score"], {"input": Z})[0].reshape(-1).astype(np.float64)

    def score_frame_raw(self, df: pd.DataFrame) -> np.ndarray:
        """df có cột feature đúng tên trong scaler.json."""
        return self.score_matrix(df[self.scaler.features].to_numpy(dtype=np.float64))


def get_scorer(model_path, scaler_path, threshold_path) -> OnnxScorer:
    key = (str(model_path), str(scaler_path), str(threshold_path))
    if key not in _CACHE:
        _CACHE[key] = OnnxScorer(*key)
    return _CACHE[key]


def score_frame(pdf: pd.DataFrame, scorer: OnnxScorer, feature_cols) -> pd.DataFrame:
    """Cho Spark: pdf có PASSTHROUGH_COLS + cột feature (đặt tên f0..f{d-1} theo thứ tự scaler)."""
    out = pdf[PASSTHROUGH_COLS].copy()
    if len(pdf) == 0:
        out["recon_error"] = pd.Series(dtype="float64")
        out["is_anomaly"] = pd.Series(dtype="bool")
        return out
    err = scorer.score_matrix(pdf[list(feature_cols)].to_numpy(dtype=np.float64))
    out["recon_error"] = err
    out["is_anomaly"] = err >= scorer.threshold
    return out
