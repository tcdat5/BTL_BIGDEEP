"""Quy ước dữ liệu dùng CHUNG cho baseline, DL và streaming (PROJECT_SPEC mục 4).

Mọi chỗ chuẩn hoá feature phải đi qua transform() ở đây.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

LABEL_COL = "Label"
BENIGN = "Benign"
TS_COL = "orig_ts"
ID_COL = "flow_id"
META_COLS = (ID_COL, TS_COL, LABEL_COL)
CLIP = 10.0
SEED = 42


@dataclass
class Scaler:
    features: List[str]
    mean: np.ndarray
    std: np.ndarray

    def to_json(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"features": list(self.features),
                       "mean": [float(v) for v in self.mean],
                       "std": [float(v) for v in self.std]}, f, indent=1)


def _fix_std(std) -> np.ndarray:
    std = np.asarray(std, dtype=np.float64).copy()
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return std


def load_scaler(path) -> Scaler:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return Scaler(list(d["features"]), np.asarray(d["mean"], dtype=np.float64), _fix_std(d["std"]))


def fit_scaler(df: pd.DataFrame, features) -> Scaler:
    """Fit trên các dòng Benign; std quần thể (ddof=0) giống Spark stddev_pop; std 0 -> 1."""
    features = list(features)
    benign = df.loc[df[LABEL_COL] == BENIGN, features].to_numpy(dtype=np.float64)
    if len(benign) == 0:
        raise ValueError("no Benign rows to fit scaler")
    return Scaler(features, benign.mean(axis=0), _fix_std(benign.std(axis=0)))


def transform(X, scaler: Scaler) -> np.ndarray:
    Z = (np.asarray(X, dtype=np.float64) - scaler.mean) / scaler.std
    return np.clip(Z, -CLIP, CLIP).astype(np.float32)


def frame_to_matrix(df: pd.DataFrame, scaler: Scaler) -> np.ndarray:
    return transform(df[scaler.features].to_numpy(dtype=np.float64), scaler)


def binarize(labels) -> np.ndarray:
    return (np.asarray(labels) != BENIGN).astype(np.int64)


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META_COLS]


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    meta: pd.DataFrame

    def subset(self, mask) -> "Split":
        mask = np.asarray(mask, dtype=bool)
        return Split(self.X[mask], self.y[mask], self.labels[mask],
                     self.meta[mask].reset_index(drop=True))


def read_split_frame(data_dir, split: str, cap=None, seed: int = SEED) -> pd.DataFrame:
    path = Path(data_dir) / split
    if not path.exists():
        raise FileNotFoundError("split not found: %s" % path)
    df = pd.read_parquet(path)
    if ID_COL in df.columns:   # thứ tự ổn định, không phụ thuộc thứ tự file part-*
        df = df.sort_values(ID_COL, kind="mergesort")
    df = df.reset_index(drop=True)
    if cap is not None and len(df) > cap:
        df = df.sample(n=cap, random_state=seed).reset_index(drop=True)
    return df


def load_split(data_dir, split: str, scaler: Scaler, cap=None, seed: int = SEED,
               benign_only: bool = False) -> Split:
    df = read_split_frame(data_dir, split, cap, seed)
    if benign_only:
        df = df[df[LABEL_COL] == BENIGN].reset_index(drop=True)
    labels = df[LABEL_COL].astype(str).to_numpy()
    meta = df[[c for c in META_COLS if c in df.columns]].reset_index(drop=True)
    return Split(frame_to_matrix(df, scaler), binarize(labels), labels, meta)
