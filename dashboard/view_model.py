"""Logic thuần cho dashboard (không import streamlit) — test được, chạy được trên Python 3.8.

Hai nguồn dữ liệu cùng quy về một khung chung (FRAME_COLUMNS):
  flow_id, orig_ts, label, score, ts      (ts = epoch ms lúc flow được xử lý)
- replay: ReplayFeed phát lại data/demo_test.parquet, ScoreCache chấm điểm ONNX theo lô (lười, cache theo model)
- stream: thư mục output Spark Streaming (streaming.output_io.load_output)
Ngưỡng áp dụng lúc hiển thị (score >= thr) nên kéo thanh trượt không cần chấm điểm lại.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import BENIGN, ID_COL, LABEL_COL, SEED, TS_COL, binarize
from common.metrics import confusion_metrics, per_attack_recall

FRAME_COLUMNS = ["flow_id", "orig_ts", "label", "score", "ts"]
KIND_NAMES = {"ae": "DeepAutoencoder", "vae": "VariationalAE", "dae": "DenoisingAE", "mlp": "DNN_Classifier"}
KIND_LABELS = {"ae": "Deep Autoencoder (AE)", "vae": "Variational AE (VAE)", "dae": "Denoising AE (DAE)",
               "mlp": "DNN Classifier (MLP)"}
REPORT_CAPTIONS = {
    "roc_pr_unsupervised.png": "ROC và Precision-Recall — nhóm unsupervised (AE, VAE, DAE, PCA, IF, OCSVM).",
    "roc_pr_supervised.png": "ROC và Precision-Recall — nhóm supervised (MLP, RF, LR, LightGBM).",
    "f1_fpr_bar.png": "F1 và FPR của từng model, tô màu theo nhóm DL/ML.",
    "score_hist_ae_family.png": "Phân phối score Benign vs Attack (trục log) kèm ngưỡng của AE/VAE/DAE.",
    "loss_curves.png": "Đường loss train/val của 4 model DL.",
    "vae_recon_kl.png": "VAE: reconstruction loss và KL theo epoch.",
    "per_attack_recall_heatmap.png": "Recall theo từng loại tấn công của mọi model.",
    "ablation_latent.png": "Ablation AE: F1 và FPR theo kích thước latent.",
    "ablation_depth.png": "Ablation AE: F1 và FPR theo độ sâu (kèm số tham số).",
    "ablation_threshold.png": "Ablation AE: Precision/Recall/FPR theo percentile ngưỡng.",
    "ablation_vae_beta.png": "Ablation VAE: F1 và FPR theo β.",
}


# ---------------------------------------------------------------- model / threshold
def list_models(models_dir) -> list:
    """Mọi models/<kind>.onnx có ngưỡng đi kèm (threshold_<kind>.json; riêng ae chấp nhận threshold.json)."""
    out = []
    mdir = Path(models_dir)
    for onnx_path in sorted(mdir.glob("*.onnx")):
        kind = onnx_path.stem
        thr_path = mdir / ("threshold_%s.json" % kind)
        if not thr_path.exists() and kind == "ae":
            thr_path = mdir / "threshold.json"
        if not thr_path.exists():
            continue
        thr = json.loads(thr_path.read_text(encoding="utf-8"))
        out.append({"kind": kind, "name": KIND_NAMES.get(kind, kind), "label": KIND_LABELS.get(kind, kind),
                    "onnx": str(onnx_path), "threshold_path": str(thr_path),
                    "threshold": float(thr["threshold"]), "percentile": thr.get("percentile")})
    order = {k: i for i, k in enumerate(KIND_NAMES)}
    return sorted(out, key=lambda m: order.get(m["kind"], 99))


def load_valscores(results_dir, model_name):
    p = Path(results_dir) / ("valscores_%s.npy" % model_name)
    return np.load(p) if p.exists() else None


def threshold_from_percentile(valscores, pct: float) -> float:
    return float(np.percentile(valscores, pct))


# ---------------------------------------------------------------- replay
class ReplayFeed:
    """Phát lại demo theo thứ tự cố định: 'shuffle' (seed 42) hoặc 'orig_ts'."""

    def __init__(self, df: pd.DataFrame, order: str = "shuffle", seed: int = SEED):
        self.df = df.reset_index(drop=True)
        if order == "orig_ts":
            ts = pd.to_datetime(self.df[TS_COL], format="%d/%m/%Y %H:%M:%S", errors="coerce")
            self.order = np.lexsort((self.df[ID_COL].to_numpy(), ts.fillna(pd.Timestamp.max).to_numpy()))
        else:
            self.order = np.random.default_rng(seed).permutation(len(self.df))
        self.order_name = order
        self.reset()

    def reset(self) -> None:
        self.pos = 0
        self.emit_ts = np.zeros(len(self.df), dtype=np.int64)

    @property
    def done(self) -> bool:
        return self.pos >= len(self.df)

    def emit(self, n: int, now_ms: int) -> np.ndarray:
        idx = self.order[self.pos:self.pos + max(int(n), 0)]
        self.emit_ts[idx] = now_ms
        self.pos += len(idx)
        return idx

    def emitted(self) -> np.ndarray:
        return self.order[:self.pos]


class ScoreCache:
    """Score theo từng model, chỉ tính cho các dòng đã phát (theo lô)."""

    def __init__(self, n_rows: int):
        self.n = n_rows
        self.scores = {}

    def get(self, key: str, scorer, df: pd.DataFrame, idx: np.ndarray, batch: int = 4096) -> np.ndarray:
        arr = self.scores.setdefault(key, np.full(self.n, np.nan))
        todo = idx[np.isnan(arr[idx])]
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            arr[part] = scorer.score_frame_raw(df.iloc[part])
        return arr


def replay_frame(feed: ReplayFeed, scores: np.ndarray) -> pd.DataFrame:
    idx = feed.emitted()
    df = feed.df.iloc[idx]
    return pd.DataFrame({"flow_id": df[ID_COL].to_numpy(), "orig_ts": df[TS_COL].to_numpy(),
                         "label": df[LABEL_COL].astype(str).to_numpy(), "score": scores[idx],
                         "ts": feed.emit_ts[idx]}, columns=FRAME_COLUMNS)


# ---------------------------------------------------------------- stream
def stream_frame(out_df: pd.DataFrame) -> pd.DataFrame:
    """Output Spark (OUTPUT_COLUMNS) -> khung chung; score = recon_error, ts = detect_ts."""
    if len(out_df) == 0:
        return pd.DataFrame(columns=FRAME_COLUMNS)
    return pd.DataFrame({"flow_id": out_df["flow_id"].to_numpy(), "orig_ts": out_df["orig_ts"].to_numpy(),
                         "label": out_df["label"].astype(str).to_numpy(),
                         "score": out_df["recon_error"].astype(float).to_numpy(),
                         "ts": out_df["detect_ts"].astype("int64").to_numpy()}, columns=FRAME_COLUMNS)


# ---------------------------------------------------------------- views
def kpis(frame: pd.DataFrame, thr: float, window_s: float = 10) -> dict:
    total = len(frame)
    if total == 0:
        return {"total": 0, "alerts": 0, "alert_rate": 0.0, "throughput": 0.0}
    alerts = int((frame["score"] >= thr).sum())
    last = frame["ts"].max()
    recent = int((frame["ts"] > last - window_s * 1000).sum())
    return {"total": total, "alerts": alerts, "alert_rate": alerts / total, "throughput": recent / window_s}


def per_second(frame: pd.DataFrame, thr: float) -> pd.DataFrame:
    if len(frame) == 0:
        return pd.DataFrame(columns=["time", "flows", "alerts"])
    g = frame.assign(sec=frame["ts"] // 1000, alert=(frame["score"] >= thr).astype(int)).groupby("sec") \
        .agg(flows=("flow_id", "size"), alerts=("alert", "sum")).reset_index()
    g["time"] = pd.to_datetime(g["sec"], unit="s")
    return g[["time", "flows", "alerts"]]


def score_timeline(frame: pd.DataFrame, last_n: int = 3000) -> pd.DataFrame:
    """N flow gần nhất theo thứ tự xử lý: seq, score (không có nhãn)."""
    t = frame.tail(last_n).reset_index(drop=True)
    return pd.DataFrame({"seq": np.arange(len(t)), "score": t["score"].to_numpy(dtype=float)})


def top_suspicious(frame: pd.DataFrame, k: int = 15, recent: int = 5000) -> pd.DataFrame:
    t = frame.tail(recent).nlargest(k, "score")
    return pd.DataFrame({"flow_id": t["flow_id"].to_numpy(), "orig_ts": t["orig_ts"].to_numpy(),
                         "score": t["score"].to_numpy(),
                         "thời điểm xử lý": pd.to_datetime(t["ts"].to_numpy(), unit="ms")})


def evaluation(frame: pd.DataFrame, thr: float) -> dict:
    y = binarize(frame["label"])
    pred = (frame["score"] >= thr).to_numpy()
    m = confusion_metrics(y, pred)
    cm = pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                      index=["Thật: Benign", "Thật: Attack"], columns=["Dự đoán: Benign", "Dự đoán: Attack"])
    pa = per_attack_recall("", frame["label"].to_numpy(), pred).drop(columns=["model"])
    return {"metrics": m, "confusion": cm, "per_attack": pa}


# ---------------------------------------------------------------- research tab
def eda_captions(eda_dir) -> dict:
    p = Path(eda_dir) / "eda_summary.md"
    if not p.exists():
        return {}
    caps = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^- `([^`]+\.png)` — (.+)$", line.strip())
        if m:
            caps[m.group(1)] = m.group(2)
    return caps


def research_assets(reports_dir) -> dict:
    """{'comparison_md': str|None, 'figures': [(path, caption)], 'eda': [(path, caption)]}."""
    rep = Path(reports_dir)
    md = rep / "model_comparison.md"
    figs = [(str(rep / name), cap) for name, cap in REPORT_CAPTIONS.items() if (rep / name).exists()]
    caps = eda_captions(rep / "eda")
    eda = [(str(rep / "eda" / name), cap) for name, cap in caps.items() if (rep / "eda" / name).exists()]
    return {"comparison_md": md.read_text(encoding="utf-8") if md.exists() else None, "figures": figs, "eda": eda}
