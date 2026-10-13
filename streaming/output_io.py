"""Định dạng output streaming (dùng chung cho stream_detect, benchmark, dashboard)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUTPUT_COLUMNS = ["flow_id", "orig_ts", "label", "recon_error", "is_anomaly", "produce_ts", "kafka_ts", "detect_ts"]


def load_output(out_dir, max_files=None) -> pd.DataFrame:
    files = sorted(Path(out_dir).glob("*.parquet"))
    if max_files:
        files = files[-max_files:]
    if not files:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
