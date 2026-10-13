"""So output baseline cũ/mới: python tests/compare_results.py OLD_DIR NEW_DIR [SUFFIX]

Bỏ qua cột thời gian và model PCA: trên dữ liệu giả 16 feature độc lập, PCA(0.95) giữ đủ 16 thành phần nên
score ~1e-14 (nhiễu làm tròn) và đổi theo chênh lệch float32/float64 của transform. Dung sai 1e-4 cho phần còn lại
(LogisticRegression/lbfgs hội tụ hơi khác khi đầu vào lệch ~1e-7).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

IGNORE_COLS = ("train_time_s", "infer_ms_per_1k")
SKIP_MODELS = ("PCA",)


def compare_dirs(old: Path, new: Path, suffix: str = "") -> None:
    for name in ("baselines_metrics%s.csv" % suffix, "baselines_per_attack%s.csv" % suffix):
        a, b = pd.read_csv(old / name), pd.read_csv(new / name)
        assert list(a.columns) == list(b.columns), (name, list(a.columns), list(b.columns))
        a = a[~a["model"].isin(SKIP_MODELS)].drop(columns=[c for c in IGNORE_COLS if c in a]).reset_index(drop=True)
        b = b[~b["model"].isin(SKIP_MODELS)].drop(columns=[c for c in IGNORE_COLS if c in b]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-4)


if __name__ == "__main__":
    compare_dirs(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
    print("IDENTICAL (trong dung sai, xem docstring)")
