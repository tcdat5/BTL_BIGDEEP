import json

import numpy as np
import pandas as pd
import pytest

from common.metrics import METRIC_COLUMNS

pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")


def test_baselines_smoke(tmp_path):
    from train import baselines
    baselines.main(["--smoke", "--out", str(tmp_path)])
    m = pd.read_csv(tmp_path / "baselines_metrics.csv")
    assert list(m.columns) == METRIC_COLUMNS
    assert len(m) == 6 and set(m["group"]) == {"unsupervised", "supervised"}
    pa = pd.read_csv(tmp_path / "baselines_per_attack.csv")
    assert list(pa.columns) == ["model", "attack", "recall", "n"]
    files = sorted(tmp_path.glob("scores_*.npz"))
    assert len(files) == 6
    assert set(np.load(files[0]).files) == {"y", "score", "thr"}


def test_baselines_holdout_suffix(tmp_path):
    from train import baselines
    baselines.main(["--smoke", "--out", str(tmp_path), "--holdout-attack", "Bot"])
    assert (tmp_path / "baselines_metrics_holdout-Bot.csv").exists()
    assert json.loads((tmp_path / "holdout.json").read_text()) == {"attack": "Bot"}
