import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
from common.metrics import METRIC_COLUMNS  # noqa: E402
from train import train_dl  # noqa: E402

pytestmark = pytest.mark.torch
NAMES = {"ae": "DeepAutoencoder", "vae": "VariationalAE", "dae": "DenoisingAE", "mlp": "DNN_Classifier"}


@pytest.mark.parametrize("kind", ["ae", "vae", "dae", "mlp"])
def test_smoke_all_models(tmp_path, kind):
    res, mdl = tmp_path / "r", tmp_path / "m"
    row = train_dl.main(["--model", kind, "--smoke", "--results-dir", str(res), "--models-dir", str(mdl)])
    m = pd.read_csv(res / ("dl_%s_metrics.csv" % kind))
    assert list(m.columns) == METRIC_COLUMNS + ["n_params"]
    assert m.loc[0, "model"] == NAMES[kind] and m.loc[0, "n_params"] > 0
    assert row["model"] == NAMES[kind]
    hist = pd.read_csv(res / ("dl_%s_history.csv" % kind))
    assert {"epoch", "train_loss", "val_loss"} <= set(hist.columns) and len(hist) == 2
    if kind == "vae":
        assert {"train_recon", "train_kl", "val_recon", "val_kl"} <= set(hist.columns)
    npz = np.load(res / ("scores_%s.npz" % NAMES[kind]))
    assert set(npz.files) == {"y", "score", "thr"}
    thr = json.loads((mdl / ("threshold_%s.json" % kind)).read_text())
    assert thr["threshold"] == pytest.approx(float(npz["thr"]))
    if kind == "mlp":
        assert thr["threshold"] == 0.5
    else:
        assert thr["percentile"] == 99
        assert (res / ("valscores_%s.npy" % NAMES[kind])).exists()
    model = onnx.load(str(mdl / ("%s.onnx" % kind)))
    assert model.ir_version <= 8 and model.opset_import[0].version == 17
    sess = ort.InferenceSession(str(mdl / ("%s.onnx" % kind)), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["input"]
    assert [o.name for o in sess.get_outputs()] == ["score"]
    if kind == "ae":
        assert (mdl / "threshold.json").exists() and (mdl / "ae.onnx").exists()
    assert (mdl / ("%s.pt" % kind)).exists()


def test_holdout_removes_attack_from_train(tmp_path):
    res = tmp_path / "r"
    train_dl.main(["--model", "mlp", "--smoke", "--holdout-attack", "Bot", "--no-export",
                   "--results-dir", str(res), "--models-dir", str(tmp_path / "m")])
    assert json.loads((res / "holdout.json").read_text()) == {"attack": "Bot"}
    pa = pd.read_csv(res / "dl_mlp_per_attack.csv")
    assert "Bot" in set(pa["attack"])     # vẫn được đánh giá trên test
