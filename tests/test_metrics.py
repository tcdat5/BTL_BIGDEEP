import numpy as np
import pytest

from common.metrics import (METRIC_COLUMNS, confusion_metrics, evaluate, per_attack_recall,
                            save_scores)

Y = [0, 0, 0, 1, 1, 1]
S = [0.1, 0.2, 0.6, 0.4, 0.7, 0.9]


def test_confusion_metrics_hand_computed():
    m = confusion_metrics(Y, np.array(S) >= 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 2, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 3)


def test_confusion_metrics_no_positive_predictions():
    m = confusion_metrics([0, 1], [0, 0])
    assert m["precision"] == 0.0 and m["f1"] == 0.0


def test_evaluate():
    pytest.importorskip("sklearn")
    m = evaluate(Y, S, 0.5)
    assert m["roc_auc"] == pytest.approx(8 / 9)
    assert 0 < m["pr_auc"] <= 1
    assert m["threshold"] == 0.5
    assert set(m) == {"precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "threshold"}


def test_evaluate_single_class_gives_nan_auc():
    pytest.importorskip("sklearn")
    m = evaluate([0, 0], [0.1, 0.2], 0.5)
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])


def test_per_attack_recall():
    labels = ["Benign", "Benign", "Benign", "Bot", "Bot", "DoS"]
    df = per_attack_recall("M", labels, np.array(S) >= 0.5)
    assert list(df.columns) == ["model", "attack", "recall", "n"]
    rows = {r.attack: (r.recall, r.n) for r in df.itertuples()}
    assert rows == {"Bot": (0.5, 2), "DoS": (1.0, 1)}


def test_save_scores_roundtrip(tmp_path):
    save_scores(tmp_path / "scores_X.npz", Y, S, 0.5)
    d = np.load(tmp_path / "scores_X.npz")
    assert set(d.files) == {"y", "score", "thr"}
    assert float(d["thr"]) == 0.5


def test_metric_columns_match_spec():
    assert METRIC_COLUMNS == ["model", "group", "precision", "recall", "f1", "fpr", "roc_auc",
                              "pr_auc", "threshold", "train_time_s", "infer_ms_per_1k"]
