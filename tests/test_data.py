import json

import numpy as np
import pandas as pd
import pytest

from common.data import (BENIGN, ID_COL, LABEL_COL, Scaler, binarize, feature_columns,
                         fit_scaler, frame_to_matrix, load_scaler, load_split, transform)
from common.fake import FAKE_FEATURES, write_fake_dataset


def test_transform_clip_and_dtype():
    s = Scaler(["a", "b"], np.array([0.0, 10.0]), np.array([1.0, 2.0]))
    Z = transform(np.array([[3.0, 10.0], [100.0, -100.0]]), s)
    assert Z.dtype == np.float32
    np.testing.assert_allclose(Z, [[3.0, 0.0], [10.0, -10.0]])


def test_load_scaler_replaces_zero_std(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"features": ["a", "b"], "mean": [1, 2], "std": [0, 3]}))
    s = load_scaler(p)
    assert s.features == ["a", "b"]
    np.testing.assert_allclose(s.std, [1.0, 3.0])


def test_fit_scaler_uses_benign_only_ddof0():
    df = pd.DataFrame({"a": [1.0, 3.0, 1000.0], "b": [5.0, 5.0, 0.0],
                       LABEL_COL: [BENIGN, BENIGN, "Bot"]})
    s = fit_scaler(df, ["a", "b"])
    np.testing.assert_allclose(s.mean, [2.0, 5.0])
    np.testing.assert_allclose(s.std, [1.0, 1.0])   # std a = 1 (ddof=0); std b = 0 -> 1


def test_scaler_json_roundtrip(tmp_path):
    s = Scaler(["x y", "z/s"], np.array([1.5, 2.0]), np.array([0.5, 1.0]))
    s.to_json(tmp_path / "m" / "scaler.json")
    t = load_scaler(tmp_path / "m" / "scaler.json")
    assert t.features == s.features
    np.testing.assert_allclose(t.mean, s.mean)


def test_frame_to_matrix_uses_scaler_column_order():
    s = Scaler(["b", "a"], np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    np.testing.assert_allclose(frame_to_matrix(df, s), [[2.0, 1.0]])


def test_binarize():
    assert binarize([BENIGN, "Bot", "DoS attacks-Hulk"]).tolist() == [0, 1, 1]


def test_load_split_cap_is_deterministic(tmp_path):
    scaler = write_fake_dataset(tmp_path, n=4000)
    a = load_split(tmp_path, "train", scaler, cap=500)
    b = load_split(tmp_path, "train", scaler, cap=500)
    assert len(a.y) == 500
    assert a.meta[ID_COL].tolist() == b.meta[ID_COL].tolist()
    assert a.X.shape == (500, len(FAKE_FEATURES)) and a.X.dtype == np.float32
    np.testing.assert_array_equal(a.y, binarize(a.labels))


def test_load_split_benign_only_and_subset(tmp_path):
    scaler = write_fake_dataset(tmp_path, n=4000)
    v = load_split(tmp_path, "val", scaler, benign_only=True)
    assert (v.labels == BENIGN).all() and v.y.sum() == 0
    full = load_split(tmp_path, "val", scaler)
    sub = full.subset(full.labels != "Bot")
    assert "Bot" not in set(sub.labels) and len(sub.meta) == len(sub.y)


def test_feature_columns_excludes_meta(tmp_path):
    write_fake_dataset(tmp_path, n=1000)
    df = pd.read_parquet(tmp_path / "test")
    assert feature_columns(df) == FAKE_FEATURES


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_split(tmp_path, "train", Scaler(["a"], np.zeros(1), np.ones(1)))
