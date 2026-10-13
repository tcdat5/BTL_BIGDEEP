import json

import numpy as np
import pandas as pd
import pytest

from common.data import BENIGN, ID_COL, LABEL_COL, TS_COL, load_scaler
from common.fake import FAKE_ATTACKS, FAKE_FEATURES, write_fake_raw_csvs

pytest.importorskip("pyspark")
pytestmark = pytest.mark.spark
N_PER_FILE = 1500
SPLIT_NAMES = ("train", "val", "test")


def _run(d, files, tag, repartition=None, demo_rows=500):
    from preprocess.spark_preprocess import main
    argv = ["--master", "local[2]", "--input-dir", str(d / "raw"), "--files", *files,
            "--out", str(d / tag / "out"), "--export-sample", str(d / tag / "sample"),
            "--sample-train", "1000", "--sample-val", "200", "--sample-test", "300",
            "--demo-out", str(d / tag / "demo_test.parquet"), "--demo-rows", str(demo_rows),
            "--scaler-out", str(d / tag / "scaler.json"), "--stats-out", str(d / tag / "stats.json")]
    if repartition:
        argv += ["--debug-repartition", str(repartition)]
    return main(argv)


@pytest.fixture(scope="module")
def run_pre(tmp_path_factory):
    d = tmp_path_factory.mktemp("pre")
    files = write_fake_raw_csvs(d / "raw", n_per_file=N_PER_FILE)
    stats = _run(d, files, "p1", repartition=1)
    return d, files, stats


def _read(d, split, tag="p1"):
    return pd.read_parquet(d / tag / "out" / split)


def test_row_counts_and_cleaning(run_pre):
    d, _, stats = run_pre
    total = sum(len(_read(d, s)) for s in SPLIT_NAMES)
    assert total == 3 * (N_PER_FILE - 2)
    assert stats["rows"]["after_dedup"] == total


def test_columns_and_unique_ids(run_pre):
    d, _, _ = run_pre
    df = _read(d, "train")
    assert list(df.columns) == [ID_COL, TS_COL] + FAKE_FEATURES + [LABEL_COL]
    ids = pd.concat([_read(d, s)[ID_COL] for s in SPLIT_NAMES])
    assert ids.is_unique


def test_split_is_hash_bucket(run_pre):
    d, _, _ = run_pre
    for name, lo, hi in (("train", 0, 70), ("val", 70, 80), ("test", 80, 100)):
        b = np.mod(_read(d, name)[ID_COL].to_numpy(), 100)     # pmod: luôn >= 0
        assert ((b >= lo) & (b < hi)).all()
    total = sum(len(_read(d, s)) for s in SPLIT_NAMES)
    assert abs(len(_read(d, "train")) / total - 0.7) < 0.05


def test_split_independent_of_partitioning(run_pre):
    d, files, _ = run_pre
    _run(d, files, "p8", repartition=8)
    for s in SPLIT_NAMES:
        a = _read(d, s, "p1").sort_values(ID_COL).reset_index(drop=True)
        b = _read(d, s, "p8").sort_values(ID_COL).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
    for f in ("sample/train", "demo_test.parquet"):
        a = pd.read_parquet(d / "p1" / f).sort_values(ID_COL).reset_index(drop=True)
        b = pd.read_parquet(d / "p8" / f).sort_values(ID_COL).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


def test_scaler_matches_numpy(run_pre):
    d, _, _ = run_pre
    s = load_scaler(d / "p1" / "scaler.json")
    assert s.features == FAKE_FEATURES
    b = _read(d, "train")
    b = b[b[LABEL_COL] == BENIGN][FAKE_FEATURES].to_numpy(np.float64)
    np.testing.assert_allclose(s.mean, b.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(s.std, b.std(axis=0), rtol=1e-6)


def test_sample_and_stats(run_pre):
    d, _, stats = run_pre
    tr = pd.read_parquet(d / "p1" / "sample" / "train")
    assert len(tr) == 1000
    saved = json.loads((d / "p1" / "stats.json").read_text(encoding="utf-8"))
    assert saved["dropped_columns"]["constant"] == ["Const Col"]
    assert set(saved["dropped_columns"]["identifier"]) >= {"Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port"}
    assert set(saved["label_distribution"]) == set(SPLIT_NAMES)
    assert "read_clean" in saved["timings_s"] and "total" in saved["timings_s"]


def test_cleaning_steps_for_eda(run_pre):
    _, _, stats = run_pre
    steps = {s["step"]: s for s in stats["cleaning"]}
    assert list(steps) == ["read_csv", "drop_repeated_headers", "drop_nan_inf", "drop_constant_cols", "drop_duplicates"]
    assert steps["drop_repeated_headers"]["rows_removed"] == 3          # 1 header lặp mỗi file
    assert steps["drop_nan_inf"]["rows_removed"] == 3 * 2               # 2 dòng bẩn mỗi file
    assert steps["drop_duplicates"]["rows_removed"] == 3 * 10           # 10 dòng trùng mỗi file
    assert steps["drop_constant_cols"]["cols"] == steps["drop_nan_inf"]["cols"] - 1


def test_demo_test_parquet(run_pre):
    d, _, stats = run_pre
    demo = pd.read_parquet(d / "p1" / "demo_test.parquet")
    test_counts = stats["label_distribution"]["test"]
    assert 450 <= len(demo) <= 500 and len(demo) == stats["rows"]["demo"]
    assert list(demo.columns) == [ID_COL, TS_COL] + FAKE_FEATURES + [LABEL_COL]
    assert set(demo[ID_COL]) <= set(_read(d, "test")[ID_COL])
    vc = demo[LABEL_COL].value_counts()
    for att in FAKE_ATTACKS:                       # mỗi tấn công >= min(500, số có trong test)
        assert vc.get(att, 0) >= min(500, test_counts.get(att, 0)) - 1


def test_demo_quota():
    from preprocess.spark_preprocess import demo_quota
    q = demo_quota({"Benign": 100000, "Bot": 300, "DoS": 50000}, 10000)
    assert q["Bot"] == 300 and q["DoS"] >= 500 and sum(q.values()) <= 10000
    assert demo_quota({"Benign": 10, "Bot": 5}, 100) == {"Benign": 10, "Bot": 5}
