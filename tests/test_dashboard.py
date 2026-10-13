import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.fake import FAKE_FEATURES, make_fake_flows
from dashboard import view_model as vm

ROOT = Path(__file__).resolve().parents[1]


def _frame():
    return pd.DataFrame({"flow_id": [1, 2, 3, 4], "orig_ts": ["a", "b", "c", "d"],
                         "label": ["Benign", "Bot", "Benign", "Bot"], "score": [0.1, 5.0, 2.0, 0.2],
                         "ts": [1000, 1500, 9000, 10500]})


def test_kpis_and_threshold():
    k = vm.kpis(_frame(), thr=1.0, window_s=2)
    assert (k["total"], k["alerts"], k["alert_rate"]) == (4, 2, 0.5)
    assert k["throughput"] == pytest.approx(1.0)             # 2 dòng có ts > 10500-2000, chia 2 s
    assert vm.kpis(_frame(), thr=10.0)["alerts"] == 0
    assert vm.kpis(_frame().head(0), thr=1.0)["total"] == 0


def test_per_second_and_top():
    ps = vm.per_second(_frame(), 1.0)
    assert ps["flows"].tolist() == [2, 1, 1] and ps["alerts"].tolist() == [1, 1, 0]
    t = vm.top_suspicious(_frame(), k=2)
    assert t["flow_id"].tolist() == [2, 3] and "label" not in t.columns


def test_evaluation_changes_with_threshold():
    a = vm.evaluation(_frame(), 1.0)
    assert a["metrics"]["precision"] == 0.5 and a["metrics"]["recall"] == 0.5
    assert a["confusion"].to_numpy().tolist() == [[1, 1], [1, 1]]
    assert a["per_attack"]["attack"].tolist() == ["Bot"]
    b = vm.evaluation(_frame(), 0.15)
    assert b["metrics"]["recall"] == 1.0


def test_replay_feed_orders_and_emit():
    df = make_fake_flows(200)
    f = vm.ReplayFeed(df, order="shuffle")
    assert sorted(f.order.tolist()) == list(range(200))
    assert vm.ReplayFeed(df).order.tolist() == f.order.tolist()        # seed 42 cố định
    idx = f.emit(50, 1234)
    assert len(idx) == 50 and f.pos == 50 and (f.emit_ts[idx] == 1234).all()
    f.emit(10_000, 99)
    assert f.done and f.pos == 200
    f.reset()
    assert f.pos == 0
    ts = vm.ReplayFeed(df, order="orig_ts")
    t = pd.to_datetime(df["orig_ts"].iloc[ts.order], format="%d/%m/%Y %H:%M:%S")
    assert t.is_monotonic_increasing


def test_stream_frame():
    out = pd.DataFrame({"flow_id": [1], "orig_ts": ["x"], "label": ["Bot"], "recon_error": [2.5],
                        "is_anomaly": [True], "produce_ts": [1], "kafka_ts": [2], "detect_ts": [3]})
    f = vm.stream_frame(out)
    assert list(f.columns) == vm.FRAME_COLUMNS and f["score"].tolist() == [2.5] and f["ts"].tolist() == [3]
    assert list(vm.stream_frame(out.head(0)).columns) == vm.FRAME_COLUMNS


def test_research_assets(tmp_path):
    assert vm.research_assets(tmp_path) == {"comparison_md": None, "figures": [], "eda": []}
    (tmp_path / "eda").mkdir()
    (tmp_path / "model_comparison.md").write_text("| model |\n", encoding="utf-8")
    (tmp_path / "f1_fpr_bar.png").write_bytes(b"x")
    (tmp_path / "eda" / "latent_pca.png").write_bytes(b"x")
    (tmp_path / "eda" / "eda_summary.md").write_text(
        "## Hình\n\n- `latent_pca.png` — PCA của latent.\n- `missing.png` — không có file.\n", encoding="utf-8")
    a = vm.research_assets(tmp_path)
    assert a["comparison_md"].startswith("| model")
    assert [Path(p).name for p, _ in a["figures"]] == ["f1_fpr_bar.png"]
    assert a["eda"] == [(str(tmp_path / "eda" / "latent_pca.png"), "PCA của latent.")]


@pytest.fixture()
def demo_env(tmp_path):
    """Thư mục giống output Kaggle: models/ (onnx + threshold + scaler), results/valscores, data/demo_test.parquet."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from common.fake import write_fake_dataset
    from tests.onnx_fixture import make_mean_square_onnx
    scaler = write_fake_dataset(tmp_path / "split", n=3000, scaler_path=tmp_path / "models" / "scaler.json")
    make_mean_square_onnx(tmp_path / "models" / "ae.onnx", len(FAKE_FEATURES))
    from common.data import load_split
    val = load_split(tmp_path / "split", "val", scaler, benign_only=True)
    vs = (val.X.astype(np.float64) ** 2).mean(axis=1)
    (tmp_path / "results").mkdir()
    np.save(tmp_path / "results" / "valscores_DeepAutoencoder.npy", vs)
    (tmp_path / "models" / "threshold_ae.json").write_text(
        json.dumps({"percentile": 99, "threshold": float(np.percentile(vs, 99)), "model": "DeepAutoencoder"}))
    (tmp_path / "data").mkdir()
    pd.read_parquet(tmp_path / "split" / "test").to_parquet(tmp_path / "data" / "demo_test.parquet", index=False)
    return tmp_path


def test_list_models_and_score_cache(demo_env):
    from streaming.scorer import OnnxScorer
    models = vm.list_models(demo_env / "models")
    assert [m["kind"] for m in models] == ["ae"] and models[0]["percentile"] == 99
    demo = pd.read_parquet(demo_env / "data" / "demo_test.parquet")
    feed = vm.ReplayFeed(demo)
    feed.emit(100, 1)
    cache = vm.ScoreCache(len(demo))
    scorer = OnnxScorer(models[0]["onnx"], demo_env / "models" / "scaler.json")
    s = cache.get("ae", scorer, feed.df, feed.emitted())
    assert np.isfinite(s[feed.emitted()]).all() and np.isnan(s).sum() == len(demo) - 100
    frame = vm.replay_frame(feed, s)
    assert list(frame.columns) == vm.FRAME_COLUMNS and len(frame) == 100


def test_app_replay_interaction(demo_env, monkeypatch):
    """Chạy app thật bằng AppTest: phát nhanh -> có số liệu; đổi ngưỡng -> tab Đánh giá thay đổi."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(demo_env)
    monkeypatch.setattr("sys.argv", ["app.py", "--models-dir", str(demo_env / "models"),
                                     "--results-dir", str(demo_env / "results"),
                                     "--reports-dir", str(demo_env / "reports"),
                                     "--demo", str(demo_env / "data" / "demo_test.parquet")])
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.button(key="burst").click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    n_demo = len(pd.read_parquet(demo_env / "data" / "demo_test.parquet"))
    assert int(metrics["Flow đã xử lý"].replace(",", "")) == min(1000, n_demo)
    recall_before = metrics["Recall"]
    at.radio(key="thr_mode").set_value("Theo percentile val Benign").run()
    at.slider(key="thr_pct").set_value(90.0).run()
    assert not at.exception
    recall_after = {m.label: m.value for m in at.metric}["Recall"]
    assert float(recall_after) >= float(recall_before)
    assert recall_after != recall_before or float(recall_before) == 1.0
    texts = " ".join(md.value for md in at.markdown)
    assert "chưa có" in texts                                 # tab Kết quả nghiên cứu: thiếu file -> "chưa có"
