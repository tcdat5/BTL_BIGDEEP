import json
import zipfile
from pathlib import Path

import pytest

from common.fake import write_fake_raw_csvs
from train.kaggle_run_all import STAGES, Runner, date_tokens, slug

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "train" / "kaggle_notebook.ipynb"


def test_date_tokens_match_both_naming_styles():
    a = date_tokens("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv")
    assert a & date_tokens("02-14-2018.csv") and a & date_tokens("Fake-14-02-2018_x.csv")
    assert not (a & date_tokens("02-16-2018.csv"))


def test_slug():
    assert slug("DDOS attack-HOIC") == "DDOS_attack_HOIC"


def test_find_data_maps_by_date_and_reports_missing(tmp_path):
    write_fake_raw_csvs(tmp_path / "raw" / "sub", n_per_file=50)
    r = Runner(tmp_path / "raw", tmp_path / "work", quick=True)
    files = r.find_data()                              # QUICK: chỉ file đầu (14-02) -> khớp theo ngày
    assert [p.name for p in files] == ["Fake-14-02-2018_TrafficForML_CICFlowMeter.csv"]
    full = Runner(tmp_path / "raw", tmp_path / "work")
    with pytest.raises(FileNotFoundError, match="FILE_MAP"):
        full.find_data()                               # 21-02 và 02-03 không có
    mapped = Runner(tmp_path / "raw", tmp_path / "work", file_map={
        "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv": "Fake-16-02-2018_TrafficForML_CICFlowMeter.csv",
        "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv": "Fake-20-02-2018_TrafficForML_CICFlowMeter.csv"})
    assert len(mapped.find_data()) == 4


def test_relative_dirs_become_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_fake_raw_csvs(tmp_path / "raw", n_per_file=50)
    r = Runner("raw", "work", quick=True)
    assert r.raw_dir.is_absolute() and r.work.is_absolute()
    assert all(p.is_absolute() for p in r.find_data())


def test_noncritical_failure_continues_critical_stops(tmp_path):
    r = Runner(tmp_path, tmp_path / "work")
    r.st_mlp = lambda: (_ for _ in ()).throw(ValueError("boom"))
    r.st_ae = lambda: (_ for _ in ()).throw(ValueError("boom"))
    assert r.stage("mlp") == "FAILED"
    with pytest.raises(RuntimeError, match="ae"):
        r.stage("ae")
    assert [t["status"] for t in r.timings] == ["FAILED", "FAILED"]
    assert (tmp_path / "work" / "results" / "kaggle_timing.csv").exists()


def test_notebook_cells_compile_and_cover_stages():
    from train.make_notebook import build
    nb = build()
    assert json.loads(NOTEBOOK.read_text(encoding="utf-8")) == json.loads(json.dumps(nb)), \
        "kaggle_notebook.ipynb cũ — chạy python train/make_notebook.py"
    src = ""
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            compile(s, "cell", "exec")
            src += s
    for s in STAGES:
        assert 'R.stage("%s")' % s in src
    assert "R.package()" in src and "QUICK = False" in src


@pytest.mark.spark
@pytest.mark.torch
def test_quick_pipeline_end_to_end(tmp_path):
    pytest.importorskip("pyspark")
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    write_fake_raw_csvs(tmp_path / "input" / "cicids", n_per_file=1500)
    r = Runner(tmp_path / "input", tmp_path / "working", quick=True)
    zpath = r.run_all()
    status = {t["stage"]: t["status"] for t in r.timings}
    for s in ("preprocess", "ae", "mlp", "vae", "dae", "baselines", "holdout"):
        assert status[s] == "ok", status
    names = set(zipfile.ZipFile(zpath).namelist())
    for f in ("models/ae.onnx", "models/threshold.json", "models/scaler.json", "models/ae.pt", "models/vae.pt",
              "data/demo_test.parquet", "results/baselines_metrics.csv", "results/dl_ae_metrics.csv",
              "results/preprocess_stats.json", "results/kaggle_timing.csv",
              "results/holdout/SSH_Bruteforce/dl_mlp_per_attack.csv",
              "results/holdout/SSH_Bruteforce/baselines_metrics_holdout-SSH-Bruteforce.csv",
              "results/holdout/Bot/dl_mlp_per_attack.csv"):
        assert f in names, f
    # dữ liệu giả không có DDOS attack-HOIC -> holdout loại này bị bỏ qua (có cảnh báo), không lỗi
    assert not any(n.startswith("results/holdout/DDOS_attack_HOIC/") for n in names)
