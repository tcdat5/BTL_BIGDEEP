# Real-time Network Anomaly Detection (BTL_BigDeep) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hệ thống phát hiện bất thường mạng thời gian thực (CSE-CIC-IDS2018) dùng Deep Autoencoder: tiền xử lý Spark trên HDFS → train 4 model DL + 6 baseline ML trên Kaggle → export ONNX → Kafka + Spark Structured Streaming suy luận trên cụm 3 VM → dashboard Streamlit + benchmark + bảng/hình cho báo cáo.

**Architecture:** Thư mục `common/` (thuần numpy/pandas, chạy được trên Python 3.8) là nguồn duy nhất cho scaler, load_split, metric. Phần cụm (preprocess, streaming, dashboard) chạy Python 3.8 + Spark 3.5.1; phần train (dl_models, train_dl, ablation, baselines, report) chạy trên Kaggle với PyTorch. Hai thế giới nối với nhau qua file: Parquet + `scaler.json` (cụm → Kaggle) và `*.onnx` + `threshold.json` (Kaggle → cụm). Mọi graph ONNX trả thẳng `score`, streaming chỉ so `score >= threshold`.

**Tech Stack:** Python 3.8 (cụm) / 3.11+ (Kaggle), PySpark 3.5.1, Kafka 3.7.2 KRaft, kafka-python, onnxruntime, PyTorch, scikit-learn, LightGBM, Streamlit, matplotlib, pytest, Docker (test môi trường cụm ở local).

## Global Constraints

- Spec gốc: `PROJECT_SPEC.md` ở gốc repo — đọc mục 4 (quy ước dữ liệu) trước mọi task.
- Cụm: Spark **3.5.1 (Scala 2.13)**, Java **8** (OpenJDK 1.8.0_402), Hadoop **3.3.0**, Python **3.8.10**.
- Kafka **3.7.2** (Scala 2.13), KRaft, single broker `master:9092` (giá trị này chỉ nằm trong `config*.yaml`, không nằm trong code).
- Connector Kafka trên cụm: `org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1` (**_2.13**). Pip `pyspark==3.5.1` ở local/Docker là build Scala 2.12 → test local dùng `_2.12`. Luôn lấy từ config key `cluster.kafka_package`.
- Code chạy trên cụm (`common/`, `preprocess/`, `streaming/`, `dashboard/`, `scripts/`) phải chạy được trên **Python 3.8**: mọi file bắt đầu bằng `from __future__ import annotations`; không dùng `match`, `X | Y` ngoài annotation, `str.removeprefix`, `dict | dict`, `functools.cache`, `zip(strict=)`, `list[int]` ngoài annotation.
- `requirements-cluster.txt` phải pin bản còn hỗ trợ Python 3.8, kiểm chứng bằng `pip install` thật trong Docker `python:3.8`.
- ONNX: opset **17**, `ir_version <= 8` (onnxruntime cũ trên cụm).
- Smoke test local cho Spark dùng **Java 8 hoặc 11** (máy dev có Java 26 → không dùng; chạy trong Docker image có OpenJDK 11).
- Seed 42 ở mọi chỗ ngẫu nhiên. Transform: `clip((x - mean) / std, -10, 10)` float32, `std == 0 → 1`, scaler fit **chỉ trên Benign của train**, std = độ lệch chuẩn quần thể (ddof=0, = Spark `stddev_pop`).
- Ngưỡng unsupervised = percentile 99 score trên **val Benign**; không chỉnh ngưỡng bằng test.
- Test set = `common.data.load_split(..., "test", cap=300000)` với `random_state=42`.
- Không bịa kết quả: README chỉ chứa số do code đã chạy sinh ra.
- Mọi script: `argparse` + `--help`, đọc config qua `common.config`, không hard-code IP/đường dẫn.
- Không cài PyTorch trên cụm. Không `collect()`/`toPandas()` toàn bộ dữ liệu lớn (chỉ trên micro-batch có giới hạn `maxOffsetsPerTrigger` hoặc kết quả agg nhỏ).
- Chính sách giá trị lỗi (ghi README): dòng có null/NaN/±Infinity ở **bất kỳ** cột feature nào bị **bỏ**.
- Inference streaming (ghi README): mặc định `foreachBatch` + `mapInPandas` chạy **trên executor** (slave1/slave2 phải cài `requirements-cluster.txt`; model/scaler/threshold phân phối qua `SparkContext.addFile`, code qua `addPyFile`). Fallback `--inference driver`: `toPandas()` micro-batch rồi suy luận trên driver (chỉ master cần onnxruntime).
- Ưu tiên khi thiếu giờ: AE > MLP > VAE > ablation latent/ngưỡng > DAE > ablation độ sâu > VAE β. Streaming chỉ cần AE.
- Commit cuối mỗi task, message kết thúc bằng dòng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

## Lịch (hôm nay thứ 2, 12/10/2026 — deadline sáng thứ 6, 16/10)

| Milestone | Hạn | Tasks |
|---|---|---|
| M0 | Thứ 2 tối | 1, 2, 3, 4 (+ Phụ lục A bước A1–A4 do người dùng làm trên cụm song song) |
| M1 | Thứ 3 trưa | 5 (+ A5: chạy preprocess trên cụm, upload mẫu lên Kaggle) |
| M1b | Thứ 3 chiều | 5b (chạy trên Kaggle CPU song song lúc GPU train) |
| M2a | Thứ 3 chiều | 6, 7, 8, 9 (+ chạy trên Kaggle) |
| M2b | Thứ 3 tối | 10, 11, 11b |
| M3 | Thứ 4 chiều | 12, 13, 14 (+ A6: chạy end-to-end trên cụm) |
| M4 | Thứ 4 tối | 15 |
| M5 | Thứ 5 sáng | 16 (+ A7: chạy benchmark trên cụm) |
| Buffer | Thứ 5 chiều | 17 |

## Bản đồ file

| File | Trách nhiệm | Môi trường |
|---|---|---|
| `common/config.py` | `load_config`, `get` (dotted key) | cụm + Kaggle |
| `common/data.py` | hằng số cột, `Scaler`, `load_scaler`, `fit_scaler`, `transform`, `frame_to_matrix`, `binarize`, `Split`, `load_split` | cụm + Kaggle |
| `common/metrics.py` | `confusion_metrics` (numpy), `evaluate` (sklearn lazy), `per_attack_recall`, `save_scores`, `METRIC_COLUMNS` | cụm + Kaggle |
| `common/fake.py` | sinh dữ liệu giả (Parquet đã xử lý + CSV thô bẩn) cho smoke test | cụm + Kaggle |
| `preprocess/spark_preprocess.py` | CSV → Parquet train/val/test + scaler + mẫu Kaggle + stats | cụm |
| `train/baselines.py` | 6 baseline (file người dùng cung cấp, refactor dùng `common/`) | Kaggle |
| `train/dl_models.py` | AE/VAE/DAE/MLP + module score cho ONNX | Kaggle |
| `train/train_dl.py` | train/eval/export 1 model DL | Kaggle |
| `train/run_ablation.py` | lưới ablation AE (+VAE β) | Kaggle |
| `train/kaggle_run_all.sh`, `train/kaggle_notebook.ipynb` | chạy toàn bộ phần train trên Kaggle | Kaggle |
| `analysis/make_report_assets.py` | bảng + hình → `reports/` | Kaggle/laptop |
| `analysis/eda.py` | trực quan hoá dữ liệu (phần 1) + latent space (phần 2 `--latent`) → `reports/eda/` | Kaggle CPU/laptop |
| `streaming/producer.py` | replay Parquet test → Kafka | cụm (master) |
| `streaming/scorer.py` | `OnnxScorer`, `get_scorer` (cache/process), `score_frame` | cụm |
| `streaming/output_io.py` | `OUTPUT_COLUMNS`, `load_output` | cụm |
| `streaming/stream_detect.py` | Structured Streaming + foreachBatch | cụm |
| `streaming/benchmark.py` | throughput/latency/F1 + hình | cụm/laptop |
| `dashboard/view_model.py`, `dashboard/app.py` | logic thuần + UI Streamlit | cụm (master) |
| `scripts/cfg.py`, `scripts/check_cluster.sh`, `scripts/run_preprocess.sh`, `scripts/run_stream.sh`, `scripts/run_benchmark.sh` | vận hành cụm | cụm |
| `docker/Dockerfile.cluster-test`, `docker/docker-compose.test.yml`, `tests/config.test.yaml` | môi trường test giống cụm (Py3.8 + Java 11 + Kafka 3.7.2) | dev |
| `tests/*` | pytest | dev |

## Interfaces dùng chung (tên chốt — mọi task dùng đúng các tên này)

```python
# common/config.py
REPO_ROOT: Path
def load_config(path=None) -> dict          # path -> $BTL_CONFIG -> config.yaml -> config.example.yaml
def get(cfg: dict, dotted: str, default=None)

# common/data.py
LABEL_COL = "Label"; BENIGN = "Benign"; TS_COL = "orig_ts"; ID_COL = "flow_id"
META_COLS = (ID_COL, TS_COL, LABEL_COL); CLIP = 10.0; SEED = 42
@dataclass class Scaler: features: List[str]; mean: np.ndarray; std: np.ndarray; def to_json(self, path)
def load_scaler(path) -> Scaler
def fit_scaler(df: pd.DataFrame, features) -> Scaler
def transform(X, scaler) -> np.ndarray[float32]
def frame_to_matrix(df, scaler) -> np.ndarray[float32]
def binarize(labels) -> np.ndarray[int64]
def feature_columns(df) -> list
@dataclass class Split: X; y; labels; meta; def subset(self, mask) -> Split
def read_split_frame(data_dir, split, cap=None, seed=SEED) -> pd.DataFrame
def load_split(data_dir, split, scaler, cap=None, seed=SEED, benign_only=False) -> Split

# common/metrics.py
METRIC_COLUMNS = ["model","group","precision","recall","f1","fpr","roc_auc","pr_auc","threshold","train_time_s","infer_ms_per_1k"]
def confusion_metrics(y, y_pred) -> dict   # precision recall f1 fpr tp fp tn fn
def evaluate(y, score, thr) -> dict        # precision recall f1 fpr roc_auc pr_auc threshold
def per_attack_recall(model, labels, y_pred) -> pd.DataFrame  # model attack recall n
def save_scores(path, y, score, thr)

# common/fake.py
FAKE_FEATURES: list; FAKE_ATTACKS: list
def make_fake_flows(n=4000, attack_frac=0.2, seed=SEED) -> pd.DataFrame
def write_fake_dataset(out_dir, n=4000, seed=SEED, scaler_path=None) -> Scaler
def write_fake_raw_csvs(out_dir, n_per_file=1500, seed=SEED) -> list   # list[str] tên file

# train/dl_models.py
MODEL_SPECS = {"ae": {"name":"DeepAutoencoder","group":"unsupervised"}, "vae": {...VariationalAE}, "dae": {...DenoisingAE}, "mlp": {"name":"DNN_Classifier","group":"supervised"}}
def build_model(kind, d, hidden=(64,32), latent=16, noise_std=0.1) -> nn.Module
def score_module(kind, model) -> nn.Module   # forward(x) -> score [batch]
def vae_loss(x, recon, mu, logvar, beta) -> (loss, recon_loss, kl)
def count_params(model) -> int
def encode_latent(model, x) -> Tensor          # AE/DAE: encoder(x); VAE: μ
def save_checkpoint(path, kind, model, d, hidden, latent, noise_std); def load_checkpoint(path) -> (kind, model)

# train/train_dl.py
def parse_args(argv=None) -> argparse.Namespace
def run(args) -> dict        # metric row (METRIC_COLUMNS + n_params)
def main(argv=None)

# streaming/scorer.py
class OnnxScorer: __init__(model_path, scaler_path, threshold_path); threshold: float; score_matrix(raw) -> np.ndarray
def get_scorer(model_path, scaler_path, threshold_path) -> OnnxScorer   # cache theo process
PASSTHROUGH_COLS = ["flow_id","orig_ts","label","produce_ts","kafka_ts"]
def score_frame(pdf, scorer, feature_cols) -> pd.DataFrame   # PASSTHROUGH_COLS + recon_error + is_anomaly

# streaming/output_io.py
OUTPUT_COLUMNS = ["flow_id","orig_ts","label","recon_error","is_anomaly","produce_ts","kafka_ts","detect_ts"]
def load_output(out_dir, max_files=None) -> pd.DataFrame

# streaming/benchmark.py
def summarize(df) -> dict
```

File output (khớp spec 5.2/5.3): `results/baselines_metrics.csv`, `results/baselines_per_attack.csv`, `results/dl_<kind>_metrics.csv` (= `METRIC_COLUMNS + ["n_params"]`), `results/dl_<kind>_per_attack.csv`, `results/dl_<kind>_history.csv`, `results/scores_<Name>.npz` (`y`, `score`, `thr`), `results/valscores_<Name>.npy` (score val Benign, cho ablation ngưỡng), `models/<kind>.onnx`, `models/<kind>.pt` (state_dict + kiến trúc, cho EDA latent), `models/threshold_<kind>.json`, `models/ae.onnx`, `models/threshold.json`, `<results_dir>/holdout.json` (`{"attack": ...}`).

---

## Phụ lục A — Việc người dùng làm trên cụm (agent không SSH được vào VM)

Agent chuẩn bị lệnh, người dùng chạy và dán kết quả lại. Mỗi bước có tiêu chí xong.

**A1. Cài Kafka 3.7.2 KRaft trên `master`** (Kafka 3.7 vẫn chạy trên Java 8):
```bash
cd /opt && sudo wget https://archive.apache.org/dist/kafka/3.7.2/kafka_2.13-3.7.2.tgz
sudo tar xzf kafka_2.13-3.7.2.tgz && sudo ln -sfn /opt/kafka_2.13-3.7.2 /opt/kafka && sudo chown -R "$USER" /opt/kafka_2.13-3.7.2
mkdir -p /opt/kafka-data
# Sửa /opt/kafka/config/kraft/server.properties:
#   listeners=PLAINTEXT://:9092,CONTROLLER://:9093
#   advertised.listeners=PLAINTEXT://master:9092
#   log.dirs=/opt/kafka-data
#   num.partitions=3
KAFKA_CLUSTER_ID=$(/opt/kafka/bin/kafka-storage.sh random-uuid)
/opt/kafka/bin/kafka-storage.sh format -t "$KAFKA_CLUSTER_ID" -c /opt/kafka/config/kraft/server.properties
KAFKA_HEAP_OPTS="-Xmx512m -Xms512m" /opt/kafka/bin/kafka-server-start.sh -daemon /opt/kafka/config/kraft/server.properties
/opt/kafka/bin/kafka-topics.sh --bootstrap-server master:9092 --list   # không lỗi là OK
```
Xong khi: lệnh `--list` chạy không lỗi từ cả `master` và `slave1`.

**A2. Cài Python package trên cả 3 VM** (sau Task 1):
```bash
python3 -m pip install --user -r requirements-cluster.txt
python3 -c "import onnxruntime, pandas, pyarrow, kafka, yaml; print('ok')"
```
Và trong `$SPARK_HOME/conf/spark-env.sh` của cả 3 VM: `export PYSPARK_PYTHON=python3`.

**A3. Đưa CSV lên HDFS:**
```bash
hdfs dfs -mkdir -p /user/$USER/cicids2018/raw
hdfs dfs -put Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv Friday-16-02-2018_TrafficForML_CICFlowMeter.csv \
  Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv Friday-02-03-2018_TrafficForML_CICFlowMeter.csv /user/$USER/cicids2018/raw/
```

**A4. Tạo `config.yaml`** từ `config.example.yaml`, sửa hostname/port/đường dẫn HDFS/RAM theo cụm thật; chạy `bash scripts/check_cluster.sh` → cả 3 dòng `[PASS]`.

**A5. (sau Task 5)** `bash scripts/run_preprocess.sh` → upload thư mục `data/kaggle_sample/` (train/, val/, test/, scaler.json, preprocess_stats.json) thành Kaggle Dataset tên `btl-cicids2018-sample`; copy `models/scaler.json` và `results/preprocess_stats.json` về repo.

**A6. (sau Task 14)** Copy `models/ae.onnx`, `models/threshold.json` từ output Kaggle vào `models/` trên master; `hdfs dfs -get <processed_dir>/test data/processed/test`; chạy theo README mục "Streaming".

**A7. (sau Task 16)** `RATES="1000 5000 10000 20000" WORKERS=2 bash scripts/run_benchmark.sh` rồi `WORKERS=1 ...`.

---

### Task 1: Khung repo, config, requirements, môi trường test

**Files:**
- Create: `.gitignore`, `.gitattributes`, `pytest.ini`, `conftest.py`
- Create: `requirements-cluster.txt`, `requirements-kaggle.txt`
- Create: `config.example.yaml`, `tests/config.test.yaml`
- Create: `common/__init__.py`, `common/config.py`
- Create: `train/__init__.py`, `streaming/__init__.py`, `dashboard/__init__.py`, `tests/__init__.py`
- Create: `docker/Dockerfile.cluster-test`, `docker/docker-compose.test.yml`
- Create: `scripts/cfg.py`
- Create: `README.md`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `common.config.load_config(path=None) -> dict`, `common.config.get(cfg, dotted, default=None)`, `common.config.REPO_ROOT`; CLI `python scripts/cfg.py <dotted.key>` in giá trị ra stdout. Các key config dùng ở mọi task (xem `config.example.yaml` bên dưới).

- [ ] **Step 1: Tạo file khung**

`.gitattributes` (bắt buộc — Windows checkout CRLF làm hỏng bash trên VM):
```
*.sh text eol=lf
*.py text eol=lf
*.yaml text eol=lf
*.yml text eol=lf
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv*/
.pytest_cache/
.cache/
config.yaml
data/
checkpoints/
spark-warehouse/
metastore_db/
derby.log
.ipynb_checkpoints/
models/*.onnx
models/*.pt
results/**/*.npz
results/**/*.npy
results/stream/
results/ablation/
results/holdout/models/
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
markers =
    spark: cần pyspark + Java 8/11 (chạy trong docker cluster-test)
    kafka: cần Kafka broker tại $KAFKA_BOOTSTRAP
    torch: cần torch/onnx (venv Kaggle/local)
```

`conftest.py`:
```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

`common/__init__.py`, `train/__init__.py`, `streaming/__init__.py`, `dashboard/__init__.py`, `tests/__init__.py`: file rỗng.

`requirements-cluster.txt` (Python 3.8 — Step 6 kiểm chứng):
```
pyspark==3.5.1
kafka-python==2.0.6
onnxruntime==1.17.3
numpy==1.24.4
pandas==2.0.3
pyarrow==14.0.2
pyyaml==6.0.1
streamlit==1.39.0
matplotlib==3.7.5
```

`requirements-kaggle.txt` (Kaggle đã có sẵn đa số; dùng cho venv local):
```
torch>=2.1
onnx>=1.15
onnxruntime>=1.17
scikit-learn>=1.3
lightgbm>=4.0
pandas>=2.0
pyarrow>=14
numpy>=1.24
matplotlib>=3.7
pyyaml>=6.0
pytest>=8.0
```

`config.example.yaml`:
```yaml
# Copy thành config.yaml rồi sửa theo cụm thật. config.yaml KHÔNG commit.
cluster:
  spark_master: spark://master:7077
  spark_master_ui: http://master:8080
  expected_workers: 2
  expected_datanodes: 2
  kafka_bootstrap: master:9092
  kafka_home: /opt/kafka
  # Spark trên cụm build Scala 2.13 -> _2.13 (pip pyspark local là Scala 2.12 -> _2.12)
  kafka_package: org.apache.spark:spark-sql-kafka-0-10_2.13:3.5.1
spark:
  driver_memory: 1g
  executor_memory: 1g
  executor_cores: 1
  total_executor_cores: 2
  shuffle_partitions: 12
hdfs:
  raw_dir: hdfs:///user/hadoop/cicids2018/raw
  processed_dir: hdfs:///user/hadoop/cicids2018/processed
  sample_dir: hdfs:///user/hadoop/cicids2018/kaggle_sample
data:
  raw_files:
    - Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv
    - Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
    - Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
    - Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
  local_dir: data/processed
  sample_local_dir: data/kaggle_sample
  test_cap: 300000
  sample_sizes:
    train: 500000
    val: 200000
    test: 300000
models:
  scaler: models/scaler.json
  stream_model: models/ae.onnx
  stream_threshold: models/threshold.json
kafka:
  topic: network-flows
  partitions: 3
streaming:
  output_dir: results/stream/default
  checkpoint_dir: checkpoints/stream_default
  batches_csv: results/stream_batches.csv
  trigger_s: 2
  max_offsets_per_trigger: 20000
  starting_offsets: latest
  inference: executor
dashboard:
  refresh_s: 3
  max_files: 2000
```

`tests/config.test.yaml`: copy y nguyên `config.example.yaml` rồi đổi đúng các dòng sau:
```yaml
cluster:
  spark_master: local[2]
  kafka_bootstrap: kafka:9092
  kafka_package: org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1
spark:
  driver_memory: 1g
  shuffle_partitions: 4
```

- [ ] **Step 2: Viết test fail cho `common/config.py`**

`tests/test_config.py`:
```python
import pytest

from common.config import REPO_ROOT, get, load_config

REQUIRED_KEYS = [
    "cluster.spark_master", "cluster.spark_master_ui", "cluster.expected_workers",
    "cluster.expected_datanodes", "cluster.kafka_bootstrap", "cluster.kafka_home",
    "cluster.kafka_package", "spark.executor_memory", "spark.driver_memory",
    "spark.executor_cores", "spark.total_executor_cores", "spark.shuffle_partitions",
    "hdfs.raw_dir", "hdfs.processed_dir", "hdfs.sample_dir", "data.raw_files",
    "data.local_dir", "data.sample_local_dir", "data.test_cap", "data.sample_sizes.train",
    "models.scaler", "models.stream_model", "models.stream_threshold", "kafka.topic",
    "kafka.partitions", "streaming.output_dir", "streaming.checkpoint_dir",
    "streaming.batches_csv", "streaming.trigger_s", "streaming.max_offsets_per_trigger",
    "streaming.starting_offsets", "streaming.inference", "dashboard.refresh_s",
    "dashboard.max_files",
]


def test_load_explicit_and_get(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("kafka:\n  topic: t1\n", encoding="utf-8")
    cfg = load_config(p)
    assert get(cfg, "kafka.topic") == "t1"
    assert get(cfg, "kafka.missing", 5) == 5
    assert get(cfg, "nope.deeper") is None


def test_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize("name", ["config.example.yaml", "tests/config.test.yaml"])
def test_shipped_configs_have_required_keys(name):
    cfg = load_config(REPO_ROOT / name)
    missing = [k for k in REQUIRED_KEYS if get(cfg, k) is None]
    assert missing == []
```

- [ ] **Step 3: Chạy test, xác nhận fail**

Tạo venv local (Windows, PowerShell):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-kaggle.txt
.venv\Scripts\python -m pytest tests/test_config.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'common.config'`.

- [ ] **Step 4: Viết `common/config.py`**

```python
"""Đọc config.yaml. Thứ tự tìm: tham số path -> $BTL_CONFIG -> config.yaml -> config.example.yaml."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    if path is not None:
        candidates = [Path(path)]
        if not candidates[0].is_file():
            raise FileNotFoundError("config not found: %s" % path)
    else:
        candidates = [Path(p) for p in (os.environ.get("BTL_CONFIG"),
                                        REPO_ROOT / "config.yaml",
                                        REPO_ROOT / "config.example.yaml") if p]
    for c in candidates:
        if c.is_file():
            with open(c, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            cfg["_path"] = str(c)
            return cfg
    raise FileNotFoundError("no config.yaml / config.example.yaml found")


def get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
```

`scripts/cfg.py`:
```python
"""In một giá trị config ra stdout. Dùng trong bash: python3 scripts/cfg.py cluster.kafka_bootstrap"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import get, load_config  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: cfg.py <dotted.key>   (config: $BTL_CONFIG hoặc config.yaml)")
        sys.exit(2)
    value = get(load_config(), sys.argv[1])
    if value is None:
        sys.stderr.write("missing config key: %s\n" % sys.argv[1])
        sys.exit(1)
    print(" ".join(map(str, value)) if isinstance(value, list) else value)
```

- [ ] **Step 5: Chạy test, xác nhận pass**

Run: `.venv\Scripts\python -m pytest tests/test_config.py -q`
Expected: `4 passed`.

- [ ] **Step 6: Docker môi trường cụm + kiểm chứng pin Python 3.8**

`docker/Dockerfile.cluster-test`:
```dockerfile
# Mô phỏng VM: Python 3.8 + Java 8 + requirements-cluster.txt
# JRE chép từ image Temurin (mirror apt của Debian bullseye đã bị archive, không cài được openjdk qua apt).
FROM python:3.8-slim-bullseye
COPY --from=eclipse-temurin:8-jre /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="/opt/java/openjdk/bin:${PATH}"
COPY requirements-cluster.txt /tmp/requirements-cluster.txt
RUN pip install --no-cache-dir -r /tmp/requirements-cluster.txt pytest==8.3.5 onnx==1.16.2
WORKDIR /repo
```

`docker/docker-compose.test.yml`:
```yaml
# Tên project riêng để không đụng container của project compose khác cùng tên thư mục "docker".
name: btl-bigdeep-test
services:
  kafka:
    image: apache/kafka:3.7.2
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@localhost:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: 0
      KAFKA_NUM_PARTITIONS: 3
  cluster:
    build:
      context: ..
      dockerfile: docker/Dockerfile.cluster-test
    volumes:
      - ..:/repo
      - ivy-cache:/root/.ivy2
    environment:
      BTL_CONFIG: /repo/tests/config.test.yaml
      KAFKA_BOOTSTRAP: kafka:9092
      # spark.jars.packages chỉ có tác dụng khi JVM khởi động -> nạp connector ngay từ đầu cho mọi test Spark
      PYSPARK_SUBMIT_ARGS: --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 pyspark-shell
    depends_on:
      - kafka
    working_dir: /repo
volumes:
  ivy-cache: {}
```

Run: `docker compose -f docker/docker-compose.test.yml build cluster`
Expected: build thành công. Nếu một pin không có wheel cho Python 3.8 (pip báo `No matching distribution`), chạy `docker run --rm python:3.8-slim-bullseye pip index versions <pkg>` để lấy bản cao nhất còn hỗ trợ 3.8, sửa `requirements-cluster.txt`, build lại. Ghi bản cuối cùng vào README.

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_config.py -q`
Expected: `4 passed` (trên Python 3.8).

- [ ] **Step 7: README khung**

`README.md`:
````markdown
# BTL_BigDeep — Real-time Network Anomaly Detection (CSE-CIC-IDS2018)

Deep Autoencoder + Kafka + Spark Structured Streaming trên cụm 3 VM. Spec đầy đủ: `PROJECT_SPEC.md`.
Kế hoạch: `docs/superpowers/plans/2026-10-12-network-anomaly-detection.md`.

## Môi trường
| Nơi | Python | Ghi chú |
|---|---|---|
| Cụm (master, slave1, slave2) | 3.8.10 | Spark 3.5.1 (Scala 2.13), Java 8, Hadoop 3.3.0, Kafka 3.7.2 KRaft trên master |
| Kaggle | 3.11+ | GPU cho DL, CPU cho baseline |
| Dev local | 3.12 venv (phần train) + Docker (phần cụm: Python 3.8 + Java 11 + Kafka 3.7.2) | |

## Cài đặt
```bash
cp config.example.yaml config.yaml        # sửa hostname/port/đường dẫn HDFS/RAM
python3 -m pip install --user -r requirements-cluster.txt   # trên CẢ 3 VM
```

## Test local
```powershell
.venv\Scripts\python -m pytest -q                                              # phần train + common
docker compose -f docker/docker-compose.test.yml run --rm cluster pytest -q    # phần cụm (Py3.8, Spark, Kafka)
```
````

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: repo skeleton, config loader, requirements, docker test env" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `common/data.py` + `common/fake.py`

**Files:**
- Create: `common/data.py`, `common/fake.py`
- Test: `tests/test_data.py`, `tests/test_fake.py`

**Interfaces:**
- Consumes: không.
- Produces: toàn bộ API `common/data.py` và `common/fake.py` ở mục "Interfaces dùng chung". `python -m common.fake --out DIR --scaler PATH --n 4000` ghi dataset giả.

- [ ] **Step 1: Viết test fail**

`tests/test_data.py`:
```python
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
```

`tests/test_fake.py`:
```python
import pandas as pd

from common.data import BENIGN, ID_COL, LABEL_COL, TS_COL
from common.fake import FAKE_ATTACKS, make_fake_flows, write_fake_raw_csvs


def test_make_fake_flows_shape_and_labels():
    df = make_fake_flows(1000, seed=1)
    assert len(df) == 1000 and df[ID_COL].is_unique
    assert set(df[LABEL_COL]) == {BENIGN, *FAKE_ATTACKS}
    assert (df[LABEL_COL] == BENIGN).mean() > 0.7
    assert df[TS_COL].str.match(r"\d{2}/\d{2}/2018 \d{2}:\d{2}:\d{2}").all()


def test_make_fake_flows_deterministic():
    pd.testing.assert_frame_equal(make_fake_flows(300, seed=7), make_fake_flows(300, seed=7))


def test_write_fake_raw_csvs_has_dirty_rows(tmp_path):
    files = write_fake_raw_csvs(tmp_path, n_per_file=200)
    assert len(files) == 3
    raw = pd.read_csv(tmp_path / files[2], dtype=str)
    assert {"Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Const Col"} <= set(raw.columns)
    assert (raw["Label"] == "Label").sum() == 1
    assert (raw["Flow Byts/s"] == "Infinity").sum() == 1
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `.venv\Scripts\python -m pytest tests/test_data.py tests/test_fake.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'common.data'`.

- [ ] **Step 3: Viết `common/data.py`**

```python
"""Quy ước dữ liệu dùng CHUNG cho baseline, DL và streaming (PROJECT_SPEC mục 4).

Mọi chỗ chuẩn hoá feature phải đi qua transform() ở đây.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

LABEL_COL = "Label"
BENIGN = "Benign"
TS_COL = "orig_ts"
ID_COL = "flow_id"
META_COLS = (ID_COL, TS_COL, LABEL_COL)
CLIP = 10.0
SEED = 42


@dataclass
class Scaler:
    features: List[str]
    mean: np.ndarray
    std: np.ndarray

    def to_json(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"features": list(self.features),
                       "mean": [float(v) for v in self.mean],
                       "std": [float(v) for v in self.std]}, f, indent=1)


def _fix_std(std) -> np.ndarray:
    std = np.asarray(std, dtype=np.float64).copy()
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return std


def load_scaler(path) -> Scaler:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return Scaler(list(d["features"]), np.asarray(d["mean"], dtype=np.float64), _fix_std(d["std"]))


def fit_scaler(df: pd.DataFrame, features) -> Scaler:
    """Fit trên các dòng Benign; std quần thể (ddof=0) giống Spark stddev_pop; std 0 -> 1."""
    features = list(features)
    benign = df.loc[df[LABEL_COL] == BENIGN, features].to_numpy(dtype=np.float64)
    if len(benign) == 0:
        raise ValueError("no Benign rows to fit scaler")
    return Scaler(features, benign.mean(axis=0), _fix_std(benign.std(axis=0)))


def transform(X, scaler: Scaler) -> np.ndarray:
    Z = (np.asarray(X, dtype=np.float64) - scaler.mean) / scaler.std
    return np.clip(Z, -CLIP, CLIP).astype(np.float32)


def frame_to_matrix(df: pd.DataFrame, scaler: Scaler) -> np.ndarray:
    return transform(df[scaler.features].to_numpy(dtype=np.float64), scaler)


def binarize(labels) -> np.ndarray:
    return (np.asarray(labels) != BENIGN).astype(np.int64)


def feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in META_COLS]


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    meta: pd.DataFrame

    def subset(self, mask) -> "Split":
        mask = np.asarray(mask, dtype=bool)
        return Split(self.X[mask], self.y[mask], self.labels[mask],
                     self.meta[mask].reset_index(drop=True))


def read_split_frame(data_dir, split: str, cap=None, seed: int = SEED) -> pd.DataFrame:
    path = Path(data_dir) / split
    if not path.exists():
        raise FileNotFoundError("split not found: %s" % path)
    df = pd.read_parquet(path)
    if ID_COL in df.columns:   # thứ tự ổn định, không phụ thuộc thứ tự file part-*
        df = df.sort_values(ID_COL, kind="mergesort")
    df = df.reset_index(drop=True)
    if cap is not None and len(df) > cap:
        df = df.sample(n=cap, random_state=seed).reset_index(drop=True)
    return df


def load_split(data_dir, split: str, scaler: Scaler, cap=None, seed: int = SEED,
               benign_only: bool = False) -> Split:
    df = read_split_frame(data_dir, split, cap, seed)
    if benign_only:
        df = df[df[LABEL_COL] == BENIGN].reset_index(drop=True)
    labels = df[LABEL_COL].astype(str).to_numpy()
    meta = df[[c for c in META_COLS if c in df.columns]].reset_index(drop=True)
    return Split(frame_to_matrix(df, scaler), binarize(labels), labels, meta)
```

- [ ] **Step 4: Viết `common/fake.py`**

```python
"""Dữ liệu giả nhỏ cho smoke test (giống cấu trúc CSE-CIC-IDS2018).

python -m common.fake --out data/fake --scaler data/fake/scaler.json --n 4000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.data import BENIGN, ID_COL, LABEL_COL, SEED, TS_COL, fit_scaler

FAKE_FEATURES = [
    "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts",
    "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Bwd Pkt Len Mean", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Fwd IAT Tot", "Pkt Size Avg", "Init Fwd Win Byts", "Idle Mean", "ACK Flag Cnt",
]
FAKE_ATTACKS = ["DoS attacks-Hulk", "Bot", "FTP-BruteForce", "SSH-Bruteforce"]


def make_fake_flows(n: int = 4000, attack_frac: float = 0.2, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = len(FAKE_FEATURES)
    n_att = int(n * attack_frac)
    parts = [rng.lognormal(mean=3.0, sigma=0.5, size=(n - n_att, d))]
    labels = [BENIGN] * (n - n_att)
    for i, att in enumerate(FAKE_ATTACKS):
        k = n_att // len(FAKE_ATTACKS) + (1 if i < n_att % len(FAKE_ATTACKS) else 0)
        shift = np.zeros(d)
        shift[i * 3:i * 3 + 3] = 2.5    # mỗi loại tấn công lệch ở 3 feature riêng
        parts.append(rng.lognormal(mean=3.0 + shift, sigma=0.5, size=(k, d)))
        labels += [att] * k
    df = pd.DataFrame(np.vstack(parts), columns=FAKE_FEATURES)
    df[LABEL_COL] = labels
    base = pd.Timestamp("2018-02-14 08:00:00")
    secs = rng.integers(0, 36000, len(df))
    df[TS_COL] = [(base + pd.Timedelta(seconds=int(s))).strftime("%d/%m/%Y %H:%M:%S") for s in secs]
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.insert(0, ID_COL, np.arange(len(df), dtype=np.int64))
    return df[[ID_COL, TS_COL] + FAKE_FEATURES + [LABEL_COL]]


def write_fake_dataset(out_dir, n: int = 4000, seed: int = SEED, scaler_path=None):
    """Ghi <out>/train|val|test/part-00000.parquet (70/10/20) và trả về scaler fit trên train Benign."""
    df = make_fake_flows(n, seed=seed)
    r = np.random.default_rng(seed).random(len(df))
    splits = {"train": df[r < 0.7], "val": df[(r >= 0.7) & (r < 0.8)], "test": df[r >= 0.8]}
    for name, part in splits.items():
        p = Path(out_dir) / name
        p.mkdir(parents=True, exist_ok=True)
        part.reset_index(drop=True).to_parquet(p / "part-00000.parquet", index=False)
    scaler = fit_scaler(splits["train"], FAKE_FEATURES)
    if scaler_path is not None:
        scaler.to_json(scaler_path)
    return scaler


def write_fake_raw_csvs(out_dir, n_per_file: int = 1500, seed: int = SEED) -> list:
    """CSV thô 'bẩn' giống CIC: header lặp, Infinity/NaN, cột hằng, dòng trùng, file 20-02 có cột ID.

    Mỗi file sau khi làm sạch còn đúng n_per_file - 2 dòng (2 dòng bẩn bị bỏ, 10 dòng trùng bị gộp).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = []
    for i, day in enumerate(["14-02-2018", "16-02-2018", "20-02-2018"]):
        df = make_fake_flows(n_per_file, seed=seed + i).drop(columns=[ID_COL])
        df = df.rename(columns={TS_COL: "Timestamp"})
        df.insert(0, "Dst Port", rng.integers(1, 65535, len(df)))
        df.insert(2, "Timestamp", df.pop("Timestamp"))
        df["Const Col"] = 0.0
        df["Label"] = df.pop(LABEL_COL)
        df["Flow Byts/s"] = df["Flow Byts/s"].astype(object)
        df["Flow Pkts/s"] = df["Flow Pkts/s"].astype(object)
        df.loc[50, "Flow Byts/s"] = "Infinity"
        df.loc[51, "Flow Pkts/s"] = "NaN"
        if day == "20-02-2018":
            df.insert(0, "Flow ID", ["f%d" % k for k in range(len(df))])
            df.insert(1, "Src IP", "172.31.0.%d" % (i + 1))
            df.insert(2, "Src Port", rng.integers(1024, 65535, len(df)))
            df.insert(3, "Dst IP", "172.31.69.25")
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)                 # dòng trùng
        header = pd.DataFrame([list(df.columns)], columns=df.columns)
        df = pd.concat([df.iloc[:100], header, df.iloc[100:]], ignore_index=True)  # header lặp
        name = "Fake-%s_TrafficForML_CICFlowMeter.csv" % day
        df.to_csv(out / name, index=False)
        names.append(name)
    return names


def main(argv=None):
    p = argparse.ArgumentParser(description="Sinh dataset giả (Parquet train/val/test + scaler.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--scaler", default=None, help="mặc định <out>/scaler.json")
    p.add_argument("--n", type=int, default=4000)
    a = p.parse_args(argv)
    write_fake_dataset(a.out, n=a.n, scaler_path=a.scaler or Path(a.out) / "scaler.json")
    print("wrote fake dataset to", a.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Chạy test, xác nhận pass (Py3.12 và Py3.8)**

Run: `.venv\Scripts\python -m pytest tests/test_data.py tests/test_fake.py -q`
Expected: `13 passed`.
Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_data.py tests/test_fake.py -q`
Expected: `13 passed`.

- [ ] **Step 6: Commit**

```bash
git add common/data.py common/fake.py tests/test_data.py tests/test_fake.py
git commit -m "feat(common): shared scaler/load_split and fake data generators" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `common/metrics.py`

**Files:**
- Create: `common/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `common.data.BENIGN`.
- Produces: `METRIC_COLUMNS`, `confusion_metrics`, `evaluate`, `per_attack_recall`, `save_scores` (mục Interfaces). `confusion_metrics` chỉ dùng numpy (chạy trên cụm không có sklearn); `evaluate` import sklearn bên trong hàm.

- [ ] **Step 1: Viết test fail**

`tests/test_metrics.py`:
```python
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
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `.venv\Scripts\python -m pytest tests/test_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'common.metrics'`.

- [ ] **Step 3: Viết `common/metrics.py`**

```python
"""Đánh giá dùng CHUNG cho baseline, DL và streaming."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.data import BENIGN

METRIC_COLUMNS = ["model", "group", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc",
                  "threshold", "train_time_s", "infer_ms_per_1k"]


def confusion_metrics(y, y_pred) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(y_pred).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def evaluate(y, score, thr) -> dict:
    """Dự đoán tấn công khi score >= thr. pr_auc = average precision."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    m = confusion_metrics(y, score >= thr)
    out = {k: float(m[k]) for k in ("precision", "recall", "f1", "fpr")}
    both = len(np.unique(y)) == 2
    out["roc_auc"] = float(roc_auc_score(y, score)) if both else float("nan")
    out["pr_auc"] = float(average_precision_score(y, score)) if both else float("nan")
    out["threshold"] = float(thr)
    return out


def per_attack_recall(model: str, labels, y_pred) -> pd.DataFrame:
    labels = np.asarray(labels).astype(str)
    y_pred = np.asarray(y_pred).astype(bool)
    rows = []
    for att in sorted(set(labels) - {BENIGN}):
        mask = labels == att
        rows.append({"model": model, "attack": att,
                     "recall": float(y_pred[mask].mean()), "n": int(mask.sum())})
    return pd.DataFrame(rows, columns=["model", "attack", "recall", "n"])


def save_scores(path, y, score, thr) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, y=np.asarray(y), score=np.asarray(score), thr=np.asarray(thr))
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv\Scripts\python -m pytest tests/test_metrics.py -q`
Expected: `7 passed`.
Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_metrics.py -q`
Expected: `5 passed, 2 skipped` (không có sklearn trên Py3.8 cụm — đúng ý đồ).

- [ ] **Step 5: Commit**

```bash
git add common/metrics.py tests/test_metrics.py
git commit -m "feat(common): shared evaluate/confusion/per-attack metrics" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `scripts/check_cluster.sh` (kết thúc M0)

**Files:**
- Create: `scripts/check_cluster.sh`
- Test: `tests/test_check_cluster.py`
- Modify: `README.md` (thêm mục "Kiểm tra cụm")

**Interfaces:**
- Consumes: `scripts/cfg.py`, config keys `cluster.expected_datanodes`, `cluster.spark_master_ui`, `cluster.expected_workers`, `cluster.kafka_home`, `cluster.kafka_bootstrap`.
- Produces: `bash scripts/check_cluster.sh [config.yaml]` → in `[PASS]`/`[FAIL]` 3 dòng, exit 0 khi tất cả PASS.

- [ ] **Step 1: Viết test fail (dùng stub cho hdfs/curl/kafka)**

`tests/test_check_cluster.py`:
```python
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash script: chạy trong docker")


def _stub(path: Path, body: str):
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    path.chmod(0o755)


def _setup(tmp_path, live_dn):
    bin_dir = tmp_path / "bin"
    kh = tmp_path / "kafka" / "bin"
    bin_dir.mkdir()
    kh.mkdir(parents=True)
    _stub(bin_dir / "hdfs", 'echo "Live datanodes (%d):"\n' % live_dn)
    _stub(bin_dir / "curl", 'echo \'{"workers":[{"state":"ALIVE"},{"state":"ALIVE"},{"state":"DEAD"}]}\'\n')
    box = tmp_path / "msg"
    _stub(kh / "kafka-topics.sh", "exit 0\n")
    _stub(kh / "kafka-console-producer.sh", 'cat > "%s"\n' % box)
    _stub(kh / "kafka-console-consumer.sh", 'cat "%s"\n' % box)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "cluster:\n  expected_datanodes: 2\n  expected_workers: 2\n"
        "  spark_master_ui: http://x:8080\n  kafka_home: %s\n  kafka_bootstrap: x:9092\n"
        % (tmp_path / "kafka"), encoding="utf-8")
    env = dict(os.environ, PATH="%s:%s" % (bin_dir, os.environ["PATH"]))
    return cfg, env


def _run(cfg, env):
    return subprocess.run(["bash", str(ROOT / "scripts" / "check_cluster.sh"), str(cfg)],
                          env=env, capture_output=True, text=True, timeout=60)


def test_all_pass(tmp_path):
    cfg, env = _setup(tmp_path, live_dn=2)
    r = _run(cfg, env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("[PASS]") == 3


def test_hdfs_fail(tmp_path):
    cfg, env = _setup(tmp_path, live_dn=1)
    r = _run(cfg, env)
    assert r.returncode == 1
    assert "[FAIL] HDFS" in r.stdout and r.stdout.count("[PASS]") == 2
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_check_cluster.py -q`
Expected: FAIL (script chưa tồn tại, `returncode == 127`).

- [ ] **Step 3: Viết `scripts/check_cluster.sh`**

```bash
#!/usr/bin/env bash
# Kiểm tra HDFS / Spark / Kafka. Dùng: bash scripts/check_cluster.sh [config.yaml]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export BTL_CONFIG="${1:-${BTL_CONFIG:-$ROOT/config.yaml}}"
cfg() { python3 "$ROOT/scripts/cfg.py" "$1"; }
FAIL=0
pass() { echo "[PASS] $1"; }
fail() { echo "[FAIL] $1"; FAIL=1; }

# 1. HDFS
EXPECTED_DN=$(cfg cluster.expected_datanodes)
LIVE=$(hdfs dfsadmin -report 2>/dev/null | sed -n 's/^Live datanodes (\([0-9]*\)).*/\1/p' | head -1)
if [ -n "$LIVE" ] && [ "$LIVE" -ge "$EXPECTED_DN" ]; then
  pass "HDFS: $LIVE datanode sống"
else
  fail "HDFS: live datanodes='${LIVE:-?}', cần >= $EXPECTED_DN"
fi

# 2. Spark master UI
UI=$(cfg cluster.spark_master_ui)
EXPECTED_W=$(cfg cluster.expected_workers)
JSON=$(curl -s --max-time 5 "$UI/json/" || true)
if [ -n "$JSON" ]; then
  W=$(printf '%s' "$JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(sum(1 for w in d.get("workers", []) if w.get("state") == "ALIVE"))' 2>/dev/null || echo 0)
  if [ "$W" -ge "$EXPECTED_W" ]; then pass "Spark: master UI OK, $W worker ALIVE"; else fail "Spark: $W worker ALIVE, cần >= $EXPECTED_W"; fi
else
  fail "Spark: master UI $UI không phản hồi"
fi

# 3. Kafka: tạo topic, gửi, nhận
KH=$(cfg cluster.kafka_home)
BS=$(cfg cluster.kafka_bootstrap)
T="btl-healthcheck"
MSG="ping-$(date +%s)-$$"
"$KH/bin/kafka-topics.sh" --bootstrap-server "$BS" --create --if-not-exists --topic "$T" \
  --partitions 1 --replication-factor 1 >/dev/null 2>&1
echo "$MSG" | "$KH/bin/kafka-console-producer.sh" --bootstrap-server "$BS" --topic "$T" >/dev/null 2>&1
GOT=$("$KH/bin/kafka-console-consumer.sh" --bootstrap-server "$BS" --topic "$T" --from-beginning \
  --timeout-ms 10000 2>/dev/null | grep -c "$MSG")
if [ "${GOT:-0}" -ge 1 ]; then pass "Kafka: gửi/nhận message OK ($BS)"; else fail "Kafka: không nhận lại được message trên $BS"; fi

exit $FAIL
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_check_cluster.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Cập nhật README, chạy toàn bộ test M0, commit**

Thêm vào `README.md`:
````markdown
## Kiểm tra cụm (M0)
```bash
bash scripts/check_cluster.sh            # đọc config.yaml; in [PASS]/[FAIL] cho HDFS, Spark, Kafka
```
````

Run: `.venv\Scripts\python -m pytest -q` → toàn bộ pass (test bash bị skip trên Windows).
Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest -q` → pass.

```bash
git add scripts/check_cluster.sh tests/test_check_cluster.py README.md
git commit -m "feat(scripts): cluster health check (M0)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Báo người dùng: làm Phụ lục A1–A4 song song trên cụm.

---

### Task 5: `preprocess/spark_preprocess.py` (M1)

**Files:**
- Create: `preprocess/__init__.py` (rỗng), `preprocess/spark_preprocess.py`, `scripts/run_preprocess.sh`
- Test: `tests/test_preprocess.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `common.config`, `common.data` (`LABEL_COL`, `BENIGN`, `TS_COL`, `ID_COL`, `SEED`, `Scaler`), `common.fake.write_fake_raw_csvs` (test).
- Produces: Parquet `<out>/{train,val,test}` với cột theo thứ tự `[flow_id, orig_ts, <features...>, Label]`; `models/scaler.json`; `results/preprocess_stats.json`; bản mẫu `<export-sample>/{train,val,test}` (mỗi tập 1 file Parquet). `main(argv=None) -> dict` (stats).

- [ ] **Step 1: Viết test fail**

`tests/test_preprocess.py`:
```python
import json

import numpy as np
import pandas as pd
import pytest

from common.data import BENIGN, ID_COL, LABEL_COL, TS_COL, load_scaler
from common.fake import FAKE_FEATURES, write_fake_raw_csvs

pytest.importorskip("pyspark")
pytestmark = pytest.mark.spark
N_PER_FILE = 1500


@pytest.fixture(scope="module")
def run_pre(tmp_path_factory):
    from preprocess.spark_preprocess import main
    d = tmp_path_factory.mktemp("pre")
    files = write_fake_raw_csvs(d / "raw", n_per_file=N_PER_FILE)
    stats = main(["--master", "local[2]", "--input-dir", str(d / "raw"), "--files", *files,
                  "--out", str(d / "out"), "--export-sample", str(d / "sample"),
                  "--sample-train", "1000", "--sample-val", "200", "--sample-test", "300",
                  "--scaler-out", str(d / "scaler.json"), "--stats-out", str(d / "stats.json")])
    return d, stats


def _read(d, split):
    return pd.read_parquet(d / "out" / split)


def test_row_counts_and_cleaning(run_pre):
    d, stats = run_pre
    total = sum(len(_read(d, s)) for s in ("train", "val", "test"))
    assert total == 3 * (N_PER_FILE - 2)
    assert stats["rows"]["after_dedup"] == total


def test_columns(run_pre):
    d, _ = run_pre
    cols = list(_read(d, "train").columns)
    assert cols == [ID_COL, TS_COL] + FAKE_FEATURES + [LABEL_COL]


def test_splits_disjoint_and_proportions(run_pre):
    d, _ = run_pre
    ids = {s: set(_read(d, s)[ID_COL]) for s in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    total = sum(len(v) for v in ids.values())
    assert abs(len(ids["train"]) / total - 0.7) < 0.05
    assert abs(len(ids["test"]) / total - 0.2) < 0.05
    test_labels = set(_read(d, "test")[LABEL_COL])
    assert BENIGN in test_labels and len(test_labels) >= 4


def test_scaler_matches_numpy(run_pre):
    d, _ = run_pre
    s = load_scaler(d / "scaler.json")
    assert s.features == FAKE_FEATURES
    b = _read(d, "train")
    b = b[b[LABEL_COL] == BENIGN][FAKE_FEATURES].to_numpy(np.float64)
    np.testing.assert_allclose(s.mean, b.mean(axis=0), rtol=1e-9)
    np.testing.assert_allclose(s.std, b.std(axis=0), rtol=1e-6)


def test_sample_and_stats(run_pre):
    d, _ = run_pre
    tr = pd.read_parquet(d / "sample" / "train")
    assert 0 < len(tr) <= 1000
    stats = json.loads((d / "stats.json").read_text(encoding="utf-8"))
    assert stats["constant_cols_dropped"] == ["Const Col"]
    assert set(stats["label_distribution"]) == {"train", "val", "test"}
    assert "read_clean" in stats["timings_s"]


def test_cleaning_steps_for_eda(run_pre):
    _, stats = run_pre
    steps = {s["step"]: s for s in stats["cleaning"]}
    assert list(steps) == ["read_csv", "drop_repeated_headers", "drop_nan_inf", "drop_constant_cols", "drop_duplicates"]
    assert steps["drop_repeated_headers"]["rows_removed"] == 3          # 1 header lặp mỗi file
    assert steps["drop_nan_inf"]["rows_removed"] == 3 * 2               # 2 dòng bẩn mỗi file
    assert steps["drop_duplicates"]["rows_removed"] == 3 * 10           # 10 dòng trùng mỗi file
    assert steps["drop_constant_cols"]["cols"] == steps["drop_nan_inf"]["cols"] - 1
```

Ghi chú test trùng lặp: 10 dòng trùng là bản sao của dòng 0–9, không trùng với dòng bẩn 50–51, nên `drop_nan_inf` không làm mất chúng.

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_preprocess.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'preprocess.spark_preprocess'`.

- [ ] **Step 3: Viết `preprocess/spark_preprocess.py`**

```python
"""CSE-CIC-IDS2018 CSV -> Parquet train/val/test + scaler.json (+ bản mẫu nhỏ cho Kaggle).

Chính sách giá trị lỗi: dòng có null/NaN/±Infinity ở BẤT KỲ cột feature nào bị bỏ.
Chia tập: mỗi dòng nhận r = rand(seed=42); train r<0.7, val 0.7<=r<0.8, test r>=0.8.
Vì r độc lập với Label, đây là random split trong từng Label (stratified theo kỳ vọng).

Cụm:   bash scripts/run_preprocess.sh
Local: python preprocess/spark_preprocess.py --master "local[*]" --input-dir data/raw --out data/processed
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from functools import reduce
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as C  # noqa: E402
from common.data import BENIGN, ID_COL, LABEL_COL, SEED, TS_COL, Scaler  # noqa: E402

DROP_COLS = ["Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port"]
RAW_TS = "Timestamp"
SPLITS = (("train", 0.0, 0.7), ("val", 0.7, 0.8), ("test", 0.8, 1.0))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--master", default=None, help="spark://host:7077 hoặc local[*]; bỏ trống khi spark-submit --master")
    p.add_argument("--input-dir", default=None, help="thư mục CSV (mặc định hdfs.raw_dir)")
    p.add_argument("--files", nargs="+", default=None, help="tên file CSV (mặc định data.raw_files)")
    p.add_argument("--out", default=None, help="thư mục Parquet ra (mặc định hdfs.processed_dir)")
    p.add_argument("--export-sample", default=None, help="thư mục bản mẫu cho Kaggle (cùng FS với --out)")
    p.add_argument("--sample-train", type=int, default=None)
    p.add_argument("--sample-val", type=int, default=None)
    p.add_argument("--sample-test", type=int, default=None)
    p.add_argument("--scaler-out", default=None, help="mặc định models.scaler")
    p.add_argument("--stats-out", default="results/preprocess_stats.json")
    p.add_argument("--executor-memory", default=None)
    p.add_argument("--driver-memory", default=None)
    p.add_argument("--shuffle-partitions", type=int, default=None)
    a = p.parse_args(argv)
    cfg = C.load_config(a.config)
    a.input_dir = a.input_dir or C.get(cfg, "hdfs.raw_dir")
    a.files = a.files or C.get(cfg, "data.raw_files")
    a.out = a.out or C.get(cfg, "hdfs.processed_dir")
    sizes = C.get(cfg, "data.sample_sizes", {})
    a.sample_train = a.sample_train or sizes.get("train", 500000)
    a.sample_val = a.sample_val or sizes.get("val", 200000)
    a.sample_test = a.sample_test or sizes.get("test", 300000)
    a.scaler_out = a.scaler_out or C.get(cfg, "models.scaler", "models/scaler.json")
    a.executor_memory = a.executor_memory or C.get(cfg, "spark.executor_memory")
    a.driver_memory = a.driver_memory or C.get(cfg, "spark.driver_memory")
    a.shuffle_partitions = a.shuffle_partitions or C.get(cfg, "spark.shuffle_partitions", 12)
    return a


@contextlib.contextmanager
def timed(stats: dict, name: str):
    t0 = time.time()
    yield
    stats["timings_s"][name] = round(time.time() - t0, 2)
    print("[time] %-14s %8.2fs" % (name, stats["timings_s"][name]), flush=True)


def build_spark(a):
    from pyspark.sql import SparkSession
    b = SparkSession.builder.appName("btl-preprocess") \
        .config("spark.sql.shuffle.partitions", str(a.shuffle_partitions))
    if a.master:
        b = b.master(a.master)
    if a.executor_memory:
        b = b.config("spark.executor.memory", a.executor_memory)
    if a.driver_memory:   # chỉ có tác dụng khi chạy bằng python (local); spark-submit dùng --driver-memory
        b = b.config("spark.driver.memory", a.driver_memory)
    return b.getOrCreate()


def read_raw(spark, paths):
    from pyspark.sql import functions as F
    frames = []
    for p in paths:
        df = spark.read.option("header", True).csv(p)        # mọi cột là string
        df = df.toDF(*[c.strip() for c in df.columns])
        frames.append(df.drop(*[c for c in DROP_COLS if c in df.columns]))
    common = set.intersection(*[set(f.columns) for f in frames])
    cols = [c for c in frames[0].columns if c in common]
    for p, f in zip(paths, frames):
        extra = sorted(set(f.columns) - common)
        if extra:
            print("[warn] %s: bỏ cột không có ở mọi file: %s" % (p, extra))
    return reduce(lambda x, y: x.unionByName(y), [f.select(*cols) for f in frames])


def drop_repeated_headers(df):
    from pyspark.sql import functions as F
    return df.filter(F.col(LABEL_COL) != LABEL_COL)


def cast_and_clean(df):
    from pyspark.sql import functions as F
    feats = [c for c in df.columns if c not in (LABEL_COL, RAW_TS)]
    df = df.select(*[F.col(c).cast("double").alias(c) for c in feats],
                   F.trim(F.col(LABEL_COL)).alias(LABEL_COL), F.col(RAW_TS).alias(TS_COL))
    bad = None
    for c in feats:
        cond = F.col(c).isNull() | F.isnan(F.col(c)) | (F.abs(F.col(c)) == float("inf"))
        bad = cond if bad is None else (bad | cond)
    return df.filter(~bad), feats


def drop_constant(df, feats):
    from pyspark.sql import functions as F
    aggs = []
    for i, c in enumerate(feats):
        aggs += [F.min(F.col(c)).alias("mn%d" % i), F.max(F.col(c)).alias("mx%d" % i)]
    row = df.agg(*aggs).collect()[0]
    const = [c for i, c in enumerate(feats) if row["mn%d" % i] == row["mx%d" % i]]
    keep = [c for c in feats if c not in const]
    return df.drop(*const), keep, const


def compute_scaler(train_df, feats) -> Scaler:
    from pyspark.sql import functions as F
    b = train_df.filter(F.col(LABEL_COL) == BENIGN)
    aggs = [F.avg(F.col(c)).alias("m%d" % i) for i, c in enumerate(feats)]
    aggs += [F.stddev_pop(F.col(c)).alias("s%d" % i) for i, c in enumerate(feats)]
    row = b.agg(*aggs).collect()[0]
    if row["m0"] is None:
        raise ValueError("train không có dòng Benign")
    mean = np.array([row["m%d" % i] for i in range(len(feats))], dtype=np.float64)
    std = np.array([row["s%d" % i] or 0.0 for i in range(len(feats))], dtype=np.float64)
    std[std == 0] = 1.0
    return Scaler(list(feats), mean, std)


def main(argv=None) -> dict:
    a = parse_args(argv)
    from pyspark import StorageLevel
    from pyspark.sql import functions as F

    stats = {"timings_s": {}, "rows": {}, "files": list(a.files)}
    t_all = time.time()
    spark = build_spark(a)
    paths = ["%s/%s" % (a.input_dir.rstrip("/"), f) for f in a.files]

    with timed(stats, "read_clean"):
        lines = read_raw(spark, paths)
        stats["rows"]["raw_lines"] = lines.count()          # gồm cả các dòng header lặp
        raw = drop_repeated_headers(lines)
        stats["rows"]["raw"] = raw.count()
        n_cols_raw = len(raw.columns)
        clean, feats = cast_and_clean(raw)
        clean = clean.persist(StorageLevel.MEMORY_AND_DISK)
        stats["rows"]["after_clean"] = clean.count()
    with timed(stats, "constant_cols"):
        clean, feats, const = drop_constant(clean, feats)
        stats["constant_cols_dropped"] = const
        stats["n_features"] = len(feats)
    with timed(stats, "dedup"):
        dedup = clean.dropDuplicates(feats + [LABEL_COL]) \
            .withColumn(ID_COL, F.monotonically_increasing_id()) \
            .withColumn("_r", F.rand(SEED)) \
            .persist(StorageLevel.MEMORY_AND_DISK)
        stats["rows"]["after_dedup"] = dedup.count()
        clean.unpersist()
    r = stats["rows"]
    # Bảng "hiệu quả làm sạch" cho analysis/eda.py (PROJECT_SPEC 5.10 mục 2)
    stats["cleaning"] = [
        {"step": "read_csv", "rows": r["raw_lines"], "rows_removed": 0, "cols": n_cols_raw},
        {"step": "drop_repeated_headers", "rows": r["raw"], "rows_removed": r["raw_lines"] - r["raw"], "cols": n_cols_raw},
        {"step": "drop_nan_inf", "rows": r["after_clean"], "rows_removed": r["raw"] - r["after_clean"], "cols": n_cols_raw},
        {"step": "drop_constant_cols", "rows": r["after_clean"], "rows_removed": 0, "cols": n_cols_raw - len(const)},
        {"step": "drop_duplicates", "rows": r["after_dedup"], "rows_removed": r["after_clean"] - r["after_dedup"],
         "cols": n_cols_raw - len(const)},
    ]

    cols = [ID_COL, TS_COL] + feats + [LABEL_COL]
    split_dfs = {}
    for name, lo, hi in SPLITS:
        with timed(stats, "write_" + name):
            part = dedup.filter((F.col("_r") >= lo) & (F.col("_r") < hi)).select(*[F.col(c) for c in cols])
            part.write.mode("overwrite").parquet("%s/%s" % (a.out, name))
        split_dfs[name] = spark.read.parquet("%s/%s" % (a.out, name))
    dedup.unpersist()

    with timed(stats, "label_dist"):
        stats["label_distribution"] = {}
        for name, df in split_dfs.items():
            rows = df.groupBy(LABEL_COL).count().collect()
            stats["label_distribution"][name] = {r[LABEL_COL]: int(r["count"]) for r in rows}
            stats["rows"][name] = int(sum(r["count"] for r in rows))
    with timed(stats, "scaler"):
        compute_scaler(split_dfs["train"], feats).to_json(a.scaler_out)
    if a.export_sample:
        with timed(stats, "export_sample"):
            for name, n in (("train", a.sample_train), ("val", a.sample_val), ("test", a.sample_test)):
                frac = min(1.0, 1.1 * n / max(stats["rows"][name], 1))
                split_dfs[name].sample(False, frac, seed=SEED).limit(n).coalesce(1) \
                    .write.mode("overwrite").parquet("%s/%s" % (a.export_sample, name))

    stats["timings_s"]["total"] = round(time.time() - t_all, 2)
    Path(a.stats_out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.stats_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=1, ensure_ascii=False)
    print(json.dumps({"rows": stats["rows"], "label_distribution": stats["label_distribution"],
                      "timings_s": stats["timings_s"]}, indent=1, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_preprocess.py -q`
Expected: `6 passed` (~1–2 phút).

- [ ] **Step 5: `scripts/run_preprocess.sh`**

```bash
#!/usr/bin/env bash
# Chạy tiền xử lý trên cụm rồi kéo bản mẫu Kaggle + scaler về master.
# Dùng: bash scripts/run_preprocess.sh [tham số thêm cho spark_preprocess.py]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export BTL_CONFIG="${BTL_CONFIG:-$ROOT/config.yaml}"
cfg() { python3 scripts/cfg.py "$1"; }
SAMPLE_HDFS=$(cfg hdfs.sample_dir)
SAMPLE_LOCAL=$(cfg data.sample_local_dir)
mkdir -p results

spark-submit --master "$(cfg cluster.spark_master)" \
  --driver-memory "$(cfg spark.driver_memory)" --executor-memory "$(cfg spark.executor_memory)" \
  --conf spark.pyspark.python=python3 \
  preprocess/spark_preprocess.py --export-sample "$SAMPLE_HDFS" "$@" 2>&1 | tee results/preprocess.log

rm -rf "$SAMPLE_LOCAL" && mkdir -p "$SAMPLE_LOCAL"
for s in train val test; do hdfs dfs -get "$SAMPLE_HDFS/$s" "$SAMPLE_LOCAL/$s"; done
cp "$(cfg models.scaler)" "$SAMPLE_LOCAL/scaler.json"
cp results/preprocess_stats.json "$SAMPLE_LOCAL/preprocess_stats.json"   # cho analysis/eda.py trên Kaggle
echo "Bản mẫu Kaggle: $SAMPLE_LOCAL  (upload thư mục này thành Kaggle Dataset)"
```

- [ ] **Step 6: README + commit**

Thêm vào `README.md`:
````markdown
## Tiền xử lý (M1)
Chính sách giá trị lỗi: dòng có null/NaN/±Infinity ở bất kỳ feature nào bị **bỏ**. Bỏ cột `Flow ID, Src IP, Dst IP, Src Port, Dst Port`; `Timestamp` giữ thành metadata `orig_ts`; bỏ cột hằng; `dropDuplicates(features + Label)` trước khi chia 70/10/20 (rand seed 42 theo từng dòng = random split trong từng Label).
```bash
bash scripts/run_preprocess.sh          # cụm -> HDFS processed_dir + data/kaggle_sample/ + models/scaler.json
python3 preprocess/spark_preprocess.py --master "local[*]" --input-dir data/raw --out data/processed   # fallback local/Kaggle
```
Thống kê (số dòng, phân phối Label, thời gian): `results/preprocess_stats.json`.
````

```bash
git add preprocess/ scripts/run_preprocess.sh tests/test_preprocess.py README.md
git commit -m "feat(preprocess): Spark CSV->Parquet split, scaler, Kaggle sample (M1)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Báo người dùng chạy Phụ lục A5.

---

### Task 5b: `analysis/eda.py` phần 1 — trực quan hoá dữ liệu (M1b)

**REQUIRED:** load skill `dataviz` trước khi viết code vẽ.

**Files:**
- Create: `analysis/__init__.py` (rỗng), `analysis/eda.py`
- Modify: `requirements-kaggle.txt` (thêm dòng `scipy>=1.10`), `README.md`
- Test: `tests/test_eda.py`

**Interfaces:**
- Consumes: `common.data` (`BENIGN`, `CLIP`, `LABEL_COL`, `SEED`, `binarize`, `frame_to_matrix`, `load_scaler`, `read_split_frame`), `common.config`, `common.fake.write_fake_dataset`; `results/preprocess_stats.json` với khoá `label_distribution` (split → {label: count}) và `cleaning` (list `{step, rows, rows_removed, cols}`) do Task 5 ghi.
- Produces: CLI `python analysis/eda.py [--data-dir --scaler --stats --out reports/eda --per-class 2000 --top-k 12 --max-rows 200000 --perplexity 30 --skip-tsne --latent --no-part1 --models-dir models --smoke]`; hàm `stratified_sample(df, per_class, seed=SEED) -> DataFrame`, `main(argv=None) -> dict` (thời gian từng bước). File trong `--out`: `label_distribution.png`, `label_counts.csv`, `cleaning_steps.csv`, `ks_table.csv`, `top_features_box.png`, `correlation_heatmap.png`, `high_corr_pairs.csv`, `scaling_before_after.png`, `clip_rate.csv`, `clip_rate.png`, `pca_explained_variance.png`, `projection_pca.png`, `projection_tsne.png`, `projection_umap.png` (chỉ khi có `umap-learn`), `eda_summary.md`. Phần 2 (`--latent`, Task 11b) thêm `latent_pca.png`, `latent_tsne.png`, `silhouette.csv`.

Quyết định thiết kế:
- Dữ liệu cho KS/tương quan/chuẩn hoá: tập train của bản mẫu, tối đa `--max-rows` dòng (lấy qua `read_split_frame`, seed 42). Mẫu phân tầng cho phép chiếu 2D: tối đa `--per-class` dòng mỗi nhãn từ cùng tập đó.
- Thang log cho feature có thể âm (vd `Init Fwd Win Byts = -1`): dùng `sign(x)·log1p(|x|)`, ghi rõ trên trục.
- Mục 3 dùng **boxplot** (spec cho phép violin/box): violin lỗi KDE khi một nhóm có giá trị hằng.
- Silhouette (Task 11b) tính với `sample_size=5000` để tránh ma trận khoảng cách O(n²) quá lớn.
- `eda_summary.md` được dựng lại từ các file đang có trong `--out` + bảng mô tả cố định `DESCRIPTIONS`, nên chạy phần 2 riêng không làm mất mô tả phần 1. Các số trong summary (tỷ lệ Benign/Attack, bảng làm sạch, số cặp tương quan cao, silhouette) đều do code tính từ file thật.
- Thiếu `preprocess_stats.json` hoặc thiếu khoá `cleaning` → in `[skip]` cho mục 1–2, các mục khác vẫn chạy.

- [ ] **Step 1: Viết test fail**

`tests/test_eda.py`:
```python
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
pytest.importorskip("matplotlib")
from analysis import eda  # noqa: E402
from common.data import LABEL_COL  # noqa: E402
from common.fake import FAKE_ATTACKS, make_fake_flows  # noqa: E402

PART1 = ["label_distribution.png", "label_counts.csv", "cleaning_steps.csv", "ks_table.csv", "top_features_box.png",
         "correlation_heatmap.png", "high_corr_pairs.csv", "scaling_before_after.png", "clip_rate.csv", "clip_rate.png",
         "pca_explained_variance.png", "projection_pca.png", "projection_tsne.png", "eda_summary.md"]


def test_stratified_sample():
    df = make_fake_flows(2000)
    s = eda.stratified_sample(df, per_class=100)
    counts = s[LABEL_COL].value_counts()
    assert set(counts.index) == {"Benign", *FAKE_ATTACKS} and (counts == 100).all()
    pd.testing.assert_frame_equal(s, eda.stratified_sample(df, per_class=100))   # seed cố định


def test_part1_smoke(tmp_path):
    out = tmp_path / "eda"
    eda.main(["--smoke", "--out", str(out), "--per-class", "150"])
    missing = [f for f in PART1 if not (out / f).exists()]
    assert missing == []
    ks = pd.read_csv(out / "ks_table.csv")
    assert list(ks.columns) == ["feature", "ks", "pvalue"] and ks["ks"].is_monotonic_decreasing
    summary = (out / "eda_summary.md").read_text(encoding="utf-8")
    for f in PART1:
        if f.endswith(".png"):
            assert "`%s`" % f in summary
    assert not (out / "latent_pca.png").exists()


def test_missing_stats_is_skipped(tmp_path):
    from common.fake import write_fake_dataset
    write_fake_dataset(tmp_path / "data", n=3000, scaler_path=tmp_path / "scaler.json")
    out = tmp_path / "eda"
    eda.main(["--data-dir", str(tmp_path / "data"), "--scaler", str(tmp_path / "scaler.json"),
              "--stats", str(tmp_path / "none.json"), "--out", str(out), "--per-class", "100", "--skip-tsne"])
    assert not (out / "label_distribution.png").exists() and (out / "projection_pca.png").exists()
    assert not (out / "projection_tsne.png").exists()
```

Run: `.venv\Scripts\python -m pytest tests/test_eda.py -q` → FAIL (`No module named 'analysis.eda'`).

- [ ] **Step 2: Viết `analysis/eda.py`**

```python
"""Trực quan hoá dữ liệu CSE-CIC-IDS2018 (PROJECT_SPEC 5.10) -> <out>/*.png + eda_summary.md.

Phần 1 (sau preprocessing):
  python analysis/eda.py --data-dir data/kaggle_sample --stats data/kaggle_sample/preprocess_stats.json
Phần 2 (sau train DL, cần models/ae.pt, models/vae.pt):
  python analysis/eda.py --latent --no-part1 --models-dir models
Smoke (dữ liệu giả):  python analysis/eda.py --smoke [--latent]
Feature có thể âm nên "thang log" là sign(x)*log1p(|x|).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as C  # noqa: E402
from common.data import (BENIGN, CLIP, LABEL_COL, SEED, binarize, frame_to_matrix,  # noqa: E402
                         load_scaler, read_split_frame)

DPI = 150
CORR_THRESHOLD = 0.95
BENIGN_COLOR = "#9ca3af"
DESCRIPTIONS = {
    "label_distribution.png": "Số dòng theo từng loại nhãn (trục log), tách train/val/test, đếm trên toàn bộ dữ liệu sau tiền xử lý.",
    "top_features_box.png": "Phân phối (thang sign·log1p) của các feature có KS Benign-vs-Attack lớn nhất, theo từng loại nhãn.",
    "correlation_heatmap.png": "Tương quan Pearson giữa các feature trên Benign train, sắp theo hierarchical clustering.",
    "scaling_before_after.png": "Histogram của 4 feature phân biệt nhất trước (sign·log1p) và sau chuẩn hoá + clip ±10.",
    "clip_rate.png": "Tỷ lệ giá trị bị clip ở ±10 sau chuẩn hoá, 20 feature cao nhất.",
    "pca_explained_variance.png": "Explained variance tích luỹ của PCA trên dữ liệu đã chuẩn hoá.",
    "projection_pca.png": "Chiếu PCA 2D của mẫu phân tầng, tô màu theo loại nhãn.",
    "projection_tsne.png": "Chiếu t-SNE 2D của cùng mẫu, tô màu theo loại nhãn.",
    "projection_umap.png": "Chiếu UMAP 2D của cùng mẫu, tô màu theo loại nhãn.",
    "latent_pca.png": "PCA 2D của dữ liệu gốc, latent AE và latent VAE (μ) trên cùng mẫu.",
    "latent_tsne.png": "t-SNE 2D của dữ liệu gốc, latent AE và latent VAE (μ) trên cùng mẫu.",
}


def log(msg):
    print("[eda] " + msg, flush=True)


def symlog1p(x):
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    log("wrote " + Path(path).name)


def stratified_sample(df: pd.DataFrame, per_class: int, seed: int = SEED) -> pd.DataFrame:
    parts = [g.sample(n=min(len(g), per_class), random_state=seed) for _, g in df.groupby(LABEL_COL, sort=True)]
    return pd.concat(parts).reset_index(drop=True)


def label_colors(labels):
    attacks = sorted(set(labels) - {BENIGN})
    cmap = plt.get_cmap("tab10")
    colors = {a: cmap(i % 10) for i, a in enumerate(attacks)}
    colors[BENIGN] = BENIGN_COLOR
    return colors


def scatter_by_label(ax, P, labels, title):
    colors = label_colors(labels)
    for lab in [BENIGN] + sorted(set(labels) - {BENIGN}):
        m = labels == lab
        if m.any():
            ax.scatter(P[m, 0], P[m, 1], s=4, alpha=0.6, color=colors[lab], label=lab, rasterized=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def tsne_2d(X, perplexity):
    from sklearn.manifold import TSNE
    perp = min(perplexity, max(5.0, (len(X) - 1) / 3.0))
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=SEED).fit_transform(X)


# ---------- Phần 1 ----------
def label_distribution(stats, out: Path):
    ld = stats.get("label_distribution") if stats else None
    if not ld:
        return log("[skip] label distribution: thiếu preprocess_stats.json")
    df = pd.DataFrame(ld).fillna(0).astype(int)
    df = df.loc[df["train"].sort_values(ascending=True).index]
    df.rename_axis("label").to_csv(out / "label_counts.csv")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1.5))
    y = np.arange(len(df))
    for i, split in enumerate(["train", "val", "test"]):
        if split in df:
            ax.barh(y + (i - 1) * 0.27, df[split], height=0.27, label=split)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.set_xscale("log")
    ax.set(xlabel="số dòng (log)", title="Phân phối nhãn sau tiền xử lý")
    ax.legend()
    save(fig, out / "label_distribution.png")


def cleaning_steps(stats, out: Path):
    steps = stats.get("cleaning") if stats else None
    if not steps:
        return log("[skip] cleaning steps: preprocess_stats.json thiếu khoá 'cleaning' (chạy lại spark_preprocess.py bản mới)")
    pd.DataFrame(steps).to_csv(out / "cleaning_steps.csv", index=False)
    log("wrote cleaning_steps.csv")


def ks_analysis(train: pd.DataFrame, features, top_k: int, sample: pd.DataFrame, out: Path) -> list:
    from scipy.stats import ks_2samp
    attack = train[LABEL_COL].to_numpy() != BENIGN
    rows = []
    for f in features:
        x = train[f].to_numpy(dtype=np.float64)
        r = ks_2samp(x[~attack], x[attack]) if attack.any() and (~attack).any() else None
        rows.append({"feature": f, "ks": float(r.statistic) if r else np.nan, "pvalue": float(r.pvalue) if r else np.nan})
    ks = pd.DataFrame(rows).sort_values("ks", ascending=False).reset_index(drop=True)
    ks.to_csv(out / "ks_table.csv", index=False)
    top = ks["feature"].head(top_k).tolist()
    groups = [BENIGN] + sorted(set(sample[LABEL_COL]) - {BENIGN})
    ncol = 4
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax, f in zip(axes.flat, top):
        data = [symlog1p(sample.loc[sample[LABEL_COL] == g, f]) for g in groups]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([g[:14] for g in groups], rotation=45, ha="right", fontsize=7)
        ax.set_title("%s (KS=%.2f)" % (f, ks.set_index("feature").loc[f, "ks"]), fontsize=9)
        ax.set_ylabel("sign·log1p", fontsize=8)
    for ax in list(axes.flat)[len(top):]:
        ax.axis("off")
    save(fig, out / "top_features_box.png")
    return top


def correlation(train: pd.DataFrame, features, out: Path) -> int:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    benign = train.loc[train[LABEL_COL] == BENIGN, list(features)]
    corr = benign.corr().fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    dist = 1.0 - corr.abs().to_numpy()
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(np.clip(dist, 0, None), checks=False), method="average"))
    c = corr.iloc[order, order]
    n = len(c)
    fig, ax = plt.subplots(figsize=(0.16 * n + 4, 0.16 * n + 3))
    im = ax.imshow(c.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    fs = 8 if n <= 30 else 5
    ax.set_xticks(range(n))
    ax.set_xticklabels(c.columns, rotation=90, fontsize=fs)
    ax.set_yticks(range(n))
    ax.set_yticklabels(c.index, fontsize=fs)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_title("Tương quan feature (Benign train)")
    save(fig, out / "correlation_heatmap.png")
    iu = np.triu_indices(n, k=1)
    vals = corr.to_numpy()[iu]
    mask = np.abs(vals) > CORR_THRESHOLD
    pairs = pd.DataFrame({"feature_a": corr.index[iu[0][mask]], "feature_b": corr.columns[iu[1][mask]], "r": vals[mask]})
    pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index)
    pairs.to_csv(out / "high_corr_pairs.csv", index=False)
    return len(pairs)


def scaling(train: pd.DataFrame, scaler, top: list, out: Path):
    Z = frame_to_matrix(train, scaler)
    rate = pd.DataFrame({"feature": scaler.features, "clip_rate": (np.abs(Z) >= CLIP).mean(axis=0)}) \
        .sort_values("clip_rate", ascending=False)
    rate.to_csv(out / "clip_rate.csv", index=False)
    head = rate.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(head) + 1.5))
    ax.barh(head["feature"], head["clip_rate"])
    ax.set(xlabel="tỷ lệ giá trị bị clip ở ±%g" % CLIP, title="Clip sau chuẩn hoá")
    save(fig, out / "clip_rate.png")
    feats = top[:4]
    idx = [scaler.features.index(f) for f in feats]
    fig, axes = plt.subplots(2, len(feats), figsize=(4 * len(feats), 6), squeeze=False)
    for j, (f, k) in enumerate(zip(feats, idx)):
        axes[0, j].hist(symlog1p(train[f]), bins=60)
        axes[0, j].set_title(f, fontsize=9)
        axes[0, j].set_xlabel("gốc (sign·log1p)", fontsize=8)
        axes[1, j].hist(Z[:, k], bins=60)
        for v in (-CLIP, CLIP):
            axes[1, j].axvline(v, color="red", ls="--", lw=1)
        axes[1, j].set_xlabel("sau transform", fontsize=8)
    save(fig, out / "scaling_before_after.png")


def projections(X, labels, out: Path, skip_tsne: bool, perplexity: float, cache: dict):
    from sklearn.decomposition import PCA
    full = PCA(random_state=SEED).fit(X)
    cum = np.cumsum(full.explained_variance_ratio_)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(cum) + 1), cum, marker=".")
    ax.axhline(0.95, color="grey", ls=":")
    ax.axvline(k95, color="grey", ls=":")
    ax.set(xlabel="số thành phần", ylabel="explained variance tích luỹ",
           title="PCA: %d thành phần đạt 95%%" % k95)
    save(fig, out / "pca_explained_variance.png")
    cache["pca_raw"] = full.transform(X)[:, :2]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter_by_label(ax, cache["pca_raw"], labels, "PCA 2D (dữ liệu chuẩn hoá)")
    ax.legend(markerscale=3, fontsize=7)
    save(fig, out / "projection_pca.png")
    if not skip_tsne:
        t0 = time.time()
        cache["tsne_raw"] = tsne_2d(X, perplexity)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        scatter_by_label(ax, cache["tsne_raw"], labels, "t-SNE 2D (perplexity %g)" % perplexity)
        ax.legend(markerscale=3, fontsize=7)
        save(fig, out / "projection_tsne.png")
        log("t-SNE %.1fs" % (time.time() - t0))
    try:
        import umap
    except ImportError:
        return log("[skip] UMAP: chưa cài umap-learn")
    P = umap.UMAP(n_components=2, random_state=SEED).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter_by_label(ax, P, labels, "UMAP 2D")
    ax.legend(markerscale=3, fontsize=7)
    save(fig, out / "projection_umap.png")


# ---------- Tổng hợp ----------
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(("%.4f" % v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_summary(out: Path, timings: dict):
    lines = ["# EDA — CSE-CIC-IDS2018", "", "## Hình", ""]
    for name, desc in DESCRIPTIONS.items():
        if (out / name).exists():
            lines.append("- `%s` — %s" % (name, desc))
    if (out / "label_counts.csv").exists():
        lc = pd.read_csv(out / "label_counts.csv").set_index("label")
        rows = []
        for split in lc.columns:
            b = int(lc.loc[BENIGN, split]) if BENIGN in lc.index else 0
            a = int(lc[split].sum()) - b
            rows.append({"split": split, "benign": b, "attack": a, "attack_ratio": a / max(a + b, 1)})
        lines += ["", "## Benign / Attack theo tập", "", md_table(pd.DataFrame(rows))]
    if (out / "cleaning_steps.csv").exists():
        lines += ["", "## Hiệu quả làm sạch", "", md_table(pd.read_csv(out / "cleaning_steps.csv"))]
    if (out / "high_corr_pairs.csv").exists():
        n = len(pd.read_csv(out / "high_corr_pairs.csv"))
        lines += ["", "## Tương quan", "", "Số cặp feature có |r| > %.2f trên Benign train: %d (chi tiết: `high_corr_pairs.csv`)."
                  % (CORR_THRESHOLD, n)]
    if (out / "silhouette.csv").exists():
        lines += ["", "## Silhouette theo nhãn nhị phân", "", md_table(pd.read_csv(out / "silhouette.csv"))]
    lines += ["", "## Thời gian chạy lần gần nhất (s)", "", md_table(pd.DataFrame([timings]))]
    (out / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("wrote eda_summary.md")


def fake_stats(data_dir: Path) -> dict:
    """Chỉ dùng cho --smoke: stats giả có cùng cấu trúc với preprocess_stats.json."""
    ld = {s: {k: int(v) for k, v in pd.read_parquet(data_dir / s)[LABEL_COL].value_counts().items()}
          for s in ("train", "val", "test")}
    n = sum(sum(v.values()) for v in ld.values())
    return {"label_distribution": ld, "cleaning": [
        {"step": "read_csv", "rows": n + 13, "rows_removed": 0, "cols": 18},
        {"step": "drop_repeated_headers", "rows": n + 10, "rows_removed": 3, "cols": 18},
        {"step": "drop_nan_inf", "rows": n + 4, "rows_removed": 6, "cols": 18},
        {"step": "drop_constant_cols", "rows": n + 4, "rows_removed": 0, "cols": 17},
        {"step": "drop_duplicates", "rows": n, "rows_removed": 4, "cols": 17}]}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None, help="mặc định data.sample_local_dir")
    p.add_argument("--scaler", default=None, help="mặc định models.scaler")
    p.add_argument("--stats", default="results/preprocess_stats.json")
    p.add_argument("--out", default="reports/eda")
    p.add_argument("--per-class", type=int, default=2000, help="số dòng tối đa mỗi nhãn cho phép chiếu 2D")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--max-rows", type=int, default=200000, help="số dòng train tối đa cho KS/tương quan/chuẩn hoá")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--skip-tsne", action="store_true")
    p.add_argument("--latent", action="store_true", help="phần 2: không gian latent AE/VAE + silhouette")
    p.add_argument("--no-part1", action="store_true", help="bỏ mục 1-6 (dùng khi chỉ chạy thêm phần 2)")
    p.add_argument("--models-dir", default="models", help="chứa ae.pt, vae.pt")
    p.add_argument("--smoke", action="store_true", help="dữ liệu giả; với --latent sẽ train AE/VAE 2 epoch")
    a = p.parse_args(argv)
    cfg = C.load_config(a.config)
    a.data_dir = a.data_dir or C.get(cfg, "data.sample_local_dir")
    a.scaler = a.scaler or C.get(cfg, "models.scaler")
    return a


def prepare_smoke(a):
    from common.fake import write_fake_dataset
    tmp = Path(tempfile.mkdtemp(prefix="eda_smoke_"))
    write_fake_dataset(tmp / "data", n=4000, scaler_path=tmp / "scaler.json")
    (tmp / "stats.json").write_text(json.dumps(fake_stats(tmp / "data")), encoding="utf-8")
    a.data_dir, a.scaler, a.stats = str(tmp / "data"), str(tmp / "scaler.json"), str(tmp / "stats.json")
    if a.latent:
        from train import train_dl
        for kind in ("ae", "vae"):
            train_dl.main(["--model", kind, "--data-dir", a.data_dir, "--scaler", a.scaler, "--epochs", "2",
                           "--cpu", "--no-export", "--results-dir", str(tmp / "results"), "--models-dir", str(tmp / "models")])
        a.models_dir = str(tmp / "models")


def main(argv=None) -> dict:
    a = parse_args(argv)
    if a.smoke:
        prepare_smoke(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    timings, cache = {}, {}
    t = time.time()
    scaler = load_scaler(a.scaler)
    stats = json.loads(Path(a.stats).read_text(encoding="utf-8")) if Path(a.stats).exists() else None
    train = read_split_frame(a.data_dir, "train", cap=a.max_rows)
    sample = stratified_sample(train, a.per_class)
    X = frame_to_matrix(sample, scaler)
    labels = sample[LABEL_COL].astype(str).to_numpy()
    timings["load"] = round(time.time() - t, 1)
    log("train %d dòng, mẫu phân tầng %d dòng, %d feature" % (len(train), len(sample), len(scaler.features)))
    if not a.no_part1:
        steps = [
            ("label_distribution", lambda: label_distribution(stats, out)),
            ("cleaning", lambda: cleaning_steps(stats, out)),
            ("ks", lambda: cache.__setitem__("top", ks_analysis(train, scaler.features, a.top_k, sample, out))),
            ("correlation", lambda: correlation(train, scaler.features, out)),
            ("scaling", lambda: scaling(train, scaler, cache["top"], out)),
            ("projections", lambda: projections(X, labels, out, a.skip_tsne, a.perplexity, cache)),
        ]
        for name, fn in steps:
            t = time.time()
            fn()
            timings[name] = round(time.time() - t, 1)
    if a.latent:
        t = time.time()
        latent_space(X, labels, Path(a.models_dir), out, a.skip_tsne, a.perplexity, cache)
        timings["latent"] = round(time.time() - t, 1)
    write_summary(out, timings)
    return timings


if __name__ == "__main__":
    main()
```

`latent_space` được viết ở Task 11b. Ở Task 5b, thêm stub tạm ngay trên `main` để `--latent` báo lỗi rõ ràng:
```python
def latent_space(X, labels, models_dir, out, skip_tsne, perplexity, cache):
    raise SystemExit("--latent cần Task 11b (analysis/eda.py phần 2)")
```

- [ ] **Step 3: Chạy test, xác nhận pass**

Run: `.venv\Scripts\python -m pytest tests/test_eda.py -q`
Expected: `3 passed` (< 1 phút).

- [ ] **Step 4: Xem hình bằng mắt**

Run: `.venv\Scripts\python analysis/eda.py --smoke --out .cache/eda_check --per-class 300`, rồi mở `top_features_box.png`, `correlation_heatmap.png`, `projection_tsne.png` bằng Read tool: chữ đọc được, legend không che dữ liệu, Benign màu xám, tấn công màu riêng.

- [ ] **Step 5: README + commit**

Thêm `scipy>=1.10` vào `requirements-kaggle.txt`. Thêm vào `README.md`:
````markdown
## Trực quan hoá dữ liệu (M1b) — Kaggle CPU hoặc laptop
```bash
python analysis/eda.py --data-dir data/kaggle_sample --scaler data/kaggle_sample/scaler.json \
  --stats data/kaggle_sample/preprocess_stats.json --out reports/eda           # phần 1, mục 1-6
python analysis/eda.py --latent --no-part1 --models-dir models --out reports/eda   # phần 2, sau train DL
```
Kết quả: `reports/eda/*.png` + `reports/eda/eda_summary.md`. UMAP chỉ vẽ khi đã `pip install umap-learn`.
Chậm nhất là t-SNE; giảm `--per-class` (mặc định 2000) nếu vượt 10 phút.
````

```bash
git add analysis/__init__.py analysis/eda.py tests/test_eda.py requirements-kaggle.txt README.md
git commit -m "feat(analysis): EDA part 1 - label dist, cleaning, KS, correlation, scaling, 2D projections (M1b)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Ghi chú: Task 11 cũng tạo `analysis/__init__.py`; nếu file đã có thì bỏ qua.

---

### Task 6: Refactor `train/baselines.py` sang `common/`

**Điều kiện tiên quyết:** người dùng đã copy `train/baselines.py` vào repo. Nếu chưa có → DỪNG, hỏi người dùng. Không tự viết lại.

**Files:**
- Modify: `train/baselines.py`
- Create: `tests/compare_results.py`, `tests/test_baselines_smoke.py`

**Interfaces:**
- Consumes: `common.data.load_scaler`, `load_split`, `Split`; `common.metrics.evaluate`, `per_attack_recall`, `save_scores`, `METRIC_COLUMNS`; `common.fake.write_fake_dataset`.
- Produces (định dạng KHÔNG đổi): `<results_dir>/baselines_metrics.csv` (cột = `METRIC_COLUMNS`), `<results_dir>/baselines_per_attack.csv`, `<results_dir>/scores_<model>.npz`. CLI phải có (thêm nếu thiếu, giữ flag cũ): `--data-dir`, `--scaler` (mặc định `models/scaler.json`), `--results-dir` (mặc định `results`), `--test-cap` (mặc định 300000), `--holdout-attack`, `--smoke`.

- [ ] **Step 1: Đọc file gốc, ghi lại hành vi**

Đọc toàn bộ `train/baselines.py`. Ghi ra (trong mô tả commit): danh sách flag CLI, tên model trong cột `model` (cần cho Task 11), cách fit scaler (trên đâu, ddof), cách lấy mẫu test, cách tính từng metric, cách xử lý `--holdout-attack`. Nếu bất kỳ điểm nào trái với PROJECT_SPEC mục 4 (vd scaler fit trên toàn bộ train, ddof=1, không clip) → ghi lại và **hỏi người dùng** trước khi đổi hành vi.

- [ ] **Step 2: Chụp output gốc (golden) trên dữ liệu giả**

```powershell
.venv\Scripts\python -m common.fake --out data/fake --n 4000
# Chạy file GỐC với flag của nó, trỏ vào data/fake, ghi ra thư mục golden.
# Ví dụ (đổi theo flag thật đã ghi ở Step 1):
.venv\Scripts\python train/baselines.py --data-dir data/fake --results-dir .cache/golden
```
Nếu file gốc không cho đổi thư mục output, chạy trong một bản copy tạm của repo và lấy `results/` ra `.cache/golden/`.

- [ ] **Step 3: Viết `tests/compare_results.py`**

```python
"""So output baseline cũ/mới: python tests/compare_results.py OLD_DIR NEW_DIR"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

IGNORE = ("train_time_s", "infer_ms_per_1k")


def compare_dirs(old: Path, new: Path) -> None:
    for name in ("baselines_metrics.csv", "baselines_per_attack.csv"):
        a, b = pd.read_csv(old / name), pd.read_csv(new / name)
        assert list(a.columns) == list(b.columns), (name, list(a.columns), list(b.columns))
        a = a.drop(columns=[c for c in IGNORE if c in a.columns])
        b = b.drop(columns=[c for c in IGNORE if c in b.columns])
        pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-6)
    for f in sorted(old.glob("scores_*.npz")):
        x, y = np.load(f), np.load(new / f.name)
        assert set(x.files) == set(y.files) == {"y", "score", "thr"}, f.name
        np.testing.assert_array_equal(x["y"], y["y"])
        np.testing.assert_allclose(x["score"], y["score"], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    compare_dirs(Path(sys.argv[1]), Path(sys.argv[2]))
    print("IDENTICAL (bỏ qua cột thời gian)")
```

- [ ] **Step 4: Viết test smoke fail**

`tests/test_baselines_smoke.py`:
```python
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from common.metrics import METRIC_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")


def test_baselines_smoke(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "train" / "baselines.py"), "--smoke",
                        "--results-dir", str(tmp_path)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    m = pd.read_csv(tmp_path / "baselines_metrics.csv")
    assert list(m.columns) == METRIC_COLUMNS
    assert len(m) == 6 and set(m["group"]) == {"unsupervised", "supervised"}
    pa = pd.read_csv(tmp_path / "baselines_per_attack.csv")
    assert list(pa.columns) == ["model", "attack", "recall", "n"]
    assert len(list(tmp_path.glob("scores_*.npz"))) == 6
```

Run: `.venv\Scripts\python -m pytest tests/test_baselines_smoke.py -q`
Expected: FAIL (chưa có `--smoke` hoặc `--results-dir`).

- [ ] **Step 5: Refactor**

Trong `train/baselines.py`:
1. Xoá định nghĩa local của `load_split`, scaler fit/transform, `evaluate`, per-attack recall, lưu npz; thay bằng:
   ```python
   import sys
   from pathlib import Path
   ROOT = Path(__file__).resolve().parents[1]
   if str(ROOT) not in sys.path:
       sys.path.insert(0, str(ROOT))
   from common.data import load_scaler, load_split
   from common.metrics import METRIC_COLUMNS, evaluate, per_attack_recall, save_scores
   ```
2. Scaler: `scaler = load_scaler(args.scaler)` (không fit trong script nữa).
3. Test: `te = load_split(args.data_dir, "test", scaler, cap=args.test_cap)`; giữ nguyên các cap train/OCSVM 20k cũ.
4. Holdout: `tr = tr.subset(tr.labels != args.holdout_attack)` (và val nếu file gốc lọc val); ghi `Path(args.results_dir) / "holdout.json"` = `{"attack": args.holdout_attack}`.
5. Ghi metric bằng `pd.DataFrame(rows, columns=METRIC_COLUMNS).to_csv(...)`.
6. Thêm `--smoke`:
   ```python
   if args.smoke:
       import tempfile
       from common.fake import write_fake_dataset
       tmp = Path(tempfile.mkdtemp(prefix="baselines_smoke_"))
       write_fake_dataset(tmp / "data", n=4000, scaler_path=tmp / "scaler.json")
       args.data_dir, args.scaler = str(tmp / "data"), str(tmp / "scaler.json")
   ```
   (và giảm các cap train/OCSVM nếu chúng lớn hơn số dòng giả — không bắt buộc vì cap > số dòng thì lấy hết).

- [ ] **Step 6: Kiểm tra hành vi giữ nguyên**

```powershell
.venv\Scripts\python train/baselines.py --data-dir data/fake --scaler data/fake/scaler.json --results-dir .cache/new
.venv\Scripts\python tests/compare_results.py .cache/golden .cache/new
```
Expected: `IDENTICAL (bỏ qua cột thời gian)`. Nếu khác vì bản gốc fit scaler khác spec (Step 1) → báo người dùng chênh lệch cụ thể; giữ phiên bản theo spec mục 4.

Run: `.venv\Scripts\python -m pytest tests/test_baselines_smoke.py -q` → `1 passed`.

- [ ] **Step 7: Commit**

```bash
git add train/baselines.py tests/compare_results.py tests/test_baselines_smoke.py
git commit -m "refactor(baselines): use common/ data+metrics, add --smoke/--results-dir" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `train/dl_models.py`

**Files:**
- Create: `train/dl_models.py`
- Test: `tests/test_dl_models.py`

**Interfaces:**
- Consumes: không.
- Produces: `MODEL_SPECS`, `AE`, `VAE`, `DAE`, `MLPClassifier`, `ReconScore`, `ProbScore`, `build_model`, `score_module`, `vae_loss`, `count_params`, `encode_latent(model, x)` (AE/DAE: output encoder; VAE: μ), `save_checkpoint(path, kind, model, d, hidden, latent, noise_std)`, `load_checkpoint(path) -> (kind, model.eval())` (mục Interfaces). AE/DAE `forward(x) -> recon`; VAE `forward(x) -> (recon, mu, logvar)` (eval dùng μ), `reconstruct_mean(x)`; MLP `forward(x) -> logits [batch]`; score module `forward(x) -> [batch]`.

Ghi chú kiến trúc: Linear → BatchNorm1d → ReLU cho mỗi tầng ẩn; tầng latent (bottleneck) và tầng ra là Linear thuần (không BN/ReLU). Nhiễu DAE nằm trong `forward` chỉ khi `self.training` → graph ONNX (eval) không có nhiễu.

- [ ] **Step 1: Viết test fail**

`tests/test_dl_models.py`:
```python
import pytest

torch = pytest.importorskip("torch")
from train.dl_models import (MODEL_SPECS, VAE, build_model, count_params,  # noqa: E402
                             score_module, vae_loss)

pytestmark = pytest.mark.torch
D = 20


def _x(n=64):
    torch.manual_seed(0)
    return torch.randn(n, D)


@pytest.mark.parametrize("kind", ["ae", "dae"])
def test_ae_shapes_and_score(kind):
    m = build_model(kind, D).eval()
    x = _x()
    recon = m(x)
    assert recon.shape == x.shape
    s = score_module(kind, m)(x)
    assert s.shape == (64,)
    torch.testing.assert_close(s, ((recon - x) ** 2).mean(dim=1))


def test_dae_noise_only_in_training():
    m = build_model("dae", D, noise_std=0.5)
    x = _x()
    m.eval()
    torch.testing.assert_close(m(x), m(x))
    m.train()
    torch.manual_seed(1)
    a = m(x)
    torch.manual_seed(2)
    b = m(x)
    assert not torch.allclose(a, b)


def test_vae_eval_is_deterministic_and_uses_mu():
    m = build_model("vae", D).eval()
    x = _x()
    r1, mu, logvar = m(x)
    r2, _, _ = m(x)
    torch.testing.assert_close(r1, r2)
    assert mu.shape == logvar.shape == (64, 16)
    torch.testing.assert_close(score_module("vae", m)(x), ((m.reconstruct_mean(x) - x) ** 2).mean(dim=1))


def test_vae_loss_components():
    x = _x()
    loss, rec, kl = vae_loss(x, torch.zeros_like(x), torch.zeros(64, 16), torch.zeros(64, 16), beta=2.0)
    torch.testing.assert_close(rec, (x ** 2).sum(dim=1).mean())
    torch.testing.assert_close(kl, torch.tensor(0.0))
    torch.testing.assert_close(loss, rec)


def test_mlp_probability():
    m = build_model("mlp", D).eval()
    logits = m(_x())
    assert logits.shape == (64,)
    p = score_module("mlp", m)(_x())
    assert ((p > 0) & (p < 1)).all()


def test_depth_changes_param_count():
    shallow = count_params(build_model("ae", D, hidden=(32,)))
    default = count_params(build_model("ae", D, hidden=(64, 32)))
    deep = count_params(build_model("ae", D, hidden=(128, 64, 32)))
    assert shallow < default < deep


def test_model_specs():
    assert {k: v["name"] for k, v in MODEL_SPECS.items()} == {
        "ae": "DeepAutoencoder", "vae": "VariationalAE", "dae": "DenoisingAE", "mlp": "DNN_Classifier"}
    assert isinstance(build_model("vae", D), VAE)


@pytest.mark.parametrize("kind", ["ae", "vae"])
def test_latent_and_checkpoint_roundtrip(tmp_path, kind):
    from train.dl_models import encode_latent, load_checkpoint, save_checkpoint
    m = build_model(kind, D, hidden=(64, 32), latent=8).eval()
    x = _x()
    z = encode_latent(m, x)
    assert z.shape == (64, 8)
    save_checkpoint(tmp_path / "m.pt", kind, m, D, (64, 32), 8, 0.1)
    k2, m2 = load_checkpoint(tmp_path / "m.pt")
    assert k2 == kind and not m2.training
    torch.testing.assert_close(encode_latent(m2, x), z)


def test_mlp_has_no_latent():
    from train.dl_models import encode_latent
    with pytest.raises(ValueError):
        encode_latent(build_model("mlp", D), _x())
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `.venv\Scripts\python -m pytest tests/test_dl_models.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'train.dl_models'`.

- [ ] **Step 3: Viết `train/dl_models.py`**

```python
"""4 model DL dạng MLP cho dữ liệu bảng: AE, VAE, DAE, DNN classifier (PROJECT_SPEC 5.3)."""
from __future__ import annotations

import torch
from torch import nn

MODEL_SPECS = {
    "ae": {"name": "DeepAutoencoder", "group": "unsupervised"},
    "vae": {"name": "VariationalAE", "group": "unsupervised"},
    "dae": {"name": "DenoisingAE", "group": "unsupervised"},
    "mlp": {"name": "DNN_Classifier", "group": "supervised"},
}


def mlp_stack(dims, dropout: float = 0.0) -> list:
    layers = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    return layers


def make_decoder(d: int, hidden, latent: int) -> nn.Sequential:
    rev = list(reversed(hidden))
    return nn.Sequential(*mlp_stack([latent] + rev), nn.Linear(rev[-1], d))


class AE(nn.Module):
    def __init__(self, d: int, hidden=(64, 32), latent: int = 16):
        super().__init__()
        hidden = list(hidden)
        self.encoder = nn.Sequential(*mlp_stack([d] + hidden), nn.Linear(hidden[-1], latent))
        self.decoder = make_decoder(d, hidden, latent)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DAE(AE):
    def __init__(self, d: int, hidden=(64, 32), latent: int = 16, noise_std: float = 0.1):
        super().__init__(d, hidden, latent)
        self.noise_std = noise_std

    def forward(self, x):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return self.decoder(self.encoder(x))


class VAE(nn.Module):
    def __init__(self, d: int, hidden=(64, 32), latent: int = 16):
        super().__init__()
        hidden = list(hidden)
        self.body = nn.Sequential(*mlp_stack([d] + hidden))
        self.mu = nn.Linear(hidden[-1], latent)
        self.logvar = nn.Linear(hidden[-1], latent)
        self.decoder = make_decoder(d, hidden, latent)

    def encode(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decoder(z), mu, logvar

    def reconstruct_mean(self, x):
        mu, _ = self.encode(x)
        return self.decoder(mu)


def vae_loss(x, recon, mu, logvar, beta: float):
    """MSE tổng theo feature, trung bình theo batch + beta * KL."""
    rec = ((recon - x) ** 2).sum(dim=1).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return rec + beta * kl, rec, kl


class MLPClassifier(nn.Module):
    def __init__(self, d: int, hidden=(128, 64, 32), dropout: float = 0.2):
        super().__init__()
        hidden = list(hidden)
        self.net = nn.Sequential(*mlp_stack([d] + hidden, dropout), nn.Linear(hidden[-1], 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ReconScore(nn.Module):
    """Score = MSE tái tạo theo mẫu (VAE dùng μ). Đây là graph được export ONNX."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        recon = self.model.reconstruct_mean(x) if isinstance(self.model, VAE) else self.model(x)
        return ((recon - x) ** 2).mean(dim=1)


class ProbScore(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def build_model(kind: str, d: int, hidden=(64, 32), latent: int = 16, noise_std: float = 0.1) -> nn.Module:
    if kind == "ae":
        return AE(d, hidden, latent)
    if kind == "dae":
        return DAE(d, hidden, latent, noise_std)
    if kind == "vae":
        return VAE(d, hidden, latent)
    if kind == "mlp":
        return MLPClassifier(d)
    raise ValueError("unknown model kind: %s" % kind)


def score_module(kind: str, model: nn.Module) -> nn.Module:
    return ProbScore(model) if kind == "mlp" else ReconScore(model)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def encode_latent(model: nn.Module, x):
    """Vector latent cho EDA: AE/DAE -> output encoder; VAE -> μ. (MLP không có latent.)"""
    if isinstance(model, VAE):
        return model.encode(x)[0]
    if isinstance(model, AE):
        return model.encoder(x)
    raise ValueError("model không có latent: %s" % type(model).__name__)


def save_checkpoint(path, kind: str, model: nn.Module, d: int, hidden, latent: int, noise_std: float) -> None:
    torch.save({"kind": kind, "d": d, "hidden": list(hidden), "latent": latent, "noise_std": noise_std,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, str(path))


def load_checkpoint(path):
    """-> (kind, model ở chế độ eval trên CPU)."""
    ck = torch.load(str(path), map_location="cpu")
    model = build_model(ck["kind"], ck["d"], hidden=tuple(ck["hidden"]), latent=ck["latent"],
                        noise_std=ck["noise_std"])
    model.load_state_dict(ck["state_dict"])
    return ck["kind"], model.eval()
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv\Scripts\python -m pytest tests/test_dl_models.py -q`
Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add train/dl_models.py tests/test_dl_models.py
git commit -m "feat(train): AE/VAE/DAE/MLP models with ONNX score wrappers" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `train/train_dl.py` (train, đánh giá, export ONNX)

**Files:**
- Create: `train/train_dl.py`
- Test: `tests/test_train_dl.py`

**Interfaces:**
- Consumes: `common.data` (`load_scaler`, `load_split`, `SEED`), `common.metrics` (`evaluate`, `per_attack_recall`, `save_scores`, `METRIC_COLUMNS`), `common.fake.write_fake_dataset`, `train.dl_models` (`MODEL_SPECS`, `build_model`, `score_module`, `vae_loss`, `count_params`).
- Produces: `parse_args(argv)`, `run(args) -> dict`, `main(argv)`. Flag: `--model {ae,vae,dae,mlp}` (bắt buộc), `--config`, `--data-dir`, `--scaler`, `--models-dir models`, `--results-dir results`, `--test-cap`, `--train-cap`, `--hidden 64,32`, `--latent 16`, `--beta 1.0`, `--kl-warmup 5`, `--noise-std 0.1`, `--percentile 99`, `--epochs 50`, `--batch-size 1024`, `--lr 1e-3`, `--patience 5`, `--holdout-attack`, `--no-export`, `--cpu`, `--smoke`. File output như mục "File output".

Ghi chú early stopping VAE: trong `kl_warmup` epoch đầu β đang tăng nên val loss không so sánh được → chỉ bắt đầu theo dõi best/patience từ epoch `kl_warmup` trở đi (ghi trong `--help`).
Ghi chú ONNX: so khớp ORT vs PyTorch bằng `allclose(rtol=1e-4, atol=1e-4)` trên 1000 mẫu (MSE có thể lớn nên dùng sai số tương đối), in `max_abs_diff`.

- [ ] **Step 1: Viết test fail**

`tests/test_train_dl.py`:
```python
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
```

- [ ] **Step 2: Chạy test, xác nhận fail**

Run: `.venv\Scripts\python -m pytest tests/test_train_dl.py -q`
Expected: FAIL — `ImportError: cannot import name 'train_dl'`.

- [ ] **Step 3: Viết `train/train_dl.py`**

```python
"""Train 1 model DL (ae|vae|dae|mlp), đánh giá trên test chung, export ONNX trả thẳng score.

Kaggle: python train/train_dl.py --model ae --data-dir /kaggle/input/btl-cicids2018-sample
Smoke:  python train/train_dl.py --model ae --smoke      (dữ liệu giả, 2 epoch, CPU)
VAE: early stopping chỉ tính từ epoch --kl-warmup (trước đó β đang tăng nên val loss không so được).
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as C  # noqa: E402
from common.data import SEED, load_scaler, load_split  # noqa: E402
from common.metrics import METRIC_COLUMNS, evaluate, per_attack_recall, save_scores  # noqa: E402
from train.dl_models import (MODEL_SPECS, build_model, count_params, save_checkpoint,  # noqa: E402
                             score_module, vae_loss)

ONNX_OPSET = 17


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None, help="thư mục có train/ val/ test/ (mặc định data.sample_local_dir)")
    p.add_argument("--scaler", default=None, help="mặc định models.scaler")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--test-cap", type=int, default=None)
    p.add_argument("--train-cap", type=int, default=None)
    p.add_argument("--hidden", default="64,32", help="tầng ẩn encoder AE/VAE/DAE (MLP cố định 128,64,32)")
    p.add_argument("--latent", type=int, default=16)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--kl-warmup", type=int, default=5)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--percentile", type=float, default=99.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--holdout-attack", default=None, help="bỏ loại tấn công này khỏi train/val")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true", help="dữ liệu giả 4000 dòng, 2 epoch, CPU")
    a = p.parse_args(argv)
    cfg = C.load_config(a.config)
    a.data_dir = a.data_dir or C.get(cfg, "data.sample_local_dir")
    a.scaler = a.scaler or C.get(cfg, "models.scaler")
    a.test_cap = a.test_cap or C.get(cfg, "data.test_cap", 300000)
    return a


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def batch_loss(kind, model, xb, yb, beta, bce):
    """Trả về (loss, recon, kl); recon/kl chỉ có ý nghĩa với VAE."""
    if kind == "vae":
        recon, mu, logvar = model(xb)
        return vae_loss(xb, recon, mu, logvar, beta)
    if kind == "mlp":
        return bce(model(xb), yb), None, None
    recon = model(xb)          # DAE tự cộng nhiễu trong forward khi training
    return ((recon - xb) ** 2).mean(), None, None


@torch.no_grad()
def eval_loss(kind, model, X, Y, beta, bce, batch=16384):
    model.eval()
    tot = rec = kl = 0.0
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        yb = Y[i:i + batch] if Y is not None else None
        loss, r, k = batch_loss(kind, model, xb, yb, beta, bce)
        w = len(xb)
        tot += loss.item() * w
        if r is not None:
            rec += r.item() * w
            kl += k.item() * w
    n = max(len(X), 1)
    return tot / n, rec / n, kl / n


def train_model(kind, model, tr, va, args, device):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Xtr = torch.from_numpy(tr.X).to(device)
    Xva = torch.from_numpy(va.X).to(device)
    Ytr = Yva = bce = None
    if kind == "mlp":
        Ytr = torch.from_numpy(tr.y.astype(np.float32)).to(device)
        Yva = torch.from_numpy(va.y.astype(np.float32)).to(device)
        n_pos = int(tr.y.sum())
        pos_weight = torch.tensor([(len(tr.y) - n_pos) / max(n_pos, 1)], device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    gen = torch.Generator().manual_seed(SEED)
    best, best_state, bad, hist = float("inf"), copy.deepcopy(model.state_dict()), 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        beta = args.beta * min(1.0, epoch / max(args.kl_warmup, 1)) if kind == "vae" else None
        model.train()
        tot = torch.zeros((), device=device)
        rec = torch.zeros((), device=device)
        kl = torch.zeros((), device=device)
        nb = 0
        perm = torch.randperm(len(Xtr), generator=gen).to(device)
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            if len(idx) < 2:          # BatchNorm cần >= 2 mẫu
                continue
            loss, r, k = batch_loss(kind, model, Xtr[idx], Ytr[idx] if Ytr is not None else None, beta, bce)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.detach()
            if r is not None:
                rec += r.detach()
                kl += k.detach()
            nb += 1
        v_loss, v_rec, v_kl = eval_loss(kind, model, Xva, Yva, beta, bce)
        row = {"epoch": epoch, "train_loss": tot.item() / max(nb, 1), "val_loss": v_loss}
        if kind == "vae":
            row.update(train_recon=rec.item() / max(nb, 1), train_kl=kl.item() / max(nb, 1),
                       val_recon=v_rec, val_kl=v_kl, beta=beta)
        hist.append(row)
        print("[%s] epoch %3d train %.5f val %.5f" % (kind, epoch, row["train_loss"], v_loss), flush=True)
        if kind == "vae" and epoch < args.kl_warmup:
            best_state = copy.deepcopy(model.state_dict())
            continue
        if v_loss < best:
            best, best_state, bad = v_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print("[%s] early stop at epoch %d (best val %.5f)" % (kind, epoch, best), flush=True)
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), time.time() - t0


@torch.no_grad()
def batched_score(smod, X, device, batch: int = 16384) -> np.ndarray:
    smod.eval()
    out = [smod(torch.from_numpy(X[i:i + batch]).to(device)).float().cpu().numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def export_onnx(smod, d: int, path: Path) -> None:
    import onnx
    m = copy.deepcopy(smod).cpu().eval()
    kwargs = dict(input_names=["input"], output_names=["score"], opset_version=ONNX_OPSET,
                  dynamic_axes={"input": {0: "batch"}, "score": {0: "batch"}})
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(m, torch.zeros(2, d), str(path), **kwargs)
    model = onnx.load(str(path))
    model.ir_version = min(model.ir_version, 8)      # onnxruntime cũ trên cụm (Py3.8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def verify_onnx(smod, path: Path, X: np.ndarray, device) -> float:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(["score"], {"input": X})[0]
    ref = batched_score(smod, X, device)
    diff = float(np.max(np.abs(got - ref))) if len(X) else 0.0
    if not np.allclose(got, ref, rtol=1e-4, atol=1e-4):
        raise RuntimeError("ONNX khác PyTorch: max_abs_diff=%g" % diff)
    print("[onnx] %s OK, max_abs_diff=%.2e trên %d mẫu" % (path.name, diff, len(X)), flush=True)
    return diff


def run(args) -> dict:
    set_seed()
    kind = args.model
    spec = MODEL_SPECS[kind]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    res, mdir = Path(args.results_dir), Path(args.models_dir)
    res.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    t_load = time.time()
    scaler = load_scaler(args.scaler)
    supervised = kind == "mlp"
    tr = load_split(args.data_dir, "train", scaler, cap=args.train_cap, benign_only=not supervised)
    va = load_split(args.data_dir, "val", scaler, benign_only=not supervised)
    if args.holdout_attack:
        tr = tr.subset(tr.labels != args.holdout_attack)
        va = va.subset(va.labels != args.holdout_attack)
        with open(res / "holdout.json", "w", encoding="utf-8") as f:
            json.dump({"attack": args.holdout_attack}, f)
    te = load_split(args.data_dir, "test", scaler, cap=args.test_cap)
    print("[data] train %d val %d test %d d=%d (%.1fs) device=%s" % (
        len(tr.y), len(va.y), len(te.y), tr.X.shape[1], time.time() - t_load, device), flush=True)

    d = tr.X.shape[1]
    hidden = tuple(int(h) for h in args.hidden.split(","))
    model = build_model(kind, d, hidden=hidden, latent=args.latent, noise_std=args.noise_std).to(device)
    n_params = count_params(model)
    model, hist, train_time = train_model(kind, model, tr, va, args, device)
    hist.to_csv(res / ("dl_%s_history.csv" % kind), index=False)
    # state_dict + kiến trúc để analysis/eda.py --latent dựng lại encoder (PROJECT_SPEC 5.10 mục 7)
    save_checkpoint(mdir / ("%s.pt" % kind), kind, model, d, hidden, args.latent, args.noise_std)

    smod = score_module(kind, model).to(device).eval()
    if supervised:
        thr, pct = 0.5, None
    else:
        val_scores = batched_score(smod, va.X, device)
        thr, pct = float(np.percentile(val_scores, args.percentile)), args.percentile
        np.save(res / ("valscores_%s.npy" % spec["name"]), val_scores)
    with open(mdir / ("threshold_%s.json" % kind), "w", encoding="utf-8") as f:
        json.dump({"percentile": pct, "threshold": thr, "model": spec["name"]}, f, indent=1)

    t0 = time.time()
    scores = batched_score(smod, te.X, device)
    infer_ms_per_1k = (time.time() - t0) / max(len(te.y), 1) * 1e6
    ev = evaluate(te.y, scores, thr)
    per_attack_recall(spec["name"], te.labels, scores >= thr).to_csv(
        res / ("dl_%s_per_attack.csv" % kind), index=False)
    save_scores(res / ("scores_%s.npz" % spec["name"]), te.y, scores, thr)
    row = {"model": spec["name"], "group": spec["group"], **ev,
           "train_time_s": round(train_time, 2), "infer_ms_per_1k": round(infer_ms_per_1k, 4),
           "n_params": n_params}
    pd.DataFrame([row], columns=METRIC_COLUMNS + ["n_params"]).to_csv(
        res / ("dl_%s_metrics.csv" % kind), index=False)

    if not args.no_export:
        onnx_path = mdir / ("%s.onnx" % kind)
        export_onnx(smod, d, onnx_path)
        verify_onnx(smod, onnx_path, te.X[:1000], device)
        if kind == "ae":      # streaming mặc định dùng models/ae.onnx + models/threshold.json
            shutil.copyfile(mdir / "threshold_ae.json", mdir / "threshold.json")
    print(json.dumps(row, indent=1), flush=True)
    return row


def main(argv=None) -> dict:
    args = parse_args(argv)
    if args.smoke:
        from common.fake import write_fake_dataset
        tmp = Path(tempfile.mkdtemp(prefix="train_dl_smoke_"))
        write_fake_dataset(tmp / "data", n=4000, scaler_path=tmp / "scaler.json")
        args.data_dir, args.scaler = str(tmp / "data"), str(tmp / "scaler.json")
        args.epochs, args.cpu = 2, True
        args.kl_warmup = min(args.kl_warmup, 2)
    return run(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv\Scripts\python -m pytest tests/test_train_dl.py -q`
Expected: `5 passed`. Nếu `torch.onnx.export` lỗi do thiếu `onnxscript` (torch mới), cài `pip install onnxscript` và ghi vào `requirements-kaggle.txt`.

- [ ] **Step 5: Kiểm tra ONNX đọc được bằng onnxruntime 1.17.3 trên Python 3.8**

```powershell
.venv\Scripts\python train/train_dl.py --model ae --smoke --models-dir .cache/onnxcheck --results-dir .cache/onnxcheck
docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster python -c "import onnxruntime as o, numpy as np; s=o.InferenceSession('.cache/onnxcheck/ae.onnx', providers=['CPUExecutionProvider']); print(s.run(['score'], {'input': np.zeros((3,16), np.float32)})[0])"
```
Expected: in ra mảng 3 số, không lỗi `Unsupported model IR version`.

- [ ] **Step 6: Commit**

```bash
git add train/train_dl.py tests/test_train_dl.py
git commit -m "feat(train): train_dl for ae/vae/dae/mlp with eval + ONNX export (opset 17)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Chạy phần train trên Kaggle (`kaggle_run_all.sh` + notebook) — kết thúc M2a

**Files:**
- Create: `train/kaggle_run_all.sh`, `train/kaggle_notebook.ipynb`
- Test: `tests/test_kaggle_runner.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `train/train_dl.py`, `train/baselines.py` (flag `--data-dir --scaler --results-dir --holdout-attack`), `train/run_ablation.py` (Task 10: `--data-dir --scaler --base-results-dir --out-root --out-csv --only --epochs --cpu`), `analysis/make_report_assets.py` (Task 11: `--results-dir --reports-dir`).
- Consumes thêm: `analysis/eda.py` (Task 5b/11b: `--data-dir --scaler --stats --out --latent --no-part1 --models-dir`).
- Produces: `bash train/kaggle_run_all.sh <DATA_DIR> <SCALER_JSON>`; `SMOKE=1 bash train/kaggle_run_all.sh` chạy cả pipeline trên dữ liệu giả vào `.cache/smoke/`. Biến môi trường: `PY` (mặc định `python`), `HOLDOUT` (mặc định `DDOS attack-HOIC`; nếu `baselines.py` gốc có giá trị mặc định khác thì dùng giá trị đó), `STAGES` (mặc định `all`).

Ghi chú: script này gọi Task 10 và 11. Viết xong Task 9 thì test cấu trúc trước (Step 3); bước chạy `SMOKE=1` đầy đủ nằm ở Task 11, Step 6.

- [ ] **Step 1: Viết test fail**

`tests/test_kaggle_runner.py`:
```python
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("bash") is None, reason="cần bash")
def test_runner_bash_syntax():
    r = subprocess.run(["bash", "-n", str(ROOT / "train" / "kaggle_run_all.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_notebook_is_valid_json_with_runner_cell():
    nb = json.loads((ROOT / "train" / "kaggle_notebook.ipynb").read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert nb["nbformat"] == 4 and "kaggle_run_all.sh" in src
```

Run: `.venv\Scripts\python -m pytest tests/test_kaggle_runner.py -q` → FAIL (file chưa có).

- [ ] **Step 2: Viết `train/kaggle_run_all.sh`**

```bash
#!/usr/bin/env bash
# Chạy toàn bộ phần train theo thứ tự ưu tiên của spec (AE > MLP > VAE > ablation latent/ngưỡng > DAE > depth > VAE β).
# Kaggle: bash train/kaggle_run_all.sh /kaggle/input/btl-cicids2018-sample /kaggle/input/btl-cicids2018-sample/scaler.json
# Smoke:  SMOKE=1 bash train/kaggle_run_all.sh      (dữ liệu giả, CPU, ghi vào .cache/smoke)
# Chạy một phần: STAGES="ae mlp" bash train/kaggle_run_all.sh ...
# EDA phần 1 trên notebook CPU riêng (song song lúc GPU train): STAGES="eda" bash train/kaggle_run_all.sh ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-python}"
HOLDOUT="${HOLDOUT:-DDOS attack-HOIC}"
STAGES="${STAGES:-all}"

if [ "${SMOKE:-0}" = "1" ]; then
  OUT=.cache/smoke
  rm -rf "$OUT" && mkdir -p "$OUT"
  "$PY" -m common.fake --out "$OUT/data" --n 4000
  DATA="$OUT/data"; SCALER="$OUT/data/scaler.json"
  RES="$OUT/results"; MOD="$OUT/models"; REP="$OUT/reports"
  STATS="$OUT/no_stats.json"          # không có -> eda bỏ qua mục 1-2
  EXTRA=(--epochs 2 --cpu); HOLDOUT="Bot"
else
  DATA="${1:?usage: kaggle_run_all.sh DATA_DIR SCALER_JSON}"; SCALER="${2:?SCALER_JSON}"
  RES=results; MOD=models; REP=reports; EXTRA=()
  STATS="${STATS:-$(dirname "$SCALER")/preprocess_stats.json}"
  mkdir -p "$MOD"
  cp "$SCALER" "$MOD/scaler.json"; SCALER="$MOD/scaler.json"
fi
COMMON=(--data-dir "$DATA" --scaler "$SCALER")
want() { [ "$STAGES" = "all" ] || [[ " $STAGES " == *" $1 "* ]]; }
step() { echo; echo "=== $(date +%H:%M:%S) $*"; }
dl() { "$PY" train/train_dl.py --model "$1" "${COMMON[@]}" --results-dir "$RES" --models-dir "$MOD" "${EXTRA[@]}"; }
ablation() { "$PY" train/run_ablation.py "${COMMON[@]}" --base-results-dir "$RES" --out-root "$RES/ablation" \
  --out-csv "$RES/ablation.csv" --only "$1" "${EXTRA[@]}"; }

want ae        && { step "AE";        dl ae; }
want mlp       && { step "MLP";       dl mlp; }
want vae       && { step "VAE";       dl vae; }
want baselines && { step "Baselines"; "$PY" train/baselines.py "${COMMON[@]}" --results-dir "$RES"; }
want eda       && { step "EDA phần 1"; "$PY" analysis/eda.py "${COMMON[@]}" --stats "$STATS" --out "$REP/eda"; }
want eda_latent && { step "EDA latent"; "$PY" analysis/eda.py "${COMMON[@]}" --latent --no-part1   --models-dir "$MOD" --out "$REP/eda"; }
want abl1      && { step "Ablation latent + threshold"; ablation latent,threshold; }
want dae       && { step "DAE";       dl dae; }
want abl2      && { step "Ablation depth"; ablation depth; }
want holdout   && {
  step "Holdout: $HOLDOUT"
  "$PY" train/train_dl.py --model mlp "${COMMON[@]}" --holdout-attack "$HOLDOUT" --no-export \
    --results-dir "$RES/holdout" --models-dir "$RES/holdout/models" "${EXTRA[@]}"
  "$PY" train/baselines.py "${COMMON[@]}" --holdout-attack "$HOLDOUT" --results-dir "$RES/holdout"
}
want abl3      && { step "Ablation VAE beta"; ablation vae_beta; }
want report    && { step "Report assets"; "$PY" analysis/make_report_assets.py --results-dir "$RES" --reports-dir "$REP"; }
step "DONE"
```

Lưu ý: dòng `want x && {...}` trả mã 1 khi stage bị bỏ qua, nên script KHÔNG được kết thúc bằng một dòng `want` (đã có `step "DONE"` ở cuối). `set -e` không dừng ở vế trái của `&&` nên các stage bị bỏ qua không làm script dừng.

- [ ] **Step 3: Viết `train/kaggle_notebook.ipynb`**

```json
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": [
   "# BTL_BigDeep — train trên Kaggle\n",
   "Input: dataset `btl-cicids2018-sample` (train/ val/ test/ + scaler.json) và dataset code `btl-bigdeep-code` (zip repo). Bật GPU."]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "!pip install -q onnx onnxruntime pyyaml\n",
   "import shutil, os\n",
   "shutil.copytree('/kaggle/input/btl-bigdeep-code', '/kaggle/working/repo', dirs_exist_ok=True)\n",
   "os.chdir('/kaggle/working/repo')"]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "!SMOKE=1 bash train/kaggle_run_all.sh"]},
  {"cell_type": "markdown", "metadata": {}, "source": [
   "EDA phần 1 chỉ cần CPU: có thể chạy ở một notebook CPU riêng song song với notebook GPU bằng ô dưới (bỏ qua ô này ở notebook GPU)."]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "!STAGES=\"eda\" bash train/kaggle_run_all.sh /kaggle/input/btl-cicids2018-sample /kaggle/input/btl-cicids2018-sample/scaler.json"]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "!bash train/kaggle_run_all.sh /kaggle/input/btl-cicids2018-sample /kaggle/input/btl-cicids2018-sample/scaler.json 2>&1 | tee results_run.log"]},
  {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": [
   "!zip -qr /kaggle/working/outputs.zip models results reports results_run.log && ls -la /kaggle/working"]}
 ]
}
```

Run: `.venv\Scripts\python -m pytest tests/test_kaggle_runner.py -q` → `2 passed` (test bash cần Git Bash trong PATH; không có thì skip).

- [ ] **Step 4: README + commit**

Thêm vào `README.md`:
````markdown
## Train trên Kaggle (M2)
1. Upload `data/kaggle_sample/` (từ cụm) thành Dataset `btl-cicids2018-sample`; zip repo thành Dataset `btl-bigdeep-code`.
2. Mở `train/kaggle_notebook.ipynb`, bật GPU, chạy lần lượt: smoke → full → zip output.
3. Tải `outputs.zip`, giải nén vào repo (`models/`, `results/`, `reports/`).
Chạy từng phần: `STAGES="ae mlp" bash train/kaggle_run_all.sh DATA SCALER`. Thử local: `SMOKE=1 bash train/kaggle_run_all.sh`.
````

```bash
git add train/kaggle_run_all.sh train/kaggle_notebook.ipynb tests/test_kaggle_runner.py README.md
git commit -m "feat(train): Kaggle runner script + notebook" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: `train/run_ablation.py`

**Files:**
- Create: `train/run_ablation.py`
- Test: `tests/test_run_ablation.py`

**Interfaces:**
- Consumes: `train.train_dl.parse_args`, `train.train_dl.run`; `common.metrics.evaluate`; `common.fake.write_fake_dataset`; file `<base>/scores_DeepAutoencoder.npz`, `<base>/valscores_DeepAutoencoder.npy`, `<base>/dl_ae_metrics.csv` (Task 8).
- Produces: `results/ablation.csv` với cột `ABLATION_COLUMNS = ["experiment","value","precision","recall","f1","fpr","roc_auc","pr_auc","n_params","train_time_s","max_epochs"]` (cột cuối ghi rõ số epoch tối đa). Experiment: `latent` (4,8,16,32), `depth` ("32","64,32","128,64,32"), `threshold` (95,97,99,99.5,99.9 — không train lại), `vae_beta` (0.5,1,2). Chạy lại với `--only X` chỉ thay các dòng của experiment X, giữ dòng khác.

- [ ] **Step 1: Viết test fail**

`tests/test_run_ablation.py`:
```python
import pandas as pd
import pytest

pytest.importorskip("torch")
from train import run_ablation  # noqa: E402

pytestmark = pytest.mark.torch


def test_ablation_smoke(tmp_path):
    csv = tmp_path / "ablation.csv"
    run_ablation.main(["--smoke", "--only", "latent,depth,threshold,vae_beta",
                       "--out-root", str(tmp_path / "runs"), "--out-csv", str(csv),
                       "--base-results-dir", str(tmp_path / "no_base")])
    df = pd.read_csv(csv)
    assert list(df.columns) == run_ablation.ABLATION_COLUMNS
    assert df.groupby("experiment").size().to_dict() == {"latent": 4, "depth": 3, "threshold": 5, "vae_beta": 3}
    thr = df[df.experiment == "threshold"].sort_values("value")
    assert thr["fpr"].is_monotonic_decreasing          # ngưỡng cao hơn -> FPR không tăng
    d = df[df.experiment == "depth"].set_index("value")["n_params"]
    assert d["32"] < d["64,32"] < d["128,64,32"]


def test_only_replaces_selected_experiments(tmp_path):
    csv = tmp_path / "ablation.csv"
    pd.DataFrame([{"experiment": "depth", "value": "x", **{c: 0 for c in run_ablation.ABLATION_COLUMNS[2:]}}]) \
        .to_csv(csv, index=False)
    run_ablation.main(["--smoke", "--only", "threshold", "--out-root", str(tmp_path / "runs"),
                       "--out-csv", str(csv), "--base-results-dir", str(tmp_path / "no_base")])
    df = pd.read_csv(csv)
    assert (df.experiment == "depth").sum() == 1 and (df.experiment == "threshold").sum() == 5
```

Run: `.venv\Scripts\python -m pytest tests/test_run_ablation.py -q` → FAIL (`cannot import name 'run_ablation'`).

- [ ] **Step 2: Viết `train/run_ablation.py`**

```python
"""Ablation cho AE (PROJECT_SPEC 5.3). Mỗi model ablation train tối đa --epochs (mặc định 20,
thấp hơn 50 epoch của model chính) — ghi ở cột max_epochs.

python train/run_ablation.py --data-dir DATA --scaler models/scaler.json --only latent,threshold
Experiments: latent (4,8,16,32) | depth (32 / 64,32 / 128,64,32) | threshold (p95..p99.9, không train lại,
dùng score của AE chính trong --base-results-dir) | vae_beta (0.5,1,2).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as C  # noqa: E402
from common.metrics import evaluate  # noqa: E402
from train import train_dl  # noqa: E402

ABLATION_COLUMNS = ["experiment", "value", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc",
                    "n_params", "train_time_s", "max_epochs"]
LATENTS = [4, 8, 16, 32]
DEPTHS = ["32", "64,32", "128,64,32"]
PERCENTILES = [95, 97, 99, 99.5, 99.9]
BETAS = [0.5, 1.0, 2.0]
EXPERIMENTS = ("latent", "depth", "threshold", "vae_beta")
_CACHE = {}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--scaler", default=None)
    p.add_argument("--only", default="latent,depth,threshold", help="tập con của: " + ",".join(EXPERIMENTS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--base-max-epochs", type=int, default=50, help="max epoch của AE chính (cho dòng threshold)")
    p.add_argument("--test-cap", type=int, default=None)
    p.add_argument("--base-results-dir", default="results")
    p.add_argument("--out-root", default="results/ablation")
    p.add_argument("--out-csv", default="results/ablation.csv")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true", help="dữ liệu giả, 2 epoch, CPU")
    a = p.parse_args(argv)
    cfg = C.load_config(a.config)
    a.data_dir = a.data_dir or C.get(cfg, "data.sample_local_dir")
    a.scaler = a.scaler or C.get(cfg, "models.scaler")
    a.test_cap = a.test_cap or C.get(cfg, "data.test_cap", 300000)
    a.only = [x.strip() for x in a.only.split(",") if x.strip()]
    bad = [x for x in a.only if x not in EXPERIMENTS]
    if bad:
        p.error("experiment không hợp lệ: %s" % bad)
    return a


def train_variant(a, kind: str, hidden: str, latent: int, beta: float = 1.0):
    key = "%s_h%s_l%d_b%g" % (kind, hidden.replace(",", "-"), latent, beta)
    if key not in _CACHE:
        out = Path(a.out_root) / key
        argv = ["--model", kind, "--data-dir", str(a.data_dir), "--scaler", str(a.scaler),
                "--hidden", hidden, "--latent", str(latent), "--beta", str(beta),
                "--epochs", str(a.epochs), "--test-cap", str(a.test_cap),
                "--results-dir", str(out), "--models-dir", str(out), "--no-export"]
        if a.cpu:
            argv.append("--cpu")
        _CACHE[key] = (train_dl.run(train_dl.parse_args(argv)), out)
    return _CACHE[key]


def to_row(experiment, value, m, n_params, train_time_s, max_epochs) -> dict:
    row = {"experiment": experiment, "value": value}
    row.update({k: m[k] for k in ("precision", "recall", "f1", "fpr", "roc_auc", "pr_auc")})
    row.update(n_params=int(n_params), train_time_s=float(train_time_s), max_epochs=int(max_epochs))
    return row


def threshold_rows(a) -> list:
    base = Path(a.base_results_dir)
    need = ["scores_DeepAutoencoder.npz", "valscores_DeepAutoencoder.npy", "dl_ae_metrics.csv"]
    max_ep = a.base_max_epochs
    if not all((base / f).exists() for f in need):
        print("[ablation] thiếu score AE chính trong %s -> train AE mặc định (%d epoch)" % (base, a.epochs))
        _, base = train_variant(a, "ae", "64,32", 16)
        max_ep = a.epochs
    d = np.load(base / "scores_DeepAutoencoder.npz")
    val = np.load(base / "valscores_DeepAutoencoder.npy")
    m0 = pd.read_csv(base / "dl_ae_metrics.csv").iloc[0]
    rows = []
    for p in PERCENTILES:
        ev = evaluate(d["y"], d["score"], float(np.percentile(val, p)))
        rows.append(to_row("threshold", p, ev, m0["n_params"], m0["train_time_s"], max_ep))
    return rows


def run_experiment(a, name: str) -> list:
    if name == "threshold":
        return threshold_rows(a)
    rows = []
    if name == "latent":
        grid = [("ae", "64,32", lat, 1.0, lat) for lat in LATENTS]
    elif name == "depth":
        grid = [("ae", h, 16, 1.0, h) for h in DEPTHS]
    else:
        grid = [("vae", "64,32", 16, b, b) for b in BETAS]
    for kind, hidden, latent, beta, value in grid:
        r, _ = train_variant(a, kind, hidden, latent, beta)
        rows.append(to_row(name, value, r, r["n_params"], r["train_time_s"], a.epochs))
    return rows


def main(argv=None) -> pd.DataFrame:
    a = parse_args(argv)
    if a.smoke:
        from common.fake import write_fake_dataset
        tmp = Path(tempfile.mkdtemp(prefix="ablation_smoke_"))
        write_fake_dataset(tmp / "data", n=4000, scaler_path=tmp / "scaler.json")
        a.data_dir, a.scaler, a.epochs, a.cpu = str(tmp / "data"), str(tmp / "scaler.json"), 2, True
    out = Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=ABLATION_COLUMNS)
    keep = keep[~keep["experiment"].isin(a.only)]
    new_rows = []
    for name in a.only:
        print("[ablation] === %s" % name, flush=True)
        new_rows += run_experiment(a, name)
        df = pd.concat([keep, pd.DataFrame(new_rows, columns=ABLATION_COLUMNS)], ignore_index=True)
        df.to_csv(out, index=False)        # ghi sau mỗi experiment để không mất kết quả nếu Kaggle timeout
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Chạy test, xác nhận pass**

Run: `.venv\Scripts\python -m pytest tests/test_run_ablation.py -q`
Expected: `2 passed`. Nếu `is_monotonic_decreasing` fail do hai percentile cho cùng FPR (dữ liệu giả ít) — đó là bằng nhau, vẫn thoả "không tăng"; pandas coi bằng nhau là monotonic nên không fail.

- [ ] **Step 4: Commit**

```bash
git add train/run_ablation.py tests/test_run_ablation.py
git commit -m "feat(train): AE ablation (latent/depth/threshold/vae beta)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: `analysis/make_report_assets.py` — kết thúc M2b

**REQUIRED:** load skill `dataviz` trước khi viết code vẽ; giữ palette 2 màu DL/ML nhất quán ở mọi hình và đổi màu nếu skill yêu cầu.

**Files:**
- Create: `analysis/__init__.py` (rỗng), `analysis/make_report_assets.py`
- Test: `tests/test_report_assets.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: các file output Task 6/8/10 trong `--results-dir`; `results/holdout/` (holdout.json, dl_mlp_per_attack.csv, baselines_per_attack.csv, baselines_metrics.csv); `common.metrics.METRIC_COLUMNS`.
- Produces trong `--reports-dir`: `model_comparison.csv`, `model_comparison.md`, `per_attack_recall.csv`, `per_attack_recall_heatmap.png`, `roc_pr_unsupervised.png`, `roc_pr_supervised.png`, `score_hist_ae_family.png`, `loss_curves.png`, `vae_recon_kl.png`, `f1_fpr_bar.png`, `ablation_latent.png`, `ablation_depth.png`, `ablation_threshold.png`, `ablation_vae_beta.png`, `holdout_recall.csv`, `holdout_recall.md`. Thiếu input nào thì in `[skip]` và bỏ qua hình đó.

- [ ] **Step 1: Viết test fail**

`tests/test_report_assets.py`:
```python
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("matplotlib")
from analysis import make_report_assets as R  # noqa: E402
from common.metrics import METRIC_COLUMNS  # noqa: E402

UNSUP = ["DeepAutoencoder", "VariationalAE", "DenoisingAE", "PCA"]
SUP = ["DNN_Classifier", "RandomForest"]
KIND = {"DeepAutoencoder": "ae", "VariationalAE": "vae", "DenoisingAE": "dae", "DNN_Classifier": "mlp"}


def _metric_row(name, group, f1):
    return {"model": name, "group": group, "precision": f1, "recall": f1, "f1": f1, "fpr": 0.01,
            "roc_auc": 0.9, "pr_auc": 0.8, "threshold": 0.5, "train_time_s": 1.0, "infer_ms_per_1k": 0.1}


def _per_attack(name):
    return pd.DataFrame({"model": name, "attack": ["Bot", "DDOS attack-HOIC"], "recall": [0.9, 0.4], "n": [10, 20]})


def _fake_results(res):
    rng = np.random.default_rng(0)
    res.mkdir(parents=True)
    pd.DataFrame([_metric_row("PCA", "unsupervised", 0.6), _metric_row("RandomForest", "supervised", 0.95)],
                 columns=METRIC_COLUMNS).to_csv(res / "baselines_metrics.csv", index=False)
    pd.concat([_per_attack("PCA"), _per_attack("RandomForest")]).to_csv(res / "baselines_per_attack.csv", index=False)
    for i, (name, kind) in enumerate(KIND.items()):
        grp = "supervised" if kind == "mlp" else "unsupervised"
        row = dict(_metric_row(name, grp, 0.7 + i * 0.05), n_params=1000 + i)
        pd.DataFrame([row]).to_csv(res / ("dl_%s_metrics.csv" % kind), index=False)
        _per_attack(name).to_csv(res / ("dl_%s_per_attack.csv" % kind), index=False)
        h = pd.DataFrame({"epoch": [1, 2, 3], "train_loss": [1.0, 0.5, 0.4], "val_loss": [1.1, 0.6, 0.5]})
        if kind == "vae":
            h = h.assign(train_recon=[0.9, 0.4, 0.3], train_kl=[0.1, 0.1, 0.1], val_recon=[1.0, 0.5, 0.4], val_kl=[0.1, 0.1, 0.1])
        h.to_csv(res / ("dl_%s_history.csv" % kind), index=False)
    for name in UNSUP + SUP:
        y = rng.integers(0, 2, 500)
        np.savez(res / ("scores_%s.npz" % name), y=y, score=rng.random(500) + y, thr=np.array(1.0))
    rows = []
    for exp, vals in (("latent", [4, 8, 16, 32]), ("depth", ["32", "64,32", "128,64,32"]),
                      ("threshold", [95, 97, 99, 99.5, 99.9]), ("vae_beta", [0.5, 1.0, 2.0])):
        for j, v in enumerate(vals):
            rows.append({"experiment": exp, "value": v, "precision": 0.8, "recall": 0.7, "f1": 0.75,
                         "fpr": 0.05 - j * 0.01, "roc_auc": 0.9, "pr_auc": 0.8, "n_params": 1000 * (j + 1),
                         "train_time_s": 1.0, "max_epochs": 20})
    pd.DataFrame(rows).to_csv(res / "ablation.csv", index=False)
    h = res / "holdout"
    h.mkdir()
    (h / "holdout.json").write_text(json.dumps({"attack": "DDOS attack-HOIC"}))
    _per_attack("DNN_Classifier").to_csv(h / "dl_mlp_per_attack.csv", index=False)
    pd.concat([_per_attack("PCA"), _per_attack("RandomForest")]).to_csv(h / "baselines_per_attack.csv", index=False)
    pd.DataFrame([_metric_row("PCA", "unsupervised", 0.6), _metric_row("RandomForest", "supervised", 0.9)],
                 columns=METRIC_COLUMNS).to_csv(h / "baselines_metrics.csv", index=False)


EXPECTED = ["model_comparison.csv", "model_comparison.md", "per_attack_recall.csv", "per_attack_recall_heatmap.png",
            "roc_pr_unsupervised.png", "roc_pr_supervised.png", "score_hist_ae_family.png", "loss_curves.png",
            "vae_recon_kl.png", "f1_fpr_bar.png", "ablation_latent.png", "ablation_depth.png",
            "ablation_threshold.png", "ablation_vae_beta.png", "holdout_recall.csv", "holdout_recall.md"]


def test_all_assets(tmp_path):
    res, rep = tmp_path / "results", tmp_path / "reports"
    _fake_results(res)
    R.main(["--results-dir", str(res), "--reports-dir", str(rep)])
    missing = [f for f in EXPECTED if not (rep / f).exists()]
    assert missing == []
    comp = pd.read_csv(rep / "model_comparison.csv")
    assert len(comp) == 6 and set(comp["family"]) == {"DL", "ML"}
    assert comp["group"].tolist()[:4] == ["unsupervised"] * 4
    md = (rep / "model_comparison.md").read_text(encoding="utf-8")
    assert "| **DeepAutoencoder** |" in md and "0.0100" in md
    ho = pd.read_csv(rep / "holdout_recall.csv")
    assert set(ho["model"]) == {"DeepAutoencoder", "VariationalAE", "DenoisingAE", "DNN_Classifier", "RandomForest"}


def test_missing_optional_inputs_are_skipped(tmp_path):
    res, rep = tmp_path / "results", tmp_path / "reports"
    _fake_results(res)
    (res / "ablation.csv").unlink()
    R.main(["--results-dir", str(res), "--reports-dir", str(rep)])
    assert (rep / "model_comparison.md").exists() and not (rep / "ablation_latent.png").exists()
```

Run: `.venv\Scripts\python -m pytest tests/test_report_assets.py -q` → FAIL (`No module named 'analysis.make_report_assets'`).

- [ ] **Step 2: Viết `analysis/make_report_assets.py`**

```python
"""Gộp metric + vẽ hình cho báo cáo.  python analysis/make_report_assets.py [--results-dir results --reports-dir reports]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DL_KINDS = {"ae": "DeepAutoencoder", "vae": "VariationalAE", "dae": "DenoisingAE", "mlp": "DNN_Classifier"}
DL_NAMES = set(DL_KINDS.values())
AE_FAMILY = ["DeepAutoencoder", "VariationalAE", "DenoisingAE"]
MAIN_MODEL = "DeepAutoencoder"
FAMILY_COLORS = {"DL": "#2563eb", "ML": "#9ca3af"}
DPI = 150
TABLE_COLS = ["model", "family", "group", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc",
              "threshold", "train_time_s", "infer_ms_per_1k", "n_params"]


def skip(msg):
    print("[skip] " + msg)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("[ok] " + str(path))


def load_comparison(res: Path) -> pd.DataFrame:
    frames = []
    if (res / "baselines_metrics.csv").exists():
        frames.append(pd.read_csv(res / "baselines_metrics.csv"))
    frames += [pd.read_csv(p) for p in sorted(res.glob("dl_*_metrics.csv"))]
    if not frames:
        raise FileNotFoundError("không có file metric trong %s" % res)
    df = pd.concat(frames, ignore_index=True, sort=False)
    df.insert(1, "family", np.where(df["model"].isin(DL_NAMES), "DL", "ML"))
    return df.sort_values(["group", "f1"], ascending=[False, False]).reset_index(drop=True)


def fmt(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return "%d" % v if float(v).is_integer() and abs(v) >= 1000 else "%.4f" % v
    return str(v)


def to_markdown(df: pd.DataFrame, cols, bold_model=MAIN_MODEL) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [fmt(r[c]) for c in cols]
        if r.get("model") == bold_model:
            cells = ["**%s**" % c if c else c for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_comparison(df, rep: Path):
    cols = [c for c in TABLE_COLS if c in df.columns]
    df[cols].to_csv(rep / "model_comparison.csv", index=False)
    (rep / "model_comparison.md").write_text(to_markdown(df, cols), encoding="utf-8")


def read_per_attack(res: Path) -> pd.DataFrame:
    files = [res / "baselines_per_attack.csv"] + sorted(res.glob("dl_*_per_attack.csv"))
    frames = [pd.read_csv(f) for f in files if f.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["model", "attack", "recall", "n"])


def per_attack_table(res: Path, rep: Path):
    pa = read_per_attack(res)
    if pa.empty:
        return skip("per-attack")
    piv = pa.pivot_table(index="model", columns="attack", values="recall")
    piv.to_csv(rep / "per_attack_recall.csv")
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 3, 0.5 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=8,
                        color="white" if v > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Recall")
    ax.set_title("Recall theo loại tấn công")
    save(fig, rep / "per_attack_recall_heatmap.png")


def load_scores(res: Path, name: str):
    p = res / ("scores_%s.npz" % name)
    return np.load(p) if p.exists() else None


def plot_roc_pr(res: Path, models, title, path):
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    avail = [m for m in models if load_scores(res, m) is not None]
    if not avail:
        return skip(title)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name in avail:
        d = load_scores(res, name)
        y, s = d["y"], d["score"]
        lw = 2.2 if name == MAIN_MODEL else 1.2
        fpr, tpr, _ = roc_curve(y, s)
        a1.plot(fpr, tpr, lw=lw, label="%s (AUC %.3f)" % (name, roc_auc_score(y, s)))
        p, r, _ = precision_recall_curve(y, s)
        a2.plot(r, p, lw=lw, label="%s (AP %.3f)" % (name, average_precision_score(y, s)))
    a1.plot([0, 1], [0, 1], ls=":", color="grey")
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    a2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    a1.legend(fontsize=8)
    a2.legend(fontsize=8)
    fig.suptitle(title)
    save(fig, path)


def plot_score_hist(res: Path, path):
    names = [n for n in AE_FAMILY if load_scores(res, n) is not None]
    if not names:
        return skip("score histogram")
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), squeeze=False)
    for ax, name in zip(axes[0], names):
        d = load_scores(res, name)
        y, s = d["y"], np.clip(d["score"].astype(float), 1e-8, None)
        bins = np.logspace(np.log10(s.min()), np.log10(s.max()) + 1e-9, 80)
        ax.hist(s[y == 0], bins=bins, alpha=0.6, density=True, label="Benign")
        ax.hist(s[y == 1], bins=bins, alpha=0.6, density=True, label="Attack")
        ax.axvline(float(d["thr"]), color="red", ls="--", label="ngưỡng")
        ax.set_xscale("log")
        ax.set(title=name, xlabel="reconstruction error (log)", ylabel="mật độ")
        ax.legend(fontsize=8)
    save(fig, path)


def plot_loss_curves(res: Path, path, vae_path):
    hist = {k: res / ("dl_%s_history.csv" % k) for k in DL_KINDS}
    if not any(p.exists() for p in hist.values()):
        return skip("loss curves")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (kind, p) in zip(axes.flat, hist.items()):
        if not p.exists():
            ax.axis("off")
            continue
        h = pd.read_csv(p)
        ax.plot(h["epoch"], h["train_loss"], label="train")
        ax.plot(h["epoch"], h["val_loss"], label="val")
        ax.set(title=DL_KINDS[kind], xlabel="epoch", ylabel="loss")
        ax.legend(fontsize=8)
    save(fig, path)
    if hist["vae"].exists():
        h = pd.read_csv(hist["vae"])
        if {"train_recon", "train_kl"} <= set(h.columns):
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
            a1.plot(h["epoch"], h["train_recon"], label="train")
            a1.plot(h["epoch"], h["val_recon"], label="val")
            a1.set(title="VAE reconstruction loss", xlabel="epoch")
            a2.plot(h["epoch"], h["train_kl"], label="train")
            a2.plot(h["epoch"], h["val_kl"], label="val")
            a2.set(title="VAE KL", xlabel="epoch")
            a1.legend(fontsize=8)
            a2.legend(fontsize=8)
            save(fig, vae_path)


def plot_f1_fpr(df: pd.DataFrame, path):
    order = df.sort_values("f1")
    colors = [FAMILY_COLORS[f] for f in order["family"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.45 * len(order) + 1.8), sharey=True)
    a1.barh(order["model"], order["f1"], color=colors)
    a1.set(xlabel="F1", xlim=(0, 1))
    a2.barh(order["model"], order["fpr"], color=colors)
    a2.set(xlabel="FPR")
    a1.legend(handles=[Patch(color=c, label=k) for k, c in FAMILY_COLORS.items()], loc="lower right")
    save(fig, path)


def plot_ablation(res: Path, rep: Path):
    p = res / "ablation.csv"
    if not p.exists():
        return skip("ablation")
    ab = pd.read_csv(p)

    def f1_fpr(sub, xlabels, title, xlabel, path):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        x = np.arange(len(sub))
        ax.plot(x, sub["f1"], marker="o", label="F1")
        ax.set(xticks=x, xlabel=xlabel, ylabel="F1", title=title)
        ax.set_xticklabels(xlabels)
        ax2 = ax.twinx()
        ax2.plot(x, sub["fpr"], marker="s", color="tab:red", label="FPR")
        ax2.set_ylabel("FPR")
        fig.legend(loc="upper right", fontsize=8)
        save(fig, path)

    lat = ab[ab["experiment"] == "latent"].assign(v=lambda d: d["value"].astype(float)).sort_values("v")
    if len(lat):
        f1_fpr(lat, ["%d" % v for v in lat["v"]], "AE: kích thước latent", "latent", rep / "ablation_latent.png")
    dep = ab[ab["experiment"] == "depth"].sort_values("n_params")
    if len(dep):
        f1_fpr(dep, ["%s\n(%s tham số)" % (v, format(int(n), ",")) for v, n in zip(dep["value"], dep["n_params"])],
               "AE: độ sâu", "hidden", rep / "ablation_depth.png")
    beta = ab[ab["experiment"] == "vae_beta"].assign(v=lambda d: d["value"].astype(float)).sort_values("v")
    if len(beta):
        f1_fpr(beta, ["%g" % v for v in beta["v"]], "VAE: β", "β", rep / "ablation_vae_beta.png")
    thr = ab[ab["experiment"] == "threshold"].assign(v=lambda d: d["value"].astype(float)).sort_values("v")
    if len(thr):
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for col in ("precision", "recall", "fpr"):
            ax.plot(thr["v"], thr[col], marker="o", label=col)
        ax.set(xlabel="percentile ngưỡng (val Benign)", title="AE: chọn ngưỡng")
        ax.legend(fontsize=8)
        save(fig, rep / "ablation_threshold.png")


def holdout_table(res: Path, rep: Path):
    hdir = res / "holdout"
    if not (hdir / "holdout.json").exists():
        return skip("holdout")
    attack = json.loads((hdir / "holdout.json").read_text(encoding="utf-8"))["attack"]
    main_pa = read_per_attack(res)
    rows = []
    for name in AE_FAMILY:
        r = main_pa[(main_pa["model"] == name) & (main_pa["attack"] == attack)]
        if len(r):
            rows.append({"model": name, "setting": "unsupervised (train chỉ Benign)", "attack": attack,
                         "recall": float(r["recall"].iloc[0]), "n": int(r["n"].iloc[0])})
    sup_models = {"DNN_Classifier"}
    if (hdir / "baselines_metrics.csv").exists():
        bm = pd.read_csv(hdir / "baselines_metrics.csv")
        sup_models |= set(bm.loc[bm["group"] == "supervised", "model"])
    hpa = read_per_attack(hdir)
    for _, r in hpa[(hpa["attack"] == attack) & hpa["model"].isin(sup_models)].iterrows():
        rows.append({"model": r["model"], "setting": "supervised, holdout khỏi train", "attack": attack,
                     "recall": float(r["recall"]), "n": int(r["n"])})
    df = pd.DataFrame(rows, columns=["model", "setting", "attack", "recall", "n"])
    df.to_csv(rep / "holdout_recall.csv", index=False)
    (rep / "holdout_recall.md").write_text(to_markdown(df, list(df.columns)), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--reports-dir", default="reports")
    a = p.parse_args(argv)
    res, rep = Path(a.results_dir), Path(a.reports_dir)
    rep.mkdir(parents=True, exist_ok=True)
    comp = load_comparison(res)
    write_comparison(comp, rep)
    per_attack_table(res, rep)
    groups = comp.groupby("group")["model"].apply(list).to_dict()
    plot_roc_pr(res, groups.get("unsupervised", []), "Nhóm unsupervised", rep / "roc_pr_unsupervised.png")
    plot_roc_pr(res, groups.get("supervised", []), "Nhóm supervised", rep / "roc_pr_supervised.png")
    plot_score_hist(res, rep / "score_hist_ae_family.png")
    plot_loss_curves(res, rep / "loss_curves.png", rep / "vae_recon_kl.png")
    plot_f1_fpr(comp, rep / "f1_fpr_bar.png")
    plot_ablation(res, rep)
    holdout_table(res, rep)
    print(to_markdown(comp, [c for c in TABLE_COLS if c in comp.columns]))


if __name__ == "__main__":
    main()
```

Ghi chú `fmt`: số nguyên lớn (n_params) in không phần thập phân; còn lại 4 chữ số thập phân.

- [ ] **Step 3: Chạy test, xác nhận pass**

Run: `.venv\Scripts\python -m pytest tests/test_report_assets.py -q`
Expected: `2 passed`.

- [ ] **Step 4: Xem hình bằng mắt**

Mở vài PNG trong `tmp` (hoặc chạy lại `R.main` vào `.cache/rep_check`) bằng Read tool: nhãn trục đọc được, legend không che dữ liệu, heatmap có số.

- [ ] **Step 5: Chạy toàn bộ pipeline Kaggle ở chế độ smoke**

Run (Git Bash): `PY=.venv/Scripts/python SMOKE=1 bash train/kaggle_run_all.sh`
Expected: kết thúc bằng `=== ... DONE`; `.cache/smoke/reports/` có đủ 16 file của danh sách EXPECTED (trừ khi baseline holdout không xuất `baselines_metrics.csv` — khi đó vẫn có holdout của DNN_Classifier).

- [ ] **Step 6: README + commit**

Thêm vào `README.md`:
````markdown
## Bảng & hình cho báo cáo
```bash
python analysis/make_report_assets.py --results-dir results --reports-dir reports
```
Kết quả: `reports/model_comparison.md` (10 model), `reports/per_attack_recall.csv`, các hình `reports/*.png`, `reports/holdout_recall.md`.
````

```bash
git add analysis/ tests/test_report_assets.py README.md
git commit -m "feat(analysis): report tables and figures (M2b)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Báo người dùng: chạy notebook trên Kaggle (Task 9 README), tải `outputs.zip` về.

---

### Task 11b: `analysis/eda.py --latent` — không gian latent + silhouette (M2b)

**Files:**
- Modify: `analysis/eda.py` (thay stub `latent_space`)
- Test: `tests/test_eda.py` (thêm test)

**Interfaces:**
- Consumes: `train.dl_models.load_checkpoint`, `train.dl_models.encode_latent` (Task 7), `models/ae.pt`, `models/vae.pt` (Task 8); `cache["pca_raw"]`, `cache["tsne_raw"]` do `projections` của Task 5b điền khi chạy cùng phần 1.
- Produces: `latent_pca.png`, `latent_tsne.png` (figure 1×3: dữ liệu gốc | latent AE | latent VAE μ, cùng mẫu phân tầng), `silhouette.csv` (cột `representation, dim, silhouette`; dòng `raw`, `AE latent`, `VAE latent (mu)`), và bảng silhouette trong `eda_summary.md`.

- [ ] **Step 1: Viết test fail**

Thêm vào `tests/test_eda.py`:
```python
def test_latent_smoke(tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "eda"
    eda.main(["--smoke", "--latent", "--out", str(out), "--per-class", "150"])
    for f in ("latent_pca.png", "latent_tsne.png", "silhouette.csv"):
        assert (out / f).exists(), f
    sil = pd.read_csv(out / "silhouette.csv")
    assert list(sil.columns) == ["representation", "dim", "silhouette"]
    assert sil["representation"].tolist() == ["raw", "AE latent", "VAE latent (mu)"]
    assert sil["silhouette"].between(-1, 1).all()
    assert "Silhouette" in (out / "eda_summary.md").read_text(encoding="utf-8")


def test_latent_only_keeps_part1_descriptions(tmp_path):
    pytest.importorskip("torch")
    out = tmp_path / "eda"
    eda.main(["--smoke", "--out", str(out), "--per-class", "100", "--skip-tsne"])
    eda.main(["--smoke", "--latent", "--no-part1", "--out", str(out), "--per-class", "100", "--skip-tsne"])
    summary = (out / "eda_summary.md").read_text(encoding="utf-8")
    assert "`projection_pca.png`" in summary and "`latent_pca.png`" in summary
```

Run: `.venv\Scripts\python -m pytest tests/test_eda.py -q` → 2 test mới FAIL (`SystemExit: --latent cần Task 11b`).

- [ ] **Step 2: Thay stub `latent_space` trong `analysis/eda.py`**

```python
def latent_space(X, labels, models_dir: Path, out: Path, skip_tsne: bool, perplexity: float, cache: dict):
    """Mục 7-8: latent AE (encoder) và VAE (μ) trên cùng mẫu với hình chiếu dữ liệu gốc."""
    import torch
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    from train.dl_models import encode_latent, load_checkpoint
    panels = [("raw", "Dữ liệu gốc (chuẩn hoá)", X)]
    for kind, key, title in (("ae", "AE latent", "Latent AE"), ("vae", "VAE latent (mu)", "Latent VAE (μ)")):
        p = models_dir / ("%s.pt" % kind)
        if not p.exists():
            log("[skip] %s: không có %s" % (key, p))
            continue
        _, model = load_checkpoint(p)
        with torch.no_grad():
            Z = encode_latent(model, torch.from_numpy(np.ascontiguousarray(X))).numpy()
        panels.append((key, title, Z))
    if len(panels) == 1:
        return log("[skip] latent: không có ae.pt / vae.pt trong %s" % models_dir)

    def figure(proj_key, project, name, method):
        fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 5), squeeze=False)
        for ax, (key, title, M) in zip(axes[0], panels):
            P = cache[proj_key] if key == "raw" and proj_key in cache else project(M)
            scatter_by_label(ax, P, labels, "%s — %s (%d chiều)" % (title, method, M.shape[1]))
        axes[0, -1].legend(markerscale=3, fontsize=7, loc="best")
        save(fig, out / name)

    figure("pca_raw", lambda M: PCA(n_components=2, random_state=SEED).fit_transform(M), "latent_pca.png", "PCA")
    if not skip_tsne:
        figure("tsne_raw", lambda M: tsne_2d(M, perplexity), "latent_tsne.png", "t-SNE")
    y = binarize(labels)
    if len(np.unique(y)) < 2:
        return log("[skip] silhouette: mẫu chỉ có một lớp")
    rows = [{"representation": key, "dim": int(M.shape[1]),
             "silhouette": float(silhouette_score(M, y, sample_size=min(5000, len(M)), random_state=SEED))}
            for key, _, M in panels]
    pd.DataFrame(rows).to_csv(out / "silhouette.csv", index=False)
    log("wrote silhouette.csv")
```

- [ ] **Step 3: Chạy test, xác nhận pass**

Run: `.venv\Scripts\python -m pytest tests/test_eda.py -q`
Expected: `5 passed`.

- [ ] **Step 4: Xem hình bằng mắt**

Run: `.venv\Scripts\python analysis/eda.py --smoke --latent --out .cache/eda_check --per-class 300`, mở `latent_pca.png` và `latent_tsne.png` bằng Read tool: 3 panel cùng cỡ, tiêu đề ghi số chiều, legend ở panel cuối không che dữ liệu.

- [ ] **Step 5: Commit**

```bash
git add analysis/eda.py tests/test_eda.py
git commit -m "feat(analysis): EDA part 2 - AE/VAE latent space projections + silhouette (M2b)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: `streaming/producer.py`

**Files:**
- Create: `streaming/producer.py`
- Test: `tests/test_producer.py`

**Interfaces:**
- Consumes: `common.config`, `common.data` (`ID_COL`, `LABEL_COL`, `TS_COL`, `load_scaler`), `common.fake.write_fake_dataset` (test).
- Produces: `parse_args`, `RateLimiter(rate, clock=time.monotonic, sleep=time.sleep).wait(n_sent)`, `load_frame(path, features, limit=0) -> DataFrame`, `iter_messages(df, features, id_offset=0)`, `ensure_topic(bootstrap, topic, partitions)`, `run(args) -> dict`, `main(argv=None) -> dict`. Stats dict: `{"sent","duration_s","actual_rate","target_rate","topic"}`, ghi ra `--stats-out` nếu có. Message theo PROJECT_SPEC 5.5, `features` chỉ gồm các feature trong `scaler.json`.

- [ ] **Step 1: Viết test fail**

`tests/test_producer.py`:
```python
import json
import os
import uuid

import pytest

from common.data import load_scaler
from common.fake import FAKE_FEATURES, write_fake_dataset
from streaming.producer import RateLimiter, iter_messages, load_frame, main


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.t += d


def test_rate_limiter_paces():
    c = FakeClock()
    lim = RateLimiter(100, clock=c, sleep=c.sleep)
    lim.wait(0)
    lim.wait(50)
    assert c.t == pytest.approx(0.5)
    lim.wait(10)                      # đang trễ hơn lịch -> không ngủ
    assert c.t == pytest.approx(0.5)


def test_rate_limiter_unlimited():
    c = FakeClock()
    lim = RateLimiter(0, clock=c, sleep=c.sleep)
    lim.wait(10 ** 6)
    assert c.t == 0.0


def test_load_frame_and_messages(tmp_path):
    write_fake_dataset(tmp_path, n=2000, scaler_path=tmp_path / "scaler.json")
    feats = load_scaler(tmp_path / "scaler.json").features
    df = load_frame(tmp_path / "test", feats, limit=5)
    assert len(df) == 5 and df["flow_id"].is_monotonic_increasing
    msgs = list(iter_messages(df, feats, id_offset=1000))
    m = msgs[0]
    assert set(m) == {"flow_id", "orig_ts", "label", "features"}
    assert m["flow_id"] == int(df["flow_id"].iloc[0]) + 1000
    assert list(m["features"]) == FAKE_FEATURES
    json.dumps(m)                     # serialize được


@pytest.mark.kafka
def test_produce_to_kafka(tmp_path):
    boot = os.environ.get("KAFKA_BOOTSTRAP")
    if not boot:
        pytest.skip("KAFKA_BOOTSTRAP chưa đặt")
    from kafka import KafkaConsumer
    write_fake_dataset(tmp_path, n=2000, scaler_path=tmp_path / "scaler.json")
    topic = "test-producer-" + uuid.uuid4().hex[:8]
    stats = main(["--bootstrap", boot, "--topic", topic, "--create-topic", "--input", str(tmp_path / "test"),
                  "--scaler", str(tmp_path / "scaler.json"), "--rate", "0", "--limit", "200",
                  "--stats-out", str(tmp_path / "p.json")])
    assert stats["sent"] == 200 and json.loads((tmp_path / "p.json").read_text())["sent"] == 200
    c = KafkaConsumer(topic, bootstrap_servers=boot, auto_offset_reset="earliest", consumer_timeout_ms=10000,
                      value_deserializer=lambda b: json.loads(b))
    got = [r.value for r in c]
    c.close()
    assert len(got) == 200 and all(isinstance(g["produce_ts"], int) for g in got)
```

Run: `docker compose -f docker/docker-compose.test.yml run --rm cluster pytest tests/test_producer.py -q`
Expected: FAIL (`No module named 'streaming.producer'`).

- [ ] **Step 2: Viết `streaming/producer.py`**

```python
"""Replay Parquet test -> Kafka, mỗi flow một message JSON (PROJECT_SPEC 5.5).

python3 streaming/producer.py --rate 2000 --limit 100000
python3 streaming/producer.py --input hdfs:///.../processed/test --rate 0 --loop
'label' chỉ dùng để đánh giá; model không đọc nó.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as C  # noqa: E402
from common.data import ID_COL, LABEL_COL, TS_COL, load_scaler  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--bootstrap", default=None, help="mặc định cluster.kafka_bootstrap")
    p.add_argument("--topic", default=None, help="mặc định kafka.topic")
    p.add_argument("--input", default=None, help="Parquet test (thư mục cục bộ hoặc hdfs://); mặc định <data.local_dir>/test")
    p.add_argument("--scaler", default=None, help="danh sách feature gửi đi; mặc định models.scaler")
    p.add_argument("--rate", type=float, default=1000.0, help="message/giây; 0 = không giới hạn")
    p.add_argument("--limit", type=int, default=0, help="số flow tối đa mỗi vòng; 0 = tất cả")
    p.add_argument("--loop", action="store_true", help="phát lại liên tục (flow_id cộng offset mỗi vòng)")
    p.add_argument("--create-topic", action="store_true")
    p.add_argument("--partitions", type=int, default=None)
    p.add_argument("--linger-ms", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=262144)
    p.add_argument("--report-every", type=float, default=5.0)
    p.add_argument("--stats-out", default=None)
    a = p.parse_args(argv)
    cfg = C.load_config(a.config)
    a.bootstrap = a.bootstrap or C.get(cfg, "cluster.kafka_bootstrap")
    a.topic = a.topic or C.get(cfg, "kafka.topic")
    a.input = a.input or str(Path(C.get(cfg, "data.local_dir")) / "test")
    a.scaler = a.scaler or C.get(cfg, "models.scaler")
    a.partitions = a.partitions or C.get(cfg, "kafka.partitions", 3)
    return a


class RateLimiter:
    """Message thứ n được gửi không sớm hơn t0 + n/rate."""

    def __init__(self, rate: float, clock=time.monotonic, sleep=time.sleep):
        self.rate, self.clock, self.sleep, self.t0 = rate, clock, sleep, None

    def wait(self, n_sent: int) -> None:
        if self.rate <= 0:
            return
        if self.t0 is None:
            self.t0 = self.clock()
        delay = self.t0 + n_sent / self.rate - self.clock()
        if delay > 0:
            self.sleep(delay)


def fetch_hdfs(path: str) -> Path:
    dest = ROOT / "data" / "cache" / path.rstrip("/").split("/")[-1]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["hdfs", "dfs", "-get", path, str(dest)], check=True)
    return dest


def load_frame(path, features, limit: int = 0) -> pd.DataFrame:
    local = fetch_hdfs(str(path)) if str(path).startswith("hdfs://") else Path(path)
    df = pd.read_parquet(local, columns=[ID_COL, TS_COL, LABEL_COL] + list(features))
    df = df.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
    return df.head(limit) if limit else df


def iter_messages(df: pd.DataFrame, features, id_offset: int = 0):
    features = list(features)
    vals = df[features].to_numpy(dtype=np.float64).tolist()
    ids = df[ID_COL].astype("int64").tolist()
    ts = df[TS_COL].astype(str).tolist()
    labels = df[LABEL_COL].astype(str).tolist()
    for i in range(len(ids)):
        yield {"flow_id": ids[i] + id_offset, "orig_ts": ts[i], "label": labels[i],
               "features": dict(zip(features, vals[i]))}


def ensure_topic(bootstrap: str, topic: str, partitions: int) -> None:
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError
    admin = KafkaAdminClient(bootstrap_servers=bootstrap)
    try:
        admin.create_topics([NewTopic(name=topic, num_partitions=partitions, replication_factor=1)])
    except TopicAlreadyExistsError:
        pass
    finally:
        admin.close()


def run(a) -> dict:
    from kafka import KafkaProducer
    features = load_scaler(a.scaler).features
    df = load_frame(a.input, features, a.limit)
    if a.create_topic:
        ensure_topic(a.bootstrap, a.topic, a.partitions)
    prod = KafkaProducer(bootstrap_servers=a.bootstrap, linger_ms=a.linger_ms, batch_size=a.batch_size, acks=1,
                         value_serializer=lambda m: json.dumps(m, separators=(",", ":")).encode("utf-8"))
    lim = RateLimiter(a.rate)
    span = int(df[ID_COL].max()) + 1 if len(df) else 0
    sent, loop = 0, 0
    t0 = last_t = time.time()
    last_n = 0
    print("[producer] %d flow -> %s/%s, rate=%s" % (len(df), a.bootstrap, a.topic, a.rate or "max"), flush=True)
    try:
        while True:
            for msg in iter_messages(df, features, id_offset=loop * span):
                if sent % 50 == 0:
                    lim.wait(sent)
                msg["produce_ts"] = int(time.time() * 1000)
                prod.send(a.topic, value=msg)
                sent += 1
                now = time.time()
                if now - last_t >= a.report_every:
                    print("[producer] sent %d, rate %.0f msg/s" % (sent, (sent - last_n) / (now - last_t)), flush=True)
                    last_t, last_n = now, sent
            loop += 1
            if not a.loop:
                break
    except KeyboardInterrupt:
        print("[producer] dừng theo yêu cầu", flush=True)
    finally:
        prod.flush()
        prod.close()
    dur = time.time() - t0
    stats = {"sent": sent, "duration_s": round(dur, 3), "actual_rate": sent / dur if dur > 0 else 0.0,
             "target_rate": a.rate, "topic": a.topic}
    print("[producer] done: %s" % json.dumps(stats), flush=True)
    if a.stats_out:
        Path(a.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.stats_out).write_text(json.dumps(stats), encoding="utf-8")
    return stats


def main(argv=None) -> dict:
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Chạy test, xác nhận pass**

Run: `docker compose -f docker/docker-compose.test.yml run --rm cluster pytest tests/test_producer.py -q`
Expected: `4 passed` (Kafka container tự khởi động qua `depends_on`; nếu test kafka fail vì broker chưa sẵn sàng, chạy lại sau 10 giây).

Run: `.venv\Scripts\python -m pytest tests/test_producer.py -q` → `3 passed, 1 skipped` (`kafka` chỉ được import bên trong `run`/`ensure_topic`, nên unit test không cần kafka-python).

- [ ] **Step 4: Commit**

```bash
git add streaming/producer.py tests/test_producer.py
git commit -m "feat(streaming): Kafka producer replaying test parquet with rate control" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: `streaming/scorer.py`, `streaming/output_io.py`, `streaming/stream_detect.py`

**Files:**
- Create: `streaming/scorer.py`, `streaming/output_io.py`, `streaming/stream_detect.py`, `tests/onnx_fixture.py`
- Test: `tests/test_scorer.py`, `tests/test_stream_detect.py`

**Interfaces:**
- Consumes: `common.data` (`load_scaler`, `transform`), `common.config`.
- Produces:
  - `streaming.scorer`: `OnnxScorer(model_path, scaler_path, threshold_path)` có `.threshold`, `.scaler`, `.score_matrix(raw: np.ndarray[n,d] float64 theo thứ tự scaler.features) -> np.ndarray[float64]`; `get_scorer(...)` (cache theo process — mỗi Python worker load model 1 lần); `PASSTHROUGH_COLS`; `score_frame(pdf, scorer, feature_cols) -> DataFrame[PASSTHROUGH_COLS + recon_error, is_anomaly]`.
  - `streaming.output_io`: `OUTPUT_COLUMNS`, `load_output(out_dir, max_files=None)`.
  - `streaming.stream_detect`: `parse_args`, `parse_stream(raw_df, features)` (cột `flow_id, orig_ts, label, produce_ts, kafka_ts, f0..f{d-1}`), `BatchHandler(args, n_features)` (callable `(batch_df, batch_id)`; thuộc tính `rows_seen`, `last_data_ms`), `main(argv) -> {"rows": int}`. File ra: `<out-dir>/batch-%08d.parquet` (ghi `.tmp` rồi `os.replace` — atomic với dashboard), `<out-dir>/_STARTED`, `--batches-csv`.
  - `tests/onnx_fixture.make_mean_square_onnx(path, d)`: ONNX opset 17 với `score = mean(input², axis=1)` (= AE tái tạo bằng 0) để test cụm không cần torch.

- [ ] **Step 1: Viết fixture + test fail cho scorer**

`tests/onnx_fixture.py`:
```python
"""ONNX nhỏ cho test phía cụm (không cần torch): score = mean(input^2, axis=1)."""
from __future__ import annotations


def make_mean_square_onnx(path, d: int) -> None:
    import onnx
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", d])
    y = helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch"])
    nodes = [helper.make_node("Mul", ["input", "input"], ["sq"]),
             helper.make_node("ReduceMean", ["sq"], ["score"], axes=[1], keepdims=0)]
    model = helper.make_model(helper.make_graph(nodes, "mean_square", [x], [y]),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
```

`tests/test_scorer.py`:
```python
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from common.data import load_scaler, transform  # noqa: E402
from common.fake import write_fake_dataset  # noqa: E402
from streaming.output_io import OUTPUT_COLUMNS, load_output  # noqa: E402
from streaming.scorer import PASSTHROUGH_COLS, OnnxScorer, get_scorer, score_frame  # noqa: E402
from tests.onnx_fixture import make_mean_square_onnx  # noqa: E402


@pytest.fixture()
def assets(tmp_path):
    scaler = write_fake_dataset(tmp_path / "data", n=1000, scaler_path=tmp_path / "scaler.json")
    make_mean_square_onnx(tmp_path / "m.onnx", len(scaler.features))
    (tmp_path / "thr.json").write_text(json.dumps({"percentile": 99, "threshold": 1.0}))
    return tmp_path, scaler


def test_score_matrix_matches_numpy(assets):
    d, scaler = assets
    s = OnnxScorer(d / "m.onnx", d / "scaler.json", d / "thr.json")
    raw = pd.read_parquet(d / "data" / "test")[scaler.features].to_numpy(np.float64)
    expected = (transform(raw, load_scaler(d / "scaler.json")).astype(np.float64) ** 2).mean(axis=1)
    np.testing.assert_allclose(s.score_matrix(raw), expected, rtol=1e-5)
    assert s.threshold == 1.0


def test_get_scorer_is_cached(assets):
    d, _ = assets
    a = get_scorer(str(d / "m.onnx"), str(d / "scaler.json"), str(d / "thr.json"))
    b = get_scorer(str(d / "m.onnx"), str(d / "scaler.json"), str(d / "thr.json"))
    assert a is b


def test_score_frame_columns_and_flags(assets):
    d, scaler = assets
    s = get_scorer(str(d / "m.onnx"), str(d / "scaler.json"), str(d / "thr.json"))
    df = pd.read_parquet(d / "data" / "test").head(50)
    pdf = pd.DataFrame({"flow_id": df["flow_id"], "orig_ts": df["orig_ts"], "label": df["Label"],
                        "produce_ts": 1, "kafka_ts": 2})
    for i, f in enumerate(scaler.features):
        pdf["f%d" % i] = df[f].to_numpy()
    out = score_frame(pdf, s, ["f%d" % i for i in range(len(scaler.features))])
    assert list(out.columns) == PASSTHROUGH_COLS + ["recon_error", "is_anomaly"]
    assert (out["is_anomaly"] == (out["recon_error"] >= 1.0)).all()
    empty = score_frame(pdf.head(0), s, ["f%d" % i for i in range(len(scaler.features))])
    assert len(empty) == 0 and list(empty.columns) == list(out.columns)


def test_load_output_ignores_tmp(tmp_path):
    row = {c: [1] for c in OUTPUT_COLUMNS}
    pd.DataFrame(row).to_parquet(tmp_path / "batch-00000001.parquet")
    (tmp_path / "batch-00000002.parquet.tmp").write_bytes(b"partial")
    assert len(load_output(tmp_path)) == 1
    assert list(load_output(tmp_path / "missing").columns) == OUTPUT_COLUMNS
```

Run: `.venv\Scripts\python -m pytest tests/test_scorer.py -q` → FAIL (`No module named 'streaming.output_io'`).

- [ ] **Step 2: Viết `streaming/output_io.py` và `streaming/scorer.py`**

`streaming/output_io.py`:
```python
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
```

`streaming/scorer.py`:
```python
"""Chấm điểm ONNX cho streaming. Graph ONNX trả thẳng score (MSE tái tạo với AE/VAE/DAE, xác suất với MLP)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from common.data import load_scaler, transform

PASSTHROUGH_COLS = ["flow_id", "orig_ts", "label", "produce_ts", "kafka_ts"]
_CACHE = {}


class OnnxScorer:
    def __init__(self, model_path, scaler_path, threshold_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.scaler = load_scaler(scaler_path)
        with open(threshold_path, encoding="utf-8") as f:
            self.threshold = float(json.load(f)["threshold"])

    def score_matrix(self, raw) -> np.ndarray:
        """raw: [n, d] theo thứ tự scaler.features. Feature thiếu (NaN) -> 0 sau chuẩn hoá (= giá trị trung bình)."""
        Z = np.nan_to_num(transform(raw, self.scaler), nan=0.0)
        return self.sess.run(["score"], {"input": Z})[0].reshape(-1).astype(np.float64)


def get_scorer(model_path, scaler_path, threshold_path) -> OnnxScorer:
    key = (str(model_path), str(scaler_path), str(threshold_path))
    if key not in _CACHE:
        _CACHE[key] = OnnxScorer(*key)
    return _CACHE[key]


def score_frame(pdf: pd.DataFrame, scorer: OnnxScorer, feature_cols) -> pd.DataFrame:
    out = pdf[PASSTHROUGH_COLS].copy()
    if len(pdf) == 0:
        out["recon_error"] = pd.Series(dtype="float64")
        out["is_anomaly"] = pd.Series(dtype="bool")
        return out
    err = scorer.score_matrix(pdf[list(feature_cols)].to_numpy(dtype=np.float64))
    out["recon_error"] = err
    out["is_anomaly"] = err >= scorer.threshold
    return out
```

Run: `.venv\Scripts\python -m pytest tests/test_scorer.py -q` → `4 passed`.
Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_scorer.py -q` → `4 passed`.

- [ ] **Step 3: Viết test fail cho stream_detect (không cần Kafka)**

`tests/test_stream_detect.py`:
```python
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
pytest.importorskip("onnx")
from common.fake import write_fake_dataset  # noqa: E402
from streaming.output_io import OUTPUT_COLUMNS, load_output  # noqa: E402
from tests.onnx_fixture import make_mean_square_onnx  # noqa: E402

pytestmark = pytest.mark.spark


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession
    s = SparkSession.builder.master("local[2]").appName("test-stream") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true").getOrCreate()
    yield s
    s.stop()


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    d = tmp_path_factory.mktemp("sd")
    scaler = write_fake_dataset(d / "data", n=2000, scaler_path=d / "scaler.json")
    make_mean_square_onnx(d / "m.onnx", len(scaler.features))
    (d / "thr.json").write_text(json.dumps({"percentile": 99, "threshold": 1.0}))
    return d, scaler


def _kafka_like_df(spark, d, scaler, n=30):
    df = pd.read_parquet(d / "data" / "test").head(n)
    rows = []
    for _, r in df.iterrows():
        msg = {"flow_id": int(r["flow_id"]), "produce_ts": 1000, "orig_ts": r["orig_ts"], "label": r["Label"],
               "features": {f: float(r[f]) for f in scaler.features}}
        rows.append((bytearray(json.dumps(msg).encode()), pd.Timestamp("2026-10-14 10:00:00")))
    return spark.createDataFrame(rows, "value binary, timestamp timestamp"), df


def test_parse_stream(spark, assets):
    from streaming.stream_detect import parse_stream
    d, scaler = assets
    raw, df = _kafka_like_df(spark, d, scaler)
    out = parse_stream(raw, scaler.features).toPandas()
    assert list(out.columns[:5]) == ["flow_id", "orig_ts", "label", "produce_ts", "kafka_ts"]
    assert out.shape[1] == 5 + len(scaler.features)
    np.testing.assert_allclose(out["f0"].to_numpy(), df[scaler.features[0]].to_numpy())
    assert (out["kafka_ts"] > 0).all()


@pytest.mark.parametrize("mode", ["driver", "executor"])
def test_batch_handler_writes_output(spark, assets, tmp_path, mode):
    from streaming.stream_detect import BatchHandler, parse_stream, prepare_executor_assets
    d, scaler = assets
    args = SimpleNamespace(model=str(d / "m.onnx"), scaler=str(d / "scaler.json"), threshold=str(d / "thr.json"),
                           out_dir=str(tmp_path / "out"), batches_csv=str(tmp_path / "b.csv"), inference=mode)
    (tmp_path / "out").mkdir()
    if mode == "executor":
        prepare_executor_assets(spark, args)
    raw, df = _kafka_like_df(spark, d, scaler)
    h = BatchHandler(args, len(scaler.features))
    h(parse_stream(raw, scaler.features), 7)
    out = load_output(tmp_path / "out")
    assert list(out.columns) == OUTPUT_COLUMNS and len(out) == len(df) and h.rows_seen == len(df)
    assert (tmp_path / "out" / "batch-00000007.parquet").exists()
    b = pd.read_csv(tmp_path / "b.csv")
    assert list(b.columns) == ["batch_id", "n_rows", "batch_start_ts", "batch_end_ts", "duration_ms"]
    assert b["n_rows"].tolist() == [len(df)]
```

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_stream_detect.py -q`
Expected: FAIL (`No module named 'streaming.stream_detect'`).

- [ ] **Step 4: Viết `streaming/stream_detect.py`**

```python
"""Spark Structured Streaming: Kafka -> chuẩn hoá -> ONNX -> Parquet (PROJECT_SPEC 5.6).

Inference (--inference):
  executor (mặc định): foreachBatch + mapInPandas chạy trên executor. Cả 3 VM cần requirements-cluster.txt;
                       model/scaler/threshold gửi bằng SparkContext.addFile, code (common/, streaming/) bằng addPyFile;
                       mỗi Python worker load ONNX một lần (streaming.scorer.get_scorer).
  driver:              foreachBatch + toPandas() micro-batch, suy luận trên driver (chỉ master cần onnxruntime).
Kết quả mỗi micro-batch (đã giới hạn bởi --max-offsets-per-trigger) gom về driver và ghi
<out-dir>/batch-<id>.parquet trên đĩa của máy chạy driver (master); dashboard/benchmark đọc thư mục này.
Cụm:  bash scripts/run_stream.sh [tham số]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as C  # noqa: E402
from common.data import load_scaler  # noqa: E402
from streaming.output_io import OUTPUT_COLUMNS  # noqa: E402

OUT_SCHEMA = ("flow_id long, orig_ts string, label string, produce_ts long, kafka_ts long, "
              "recon_error double, is_anomaly boolean")
BATCH_COLUMNS = ["batch_id", "n_rows", "batch_start_ts", "batch_end_ts", "duration_ms"]


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--master", default=None, help="vd local[2]; bỏ trống khi dùng spark-submit --master")
    p.add_argument("--bootstrap", default=None)
    p.add_argument("--topic", default=None)
    p.add_argument("--model", default=None, help="mặc định models.stream_model")
    p.add_argument("--threshold", default=None, help="mặc định models.stream_threshold")
    p.add_argument("--scaler", default=None, help="mặc định models.scaler")
    p.add_argument("--out-dir", default=None, help="thư mục cục bộ trên driver")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batches-csv", default=None)
    p.add_argument("--trigger", type=float, default=None, help="giây")
    p.add_argument("--max-offsets-per-trigger", type=int, default=None)
    p.add_argument("--starting-offsets", default=None, choices=["latest", "earliest"])
    p.add_argument("--inference", default=None, choices=["executor", "driver"])
    p.add_argument("--kafka-package", default=None, help="chỉ dùng khi chạy bằng python (spark-submit dùng --packages)")
    p.add_argument("--timeout", type=float, default=0, help="dừng sau N giây (0 = không)")
    p.add_argument("--stop-when-idle", type=float, default=0, help="dừng khi N giây không có dữ liệu mới (sau khi đã có)")
    p.add_argument("--clean", action="store_true", help="xoá out-dir + checkpoint cục bộ trước khi chạy")
    a = p.parse_args(argv)
    cfg = C.load_config(a.config)
    defaults = {"bootstrap": "cluster.kafka_bootstrap", "topic": "kafka.topic", "model": "models.stream_model",
                "threshold": "models.stream_threshold", "scaler": "models.scaler", "out_dir": "streaming.output_dir",
                "checkpoint": "streaming.checkpoint_dir", "batches_csv": "streaming.batches_csv",
                "trigger": "streaming.trigger_s", "max_offsets_per_trigger": "streaming.max_offsets_per_trigger",
                "starting_offsets": "streaming.starting_offsets", "inference": "streaming.inference",
                "kafka_package": "cluster.kafka_package"}
    for attr, key in defaults.items():
        if getattr(a, attr) is None:
            setattr(a, attr, C.get(cfg, key))
    return a


def build_spark(a):
    from pyspark.sql import SparkSession
    b = SparkSession.builder.appName("btl-stream-detect") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "4")
    if a.master:
        b = b.master(a.master)
    if a.kafka_package:
        b = b.config("spark.jars.packages", a.kafka_package)
    return b.getOrCreate()


def prepare_executor_assets(spark, a) -> None:
    """Gửi model/scaler/threshold + code cho executor."""
    sc = spark.sparkContext
    for f in (a.model, a.scaler, a.threshold):
        sc.addFile(str(Path(f).resolve()))
    zpath = Path(tempfile.gettempdir()) / "btl_bigdeep_code.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for pkg in ("common", "streaming"):
            for f in sorted((ROOT / pkg).rglob("*.py")):
                z.write(f, f.relative_to(ROOT).as_posix())
    sc.addPyFile(str(zpath))


def parse_stream(raw, features):
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType
    feat = StructType([StructField(f, DoubleType()) for f in features])
    schema = StructType([StructField("flow_id", LongType()), StructField("produce_ts", LongType()),
                         StructField("orig_ts", StringType()), StructField("label", StringType()),
                         StructField("features", feat)])
    m = raw.select(F.from_json(F.col("value").cast("string"), schema).alias("m"),
                   (F.col("timestamp").cast("double") * 1000).cast("long").alias("kafka_ts"))
    cols = [F.col("m.flow_id").alias("flow_id"), F.col("m.orig_ts").alias("orig_ts"),
            F.col("m.label").alias("label"), F.col("m.produce_ts").alias("produce_ts"), F.col("kafka_ts")]
    cols += [F.col("m.features").getField(f).alias("f%d" % i) for i, f in enumerate(features)]
    return m.select(*cols).filter(F.col("flow_id").isNotNull() & F.col("produce_ts").isNotNull())


def _make_infer(names, feat_cols):
    model_name, scaler_name, thr_name = names

    def infer(it):
        from pyspark import SparkFiles

        from streaming.scorer import get_scorer, score_frame
        scorer = get_scorer(SparkFiles.get(model_name), SparkFiles.get(scaler_name), SparkFiles.get(thr_name))
        for pdf in it:
            yield score_frame(pdf, scorer, feat_cols)
    return infer


def append_batch_row(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BATCH_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


class BatchHandler:
    def __init__(self, a, n_features: int):
        self.a = a
        self.feat_cols = ["f%d" % i for i in range(n_features)]
        self.out = Path(a.out_dir)
        self.names = (Path(a.model).name, Path(a.scaler).name, Path(a.threshold).name)
        self.rows_seen = 0
        self.last_data_ms = None

    def __call__(self, batch_df, batch_id):
        t0 = now_ms()
        if self.a.inference == "executor":
            pdf = batch_df.mapInPandas(_make_infer(self.names, self.feat_cols), OUT_SCHEMA).toPandas()
        else:
            from streaming.scorer import get_scorer, score_frame
            scorer = get_scorer(self.a.model, self.a.scaler, self.a.threshold)
            pdf = score_frame(batch_df.toPandas(), scorer, self.feat_cols)
        n = len(pdf)
        if n:
            pdf["detect_ts"] = now_ms()
            dst = self.out / ("batch-%08d.parquet" % batch_id)
            tmp = self.out / ("batch-%08d.parquet.tmp" % batch_id)
            pdf[OUTPUT_COLUMNS].to_parquet(tmp, index=False)
            os.replace(tmp, dst)
            self.rows_seen += n
            self.last_data_ms = now_ms()
        t1 = now_ms()
        append_batch_row(self.a.batches_csv, {"batch_id": batch_id, "n_rows": n, "batch_start_ts": t0,
                                              "batch_end_ts": t1, "duration_ms": t1 - t0})
        print("[stream] batch %d: %d rows in %d ms (total %d)" % (batch_id, n, t1 - t0, self.rows_seen), flush=True)


def main(argv=None) -> dict:
    a = parse_args(argv)
    if a.clean:
        for p in (a.out_dir, a.checkpoint):
            if not str(p).startswith("hdfs://"):
                shutil.rmtree(p, ignore_errors=True)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    features = load_scaler(a.scaler).features
    spark = build_spark(a)
    if a.inference == "executor":
        prepare_executor_assets(spark, a)
    reader = spark.readStream.format("kafka") \
        .option("kafka.bootstrap.servers", a.bootstrap) \
        .option("subscribe", a.topic) \
        .option("startingOffsets", a.starting_offsets) \
        .option("failOnDataLoss", "false")
    if a.max_offsets_per_trigger:
        reader = reader.option("maxOffsetsPerTrigger", str(a.max_offsets_per_trigger))
    handler = BatchHandler(a, len(features))
    q = parse_stream(reader.load(), features).writeStream \
        .foreachBatch(handler) \
        .option("checkpointLocation", a.checkpoint) \
        .trigger(processingTime="%g seconds" % a.trigger) \
        .start()
    (Path(a.out_dir) / "_STARTED").write_text(str(now_ms()), encoding="utf-8")
    print("[stream] query started: topic=%s inference=%s out=%s" % (a.topic, a.inference, a.out_dir), flush=True)
    t_start = time.time()
    try:
        while q.isActive:
            q.awaitTermination(1)
            if a.timeout and time.time() - t_start > a.timeout:
                print("[stream] timeout %gs -> stop" % a.timeout, flush=True)
                break
            if a.stop_when_idle and handler.last_data_ms and now_ms() - handler.last_data_ms > a.stop_when_idle * 1000:
                print("[stream] idle %gs -> stop" % a.stop_when_idle, flush=True)
                break
    except KeyboardInterrupt:
        print("[stream] dừng theo yêu cầu", flush=True)
    finally:
        q.stop()
    return {"rows": handler.rows_seen}


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Chạy test, xác nhận pass**

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_stream_detect.py tests/test_scorer.py -q`
Expected: `7 passed`.

- [ ] **Step 6: Commit**

```bash
git add streaming/scorer.py streaming/output_io.py streaming/stream_detect.py tests/onnx_fixture.py tests/test_scorer.py tests/test_stream_detect.py
git commit -m "feat(streaming): ONNX scorer and Spark Structured Streaming detector" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: End-to-end Kafka → Spark → Parquet + `run_stream.sh` — kết thúc M3

**Files:**
- Create: `tests/test_e2e_stream.py`, `scripts/run_stream.sh`
- Modify: `README.md`

**Interfaces:**
- Consumes: `streaming.producer.main`, `streaming.stream_detect.main`, `streaming.output_io.load_output`, `common.metrics.confusion_metrics`, `common.data.load_split`, `tests.onnx_fixture`.
- Produces: bằng chứng “F1 streaming ≈ F1 offline”; `bash scripts/run_stream.sh [tham số stream_detect]`.

- [ ] **Step 1: Viết test e2e**

`tests/test_e2e_stream.py`:
```python
import json
import os
import uuid

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")
pytest.importorskip("onnx")
from common.data import load_split  # noqa: E402
from common.fake import write_fake_dataset  # noqa: E402
from common.metrics import confusion_metrics  # noqa: E402
from streaming import producer, stream_detect  # noqa: E402
from streaming.output_io import load_output  # noqa: E402
from tests.onnx_fixture import make_mean_square_onnx  # noqa: E402

pytestmark = [pytest.mark.spark, pytest.mark.kafka]
BOOT = os.environ.get("KAFKA_BOOTSTRAP")


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    if not BOOT:
        pytest.skip("KAFKA_BOOTSTRAP chưa đặt")
    d = tmp_path_factory.mktemp("e2e")
    scaler = write_fake_dataset(d / "data", n=5000, scaler_path=d / "scaler.json")
    make_mean_square_onnx(d / "m.onnx", len(scaler.features))
    val = load_split(d / "data", "val", scaler, benign_only=True)
    thr = float(np.percentile((val.X.astype(np.float64) ** 2).mean(axis=1), 99))
    (d / "thr.json").write_text(json.dumps({"percentile": 99, "threshold": thr}))
    return d, scaler, thr


@pytest.mark.parametrize("mode", ["driver", "executor"])
def test_stream_matches_offline(prepared, mode):
    d, scaler, thr = prepared
    topic = "e2e-%s-%s" % (mode, uuid.uuid4().hex[:8])
    producer.main(["--bootstrap", BOOT, "--topic", topic, "--create-topic", "--input", str(d / "data" / "test"),
                   "--scaler", str(d / "scaler.json"), "--rate", "0"])
    out = d / ("out_" + mode)
    res = stream_detect.main(["--master", "local[2]", "--bootstrap", BOOT, "--topic", topic,
                              "--model", str(d / "m.onnx"), "--threshold", str(d / "thr.json"),
                              "--scaler", str(d / "scaler.json"), "--out-dir", str(out),
                              "--checkpoint", str(d / ("ckpt_" + mode)), "--batches-csv", str(d / ("b_%s.csv" % mode)),
                              "--starting-offsets", "earliest", "--trigger", "1", "--max-offsets-per-trigger", "300",
                              "--inference", mode, "--stop-when-idle", "10", "--timeout", "300", "--clean"])
    te = load_split(d / "data", "test", scaler)
    exp = pd.DataFrame({"flow_id": te.meta["flow_id"], "err": (te.X.astype(np.float64) ** 2).mean(axis=1),
                        "y": te.y}).sort_values("flow_id").reset_index(drop=True)
    got = load_output(out).sort_values("flow_id").reset_index(drop=True)
    assert res["rows"] == len(exp) == len(got)
    np.testing.assert_array_equal(got["flow_id"].to_numpy(), exp["flow_id"].to_numpy())
    np.testing.assert_allclose(got["recon_error"].to_numpy(), exp["err"].to_numpy(), rtol=1e-5)
    f1_stream = confusion_metrics((got["label"] != "Benign").astype(int), got["is_anomaly"])["f1"]
    f1_off = confusion_metrics(exp["y"], exp["err"] >= thr)["f1"]
    assert abs(f1_stream - f1_off) < 0.01
    assert (got["detect_ts"] >= got["produce_ts"]).all()
    assert len(pd.read_csv(d / ("b_%s.csv" % mode))) >= 2     # nhiều micro-batch (maxOffsetsPerTrigger=300)
```

- [ ] **Step 2: Chạy test e2e**

Run: `docker compose -f docker/docker-compose.test.yml run --rm cluster pytest tests/test_e2e_stream.py -q -s`
Expected: `2 passed` (lần đầu tải connector Kafka từ Maven vào volume `ivy-cache`, ~1–2 phút). Nếu fail: dùng skill `superpowers:systematic-debugging`; KHÔNG nới tolerance để test pass.

- [ ] **Step 3: `scripts/run_stream.sh`**

```bash
#!/usr/bin/env bash
# Chạy stream_detect trên cụm. Tham số thêm chuyển thẳng cho streaming/stream_detect.py.
# Số core executor tổng: TOTAL_CORES (mặc định spark.total_executor_cores) — dùng để so 1 vs 2 worker.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export BTL_CONFIG="${BTL_CONFIG:-$ROOT/config.yaml}"
cfg() { python3 scripts/cfg.py "$1"; }
spark-submit --master "$(cfg cluster.spark_master)" \
  --packages "$(cfg cluster.kafka_package)" \
  --driver-memory "$(cfg spark.driver_memory)" --executor-memory "$(cfg spark.executor_memory)" \
  --executor-cores "$(cfg spark.executor_cores)" \
  --total-executor-cores "${TOTAL_CORES:-$(cfg spark.total_executor_cores)}" \
  --conf spark.pyspark.python=python3 \
  streaming/stream_detect.py "$@"
```

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster bash -n scripts/run_stream.sh` → không lỗi.

- [ ] **Step 4: README + commit**

Thêm vào `README.md`:
````markdown
## Streaming (M3) — chạy trên master
Inference mặc định chạy **trên executor** (`foreachBatch` + `mapInPandas`): cả 3 VM phải cài `requirements-cluster.txt`;
model/scaler/threshold được gửi qua `SparkContext.addFile`. Nếu slave thiếu package: thêm `--inference driver`
(suy luận trên driver/master bằng `toPandas()` từng micro-batch).
Output: `results/stream/default/batch-*.parquet` (đĩa master) + `results/stream_batches.csv`.
```bash
# Terminal 1: detector
bash scripts/run_stream.sh --clean --starting-offsets latest
# Terminal 2: producer (đọc data/processed/test hoặc --input hdfs:///...)
python3 streaming/producer.py --create-topic --rate 2000 --limit 200000
# Đổi model: --model models/vae.onnx --threshold models/threshold_vae.json
```
````

```bash
git add tests/test_e2e_stream.py scripts/run_stream.sh README.md
git commit -m "test(streaming): end-to-end Kafka->Spark->ONNX matches offline (M3)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Báo người dùng chạy Phụ lục A6 trên cụm và gửi lại: log `[stream] batch ...`, kết quả `python3 streaming/benchmark.py --out-dir results/stream/default --rate 2000 --trigger 2 --n-workers 2 --no-append` (Task 16) để so F1 với `results/dl_ae_metrics.csv`.

---

### Task 15: Dashboard Streamlit — M4

**Files:**
- Create: `dashboard/view_model.py`, `dashboard/app.py`
- Test: `tests/test_dashboard.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `streaming.output_io.load_output`, `common.metrics.confusion_metrics`, `common.data.binarize`, `common.config`.
- Produces: `view_model.kpis(df, window_s=10) -> {"total","alerts","alert_rate","recent_throughput"}`, `view_model.per_second(df) -> DataFrame[time, flows, alerts]`, `view_model.top_suspicious(df, k=20, recent=5000) -> DataFrame[flow_id, orig_ts, recon_error, detect_time]`, `view_model.evaluation(df) -> {"metrics": dict, "per_label": DataFrame[label, n, alert_rate]}`. Tab "Monitor" KHÔNG hiển thị nhãn thật; nhãn chỉ ở tab "Evaluation".

- [ ] **Step 1: Viết test fail**

`tests/test_dashboard.py`:
```python
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from dashboard.view_model import evaluation, kpis, per_second, top_suspicious

ROOT = Path(__file__).resolve().parents[1]


def _df():
    return pd.DataFrame({
        "flow_id": [1, 2, 3, 4], "orig_ts": ["a", "b", "c", "d"], "label": ["Benign", "Bot", "Benign", "Bot"],
        "recon_error": [0.1, 5.0, 2.0, 0.2], "is_anomaly": [False, True, True, False],
        "produce_ts": [0, 0, 0, 0], "kafka_ts": [0, 0, 0, 0], "detect_ts": [1000, 1500, 9000, 10500]})


def test_kpis():
    k = kpis(_df(), window_s=2)
    assert k["total"] == 4 and k["alerts"] == 2 and k["alert_rate"] == 0.5
    assert k["recent_throughput"] == pytest.approx(1.0)   # 2 dòng có detect_ts > 10500-2000, chia 2s


def test_kpis_empty():
    assert kpis(_df().head(0))["total"] == 0


def test_per_second():
    ps = per_second(_df())
    assert ps["flows"].tolist() == [2, 1, 1] and ps["alerts"].tolist() == [1, 1, 0]


def test_top_suspicious_hides_label():
    t = top_suspicious(_df(), k=2)
    assert t["flow_id"].tolist() == [2, 3] and "label" not in t.columns


def test_evaluation():
    ev = evaluation(_df())
    assert ev["metrics"]["precision"] == 0.5 and ev["metrics"]["recall"] == 0.5
    assert set(ev["per_label"]["label"]) == {"Benign", "Bot"}


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_app_starts_headless(tmp_path):
    pytest.importorskip("streamlit")
    _df().to_parquet(tmp_path / "batch-00000001.parquet")
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
                             "--server.headless", "true", "--server.port", str(port), "--",
                             "--out-dir", str(tmp_path), "--threshold", str(tmp_path / "none.json")],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        ok = False
        for _ in range(60):
            try:
                ok = urllib.request.urlopen("http://127.0.0.1:%d/_stcore/health" % port, timeout=1).read() == b"ok"
                if ok:
                    break
            except OSError:
                time.sleep(0.5)
        assert ok
    finally:
        proc.terminate()
        proc.wait(10)
```

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_dashboard.py -q` → FAIL (`No module named 'dashboard.view_model'`).

- [ ] **Step 2: Viết `dashboard/view_model.py`**

```python
"""Logic thuần (không import streamlit) cho dashboard — test được."""
from __future__ import annotations

import pandas as pd

from common.data import binarize
from common.metrics import confusion_metrics


def kpis(df: pd.DataFrame, window_s: float = 10) -> dict:
    total = len(df)
    if total == 0:
        return {"total": 0, "alerts": 0, "alert_rate": 0.0, "recent_throughput": 0.0}
    alerts = int(df["is_anomaly"].sum())
    last = df["detect_ts"].max()
    recent = int((df["detect_ts"] > last - window_s * 1000).sum())
    return {"total": total, "alerts": alerts, "alert_rate": alerts / total, "recent_throughput": recent / window_s}


def per_second(df: pd.DataFrame) -> pd.DataFrame:
    g = df.assign(sec=df["detect_ts"] // 1000).groupby("sec") \
        .agg(flows=("flow_id", "size"), alerts=("is_anomaly", "sum")).reset_index()
    g["time"] = pd.to_datetime(g["sec"], unit="s")
    g["alerts"] = g["alerts"].astype(int)
    return g[["time", "flows", "alerts"]]


def top_suspicious(df: pd.DataFrame, k: int = 20, recent: int = 5000) -> pd.DataFrame:
    t = df.sort_values("detect_ts").tail(recent).nlargest(k, "recon_error")
    t = t.assign(detect_time=pd.to_datetime(t["detect_ts"], unit="ms"))
    return t[["flow_id", "orig_ts", "recon_error", "detect_time"]].reset_index(drop=True)


def evaluation(df: pd.DataFrame) -> dict:
    m = confusion_metrics(binarize(df["label"]), df["is_anomaly"])
    per_label = df.groupby("label").agg(n=("flow_id", "size"), alert_rate=("is_anomaly", "mean")).reset_index()
    return {"metrics": m, "per_label": per_label}
```

- [ ] **Step 3: Viết `dashboard/app.py`**

```python
"""Dashboard giám sát. Chạy: streamlit run dashboard/app.py -- [--out-dir DIR --threshold models/threshold.json]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from common import config as C  # noqa: E402
from dashboard.view_model import evaluation, kpis, per_second, top_suspicious  # noqa: E402
from streaming.output_io import load_output  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--threshold", default=None)
    p.add_argument("--refresh", type=float, default=None)
    p.add_argument("--max-files", type=int, default=None)
    a, _ = p.parse_known_args()
    cfg = C.load_config(a.config)
    a.out_dir = a.out_dir or C.get(cfg, "streaming.output_dir")
    a.threshold = a.threshold or C.get(cfg, "models.stream_threshold")
    a.refresh = a.refresh or C.get(cfg, "dashboard.refresh_s", 3)
    a.max_files = a.max_files or C.get(cfg, "dashboard.max_files", 2000)
    return a


args = parse_args()
st.set_page_config(page_title="Network Anomaly Detection", layout="wide")
st.title("Real-time Network Anomaly Detection — Deep Autoencoder")
thr = None
if Path(args.threshold).exists():
    thr = float(json.loads(Path(args.threshold).read_text(encoding="utf-8"))["threshold"])
df = load_output(args.out_dir, max_files=args.max_files)
tab_mon, tab_eval = st.tabs(["Monitor", "Evaluation"])

with tab_mon:
    if df.empty:
        st.info("Chưa có dữ liệu trong %s — hãy chạy stream_detect + producer." % args.out_dir)
    else:
        k = kpis(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flow đã xử lý", "{:,}".format(k["total"]))
        c2.metric("Cảnh báo", "{:,}".format(k["alerts"]))
        c3.metric("Tỷ lệ cảnh báo", "%.2f%%" % (100 * k["alert_rate"]))
        c4.metric("Throughput 10s gần nhất", "%.0f flow/s" % k["recent_throughput"])
        st.subheader("Số flow và cảnh báo theo giây")
        st.line_chart(per_second(df).set_index("time")[["flows", "alerts"]])
        st.subheader("Reconstruction error (3000 flow gần nhất)" + (" — ngưỡng %.4g" % thr if thr else ""))
        recent = df.sort_values("detect_ts").tail(3000).reset_index(drop=True)
        recent = recent.assign(seq=range(len(recent)), err=recent["recon_error"].clip(lower=1e-6))
        chart = alt.Chart(recent).mark_circle(size=12).encode(
            x=alt.X("seq:Q", title="flow (cũ → mới)"),
            y=alt.Y("err:Q", scale=alt.Scale(type="log"), title="reconstruction error"),
            color=alt.Color("is_anomaly:N", title="cảnh báo"))
        if thr:
            chart = chart + alt.Chart(pd.DataFrame({"thr": [thr]})).mark_rule(color="red", strokeDash=[4, 4]).encode(y="thr:Q")
        st.altair_chart(chart, use_container_width=True)
        st.subheader("Flow đáng ngờ nhất gần đây")
        st.dataframe(top_suspicious(df), use_container_width=True)

with tab_eval:
    st.caption("Chỉ để đánh giá: hệ thống thật không có nhãn.")
    if not df.empty:
        ev = evaluation(df)
        m = ev["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", "%.4f" % m["precision"])
        c2.metric("Recall", "%.4f" % m["recall"])
        c3.metric("F1", "%.4f" % m["f1"])
        c4.metric("FPR", "%.4f" % m["fpr"])
        st.dataframe(ev["per_label"], use_container_width=True)

time.sleep(args.refresh)
st.rerun()
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_dashboard.py -q`
Expected: `6 passed`. Nếu `st.rerun` không có trong bản Streamlit đã pin (bản < 1.27), dùng `st.experimental_rerun()`.

- [ ] **Step 5: Xem dashboard bằng mắt**

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps -p 8501:8501 cluster streamlit run dashboard/app.py --server.headless true -- --out-dir <thư mục output của test e2e>`; mở http://localhost:8501, kiểm tra 4 KPI, 2 biểu đồ, bảng, tab Evaluation.

- [ ] **Step 6: README + commit**

Thêm vào `README.md`:
````markdown
## Dashboard (M4) — trên master
```bash
streamlit run dashboard/app.py --server.port 8501 -- --out-dir results/stream/default
```
Mở `http://<master>:8501` khi producer + stream_detect đang chạy. Tab "Evaluation" mới dùng nhãn thật.
````

```bash
git add dashboard/ tests/test_dashboard.py README.md
git commit -m "feat(dashboard): Streamlit monitor + evaluation tabs (M4)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 16: `streaming/benchmark.py` + `scripts/run_benchmark.sh` — M5

**Files:**
- Create: `streaming/benchmark.py`, `scripts/run_benchmark.sh`
- Test: `tests/test_benchmark.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `streaming.output_io.load_output`, `common.metrics.confusion_metrics`, `common.data.binarize`; file `_producer.json` từ `producer.py --stats-out`.
- Produces: `summarize(df) -> dict` (`n_rows, duration_s, throughput_rps, lat_p50_ms, lat_p95_ms, lat_p99_ms, lat_max_ms, kafka_p50_ms, kafka_p95_ms, kafka_p99_ms, precision, recall, f1, fpr`); CLI append 1 dòng vào `results/streaming_benchmark.csv` (thêm `timestamp, tag, rate, actual_rate, trigger_s, n_workers, max_offsets, inference`); `--plot` vẽ `reports/stream_throughput.png`, `reports/stream_latency.png`, `reports/stream_workers.png`.

- [ ] **Step 1: Viết test fail**

`tests/test_benchmark.py`:
```python
import pandas as pd
import pytest

from streaming.benchmark import main, summarize

pytest.importorskip("matplotlib")


def _df():
    return pd.DataFrame({
        "flow_id": [1, 2, 3, 4], "orig_ts": "x", "label": ["Benign", "Bot", "Benign", "Bot"],
        "recon_error": [0.1, 5.0, 2.0, 0.2], "is_anomaly": [False, True, True, False],
        "produce_ts": [0, 500, 1500, 2000], "kafka_ts": [10, 510, 1510, 2010], "detect_ts": [1000, 1000, 2000, 3000]})


def test_summarize():
    s = summarize(_df())
    assert s["n_rows"] == 4 and s["duration_s"] == 2.0 and s["throughput_rps"] == 2.0
    assert s["lat_max_ms"] == 1000 and s["lat_p50_ms"] == pytest.approx(750)
    assert s["kafka_p50_ms"] == pytest.approx(740)
    assert s["precision"] == 0.5 and s["recall"] == 0.5


def test_cli_appends_and_plots(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    _df().to_parquet(out / "batch-00000001.parquet")
    csv = tmp_path / "bench.csv"
    for rate, w in ((1000, 1), (1000, 2), (5000, 2)):
        main(["--out-dir", str(out), "--rate", str(rate), "--trigger", "2", "--n-workers", str(w), "--csv", str(csv)])
    df = pd.read_csv(csv)
    assert len(df) == 3 and {"rate", "n_workers", "throughput_rps", "lat_p95_ms", "f1"} <= set(df.columns)
    main(["--plot", "--csv", str(csv), "--reports-dir", str(tmp_path / "rep")])
    for f in ("stream_throughput.png", "stream_latency.png", "stream_workers.png"):
        assert (tmp_path / "rep" / f).exists()
```

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_benchmark.py -q` → FAIL.

- [ ] **Step 2: Viết `streaming/benchmark.py`**

```python
"""Đo throughput/latency/F1 từ output một lần chạy streaming (PROJECT_SPEC 5.7).

python3 streaming/benchmark.py --out-dir results/stream/r5000_w2 --rate 5000 --trigger 2 --n-workers 2
python3 streaming/benchmark.py --plot          # vẽ hình từ results/streaming_benchmark.csv
throughput = n / (max detect_ts - min detect_ts); latency e2e = detect_ts - produce_ts; Kafka->detect = detect_ts - kafka_ts.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data import binarize  # noqa: E402
from common.metrics import confusion_metrics  # noqa: E402
from streaming.output_io import load_output  # noqa: E402


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        raise ValueError("output rỗng")
    span = (df["detect_ts"].max() - df["detect_ts"].min()) / 1000.0
    e2e = (df["detect_ts"] - df["produce_ts"]).to_numpy(dtype=np.float64)
    kd = (df["detect_ts"] - df["kafka_ts"]).to_numpy(dtype=np.float64)
    m = confusion_metrics(binarize(df["label"]), df["is_anomaly"])
    row = {"n_rows": n, "duration_s": span, "throughput_rps": n / span if span > 0 else float("nan")}
    for q in (50, 95, 99):
        row["lat_p%d_ms" % q] = float(np.percentile(e2e, q))
    row["lat_max_ms"] = float(e2e.max())
    for q in (50, 95, 99):
        row["kafka_p%d_ms" % q] = float(np.percentile(kd, q))
    row.update({k: m[k] for k in ("precision", "recall", "f1", "fpr")})
    return row


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if csv_path.exists():
        new = pd.concat([pd.read_csv(csv_path), new], ignore_index=True, sort=False)
    new.to_csv(csv_path, index=False)


def plot(csv_path: Path, rep: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rep.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path).sort_values("rate")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for w, g in df.groupby("n_workers"):
        ax.plot(g["rate"], g["throughput_rps"], marker="o", label="%d worker" % w)
    lim = df["rate"].max()
    ax.plot([0, lim], [0, lim], ls=":", color="grey", label="lý tưởng (= rate)")
    ax.set(xlabel="rate producer (msg/s)", ylabel="throughput (flow/s)", title="Throughput theo rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rep / "stream_throughput.png", dpi=150)
    plt.close(fig)

    wmax = df["n_workers"].max()
    g = df[df["n_workers"] == wmax]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for q in ("lat_p50_ms", "lat_p95_ms", "lat_p99_ms"):
        ax.plot(g["rate"], g[q], marker="o", label=q.replace("lat_", "").replace("_ms", ""))
    ax.set(xlabel="rate producer (msg/s)", ylabel="latency end-to-end (ms)", yscale="log",
           title="Latency theo rate (%d worker)" % wmax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rep / "stream_latency.png", dpi=150)
    plt.close(fig)

    last = df.sort_values("timestamp").groupby(["rate", "n_workers"]).tail(1) if "timestamp" in df else df
    piv_t = last.pivot_table(index="rate", columns="n_workers", values="throughput_rps")
    piv_l = last.pivot_table(index="rate", columns="n_workers", values="lat_p95_ms")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    piv_t.plot.bar(ax=a1)
    a1.set(ylabel="throughput (flow/s)", title="Throughput: số worker")
    piv_l.plot.bar(ax=a2)
    a2.set(ylabel="latency p95 (ms)", title="Latency p95: số worker")
    for ax in (a1, a2):
        ax.legend(title="worker", fontsize=8)
    fig.tight_layout()
    fig.savefig(rep / "stream_workers.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", help="thư mục output của một lần chạy stream_detect")
    p.add_argument("--rate", type=float, default=None)
    p.add_argument("--trigger", type=float, default=None)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--max-offsets", type=int, default=None)
    p.add_argument("--inference", default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--producer-stats", default=None, help="file JSON từ producer --stats-out")
    p.add_argument("--csv", default="results/streaming_benchmark.csv")
    p.add_argument("--no-append", action="store_true", help="chỉ in kết quả")
    p.add_argument("--plot", action="store_true", help="vẽ hình từ --csv rồi thoát")
    p.add_argument("--reports-dir", default="reports")
    a = p.parse_args(argv)
    if a.plot:
        plot(Path(a.csv), Path(a.reports_dir))
        return None
    if not a.out_dir:
        p.error("--out-dir là bắt buộc (trừ khi --plot)")
    s = summarize(load_output(a.out_dir))
    actual = None
    if a.producer_stats and Path(a.producer_stats).exists():
        actual = json.loads(Path(a.producer_stats).read_text(encoding="utf-8")).get("actual_rate")
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": a.tag, "rate": a.rate, "actual_rate": actual,
           "trigger_s": a.trigger, "n_workers": a.n_workers, "max_offsets": a.max_offsets,
           "inference": a.inference, **s}
    print(json.dumps(row, indent=1))
    if not a.no_append:
        append_row(Path(a.csv), row)
    return row


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Chạy test, xác nhận pass**

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest tests/test_benchmark.py -q`
Expected: `2 passed`.

- [ ] **Step 4: `scripts/run_benchmark.sh`**

```bash
#!/usr/bin/env bash
# Benchmark nhiều mức rate. Mỗi run: topic mới -> stream_detect nền -> producer -> chờ detector dừng -> benchmark.
# Dùng: RATES="1000 5000 10000 20000" WORKERS=2 DURATION=60 bash scripts/run_benchmark.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export BTL_CONFIG="${BTL_CONFIG:-$ROOT/config.yaml}"
cfg() { python3 scripts/cfg.py "$1"; }
RATES="${RATES:-1000 5000 10000 20000}"
WORKERS="${WORKERS:-2}"
DURATION="${DURATION:-60}"
TRIGGER="${TRIGGER:-$(cfg streaming.trigger_s)}"
MAXOFF="${MAXOFF:-$(cfg streaming.max_offsets_per_trigger)}"
INFER="${INFER:-$(cfg streaming.inference)}"
export TOTAL_CORES=$((WORKERS * $(cfg spark.executor_cores)))
mkdir -p results/stream logs

for RATE in $RATES; do
  TAG="r${RATE}_w${WORKERS}_$(date +%s)"
  OUT="results/stream/$TAG"
  TOPIC="bench-$TAG"
  LIMIT=$((RATE * DURATION))
  echo "=== $TAG: rate=$RATE limit=$LIMIT cores=$TOTAL_CORES"
  python3 streaming/producer.py --topic "$TOPIC" --create-topic --limit 1 --rate 0 --input "$(cfg data.local_dir)/test" >/dev/null
  bash scripts/run_stream.sh --topic "$TOPIC" --starting-offsets earliest --out-dir "$OUT" \
    --checkpoint "checkpoints/$TAG" --batches-csv "$OUT/_batches.csv" --trigger "$TRIGGER" \
    --max-offsets-per-trigger "$MAXOFF" --inference "$INFER" --stop-when-idle 20 \
    --timeout $((DURATION * 3 + 180)) --clean > "logs/$TAG.stream.log" 2>&1 &
  SPID=$!
  for _ in $(seq 1 120); do [ -f "$OUT/_STARTED" ] && break; sleep 1; done
  [ -f "$OUT/_STARTED" ] || { echo "stream_detect không khởi động, xem logs/$TAG.stream.log"; kill $SPID; exit 1; }
  sleep 5
  python3 streaming/producer.py --topic "$TOPIC" --rate "$RATE" --limit "$LIMIT" --stats-out "$OUT/_producer.json"
  wait $SPID || true
  python3 streaming/benchmark.py --out-dir "$OUT" --rate "$RATE" --trigger "$TRIGGER" --n-workers "$WORKERS" \
    --max-offsets "$MAXOFF" --inference "$INFER" --tag "$TAG" --producer-stats "$OUT/_producer.json"
done
python3 streaming/benchmark.py --plot
```

Ghi chú: dòng producer `--limit 1` đầu tiên chỉ để tạo topic trước khi detector subscribe; flow đó cũng được tính vào run (1 dòng, không ảnh hưởng thống kê). Đúng-và-đơn-giản hơn là thêm cờ riêng.

Run: `docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster bash -n scripts/run_benchmark.sh` → không lỗi.

- [ ] **Step 5: README + commit**

Thêm vào `README.md`:
````markdown
## Benchmark (M5) — trên master
```bash
RATES="1000 5000 10000 20000" WORKERS=2 DURATION=60 bash scripts/run_benchmark.sh
RATES="1000 5000 10000 20000" WORKERS=1 DURATION=60 bash scripts/run_benchmark.sh
python3 streaming/benchmark.py --plot      # reports/stream_throughput.png, stream_latency.png, stream_workers.png
```
Kết quả: `results/streaming_benchmark.csv` (một dòng/run).
````

```bash
git add streaming/benchmark.py scripts/run_benchmark.sh tests/test_benchmark.py README.md
git commit -m "feat(streaming): benchmark throughput/latency/F1 + rate sweep script (M5)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Báo người dùng chạy Phụ lục A7.

---

### Task 17: Rà soát cuối, README hoàn chỉnh

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Chạy toàn bộ test hai môi trường**

```powershell
.venv\Scripts\python -m pytest -q
docker compose -f docker/docker-compose.test.yml run --rm cluster pytest -q
```
Expected: tất cả pass (các test bị skip đúng lý do: bash trên Windows, sklearn/torch trên Py3.8, Kafka/Spark trên venv Windows). Dán tóm tắt output vào mô tả commit.

- [ ] **Step 2: Kiểm tra spec coverage bằng checklist**

Với mỗi mục PROJECT_SPEC 5.1–5.9, mở file tương ứng và xác nhận: có `--help`, đọc config, có smoke/test, ghi log thời gian ở bước nặng. Với mỗi file output trong mục "File output" của plan, xác nhận có test kiểm tra tên + cột.

- [ ] **Step 3: Điền mục "Kết quả" trong README CHỈ từ file thật**

Thêm vào `README.md` mục "Kết quả" với đường dẫn đến `reports/model_comparison.md`, `results/ablation.csv`, `results/streaming_benchmark.csv`, `results/preprocess_stats.json`. Chỉ chép số từ các file này nếu chúng đã được sinh từ lần chạy thật trên Kaggle/cụm; nếu chưa có thì ghi "chưa chạy". Không chép số từ dữ liệu giả.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: final README with run commands and result pointers" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Ánh xạ spec → task

| Spec | Task |
|---|---|
| 2 Môi trường / config | 1 (+ Phụ lục A) |
| 4 Quy ước dữ liệu | 2, 3 |
| 5.1 spark_preprocess | 5 |
| 5.2 baselines | 6 |
| 5.3 dl_models / train_dl / ablation | 7, 8, 10 (+ 9 Kaggle) |
| 5.4 make_report_assets | 11 |
| 5.10 eda (phần 1 / phần 2) | 5b / 11b (+ Task 5 ghi `cleaning`, Task 7–8 lưu `.pt`) |
| 5.5 producer | 12 |
| 5.6 stream_detect | 13, 14 |
| 5.7 benchmark | 16 |
| 5.8 dashboard | 15 |
| 5.9 check_cluster | 4 |
| 6 Milestones / 7 Quy tắc | Lịch + Global Constraints + 17 |
