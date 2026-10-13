# BTL_BigDeep — Real-time Network Anomaly Detection (CSE-CIC-IDS2018)

Deep Autoencoder + Kafka + Spark Structured Streaming trên cụm 3 VM. Spec đầy đủ: `PROJECT_SPEC.md`.
Kế hoạch: `docs/superpowers/plans/2026-10-12-network-anomaly-detection.md`.

## Môi trường
| Nơi | Python | Ghi chú |
|---|---|---|
| Cụm (master, slave1, slave2) | 3.8.10 | Spark 3.5.1 (Scala 2.13), Java 8, Hadoop 3.3.0, Kafka 3.7.2 KRaft trên master |
| Kaggle | 3.11+ | GPU cho DL, CPU cho baseline |
| Dev local | 3.12 venv (phần train) + Docker (phần cụm: Python 3.8 + Java 8 + Kafka 3.7.2) | |

`requirements-cluster.txt` đã kiểm chứng cài được trên Python 3.8 (image `docker/Dockerfile.cluster-test`):
pyspark 3.5.1, kafka-python 2.0.6, onnxruntime 1.17.3, numpy 1.24.4, pandas 2.0.3, pyarrow 14.0.2,
pyyaml 6.0.1, streamlit 1.39.0, matplotlib 3.7.5.

## Cài đặt
```bash
cp config.example.yaml config.yaml        # sửa hostname/port/đường dẫn HDFS/RAM theo cụm thật
python3 -m pip install --user -r requirements-cluster.txt   # trên CẢ 3 VM
```
Đọc một giá trị config trong shell: `python3 scripts/cfg.py cluster.kafka_bootstrap`.

## Tiền xử lý
- Giá trị lỗi: dòng có null/NaN/±Infinity ở **bất kỳ** feature nào bị **bỏ**.
- Bỏ cột định danh `Flow ID, Src IP, Dst IP, Src Port, Dst Port`; `Timestamp` giữ thành metadata `orig_ts`; bỏ cột hằng.
- Gộp dòng trùng (feature + Label), giữ `orig_ts` nhỏ nhất.
- Chia tập **tất định bằng hash** (giống hệt nhau trên Kaggle và trên cụm, không phụ thuộc số partition):
  `flow_id = xxhash64(feature..., Label)`, `bucket = pmod(flow_id, 100)`: `<70` train, `70–79` val, `≥80` test.
- Xuất thêm: bản mẫu Kaggle (`data.sample_sizes`), `data/demo_test.parquet` (50.000 dòng test phân tầng, mỗi
  loại tấn công ≥ min(500, số có)), `models/scaler.json`, `results/preprocess_stats.json`.
```bash
# Kaggle / local
python preprocess/spark_preprocess.py --master "local[*]" --input-dir <thư mục CSV> --out data/processed \
  --export-sample data/kaggle_sample --demo-out data/demo_test.parquet --scaler-out models/scaler.json
# Cụm (sau này)
bash scripts/run_preprocess.sh
```

## Test local
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-kaggle.txt
.venv\Scripts\python -m pytest -q                                                      # phần train + common
docker compose -f docker/docker-compose.test.yml build cluster
docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest -q  # phần cụm (Py3.8 + Java 8)
docker compose -f docker/docker-compose.test.yml run --rm cluster pytest -q            # kèm Kafka 3.7.2 (test kafka/e2e)
```
Máy dev Windows có Smart App Control: numpy 2.5 bị chặn DLL `numpy.random._mt19937` → trong venv dùng
`pip install "numpy==2.2.6"` (chỉ cho máy dev, Kaggle không cần).
