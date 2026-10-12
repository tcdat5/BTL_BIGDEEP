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

## Test local
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-kaggle.txt
.venv\Scripts\python -m pytest -q                                                      # phần train + common
docker compose -f docker/docker-compose.test.yml build cluster
docker compose -f docker/docker-compose.test.yml run --rm --no-deps cluster pytest -q  # phần cụm (Py3.8 + Java 8)
docker compose -f docker/docker-compose.test.yml run --rm cluster pytest -q            # kèm Kafka 3.7.2 (test kafka/e2e)
```
