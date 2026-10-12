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
