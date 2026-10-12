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
