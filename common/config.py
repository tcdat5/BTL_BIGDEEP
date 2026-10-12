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
