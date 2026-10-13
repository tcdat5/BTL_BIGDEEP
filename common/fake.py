"""Dữ liệu giả nhỏ cho smoke test (giống cấu trúc CSE-CIC-IDS2018).

python -m common.fake --out data/fake --n 4000            # Parquet train/val/test + scaler.json
python -m common.fake --raw-csv data/fake_raw --n 1500    # CSV thô "bẩn" như dữ liệu gốc
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
RAW_DAYS = ["14-02-2018", "16-02-2018", "20-02-2018"]


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
    for i, day in enumerate(RAW_DAYS):
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
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)                      # dòng trùng
        header = pd.DataFrame([list(df.columns)], columns=df.columns)
        df = pd.concat([df.iloc[:100], header, df.iloc[100:]], ignore_index=True)  # header lặp
        name = "Fake-%s_TrafficForML_CICFlowMeter.csv" % day
        df.to_csv(out / name, index=False)
        names.append(name)
    return names


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=None, help="thư mục ghi Parquet train/val/test + scaler.json")
    p.add_argument("--scaler", default=None, help="mặc định <out>/scaler.json")
    p.add_argument("--raw-csv", default=None, help="thư mục ghi CSV thô giả (3 file)")
    p.add_argument("--n", type=int, default=4000, help="số dòng (với --raw-csv: số dòng mỗi file)")
    a = p.parse_args(argv)
    if not a.out and not a.raw_csv:
        p.error("cần --out hoặc --raw-csv")
    if a.out:
        write_fake_dataset(a.out, n=a.n, scaler_path=a.scaler or Path(a.out) / "scaler.json")
        print("wrote fake dataset to", a.out)
    if a.raw_csv:
        print("wrote", write_fake_raw_csvs(a.raw_csv, n_per_file=a.n), "to", a.raw_csv)


if __name__ == "__main__":
    main()
