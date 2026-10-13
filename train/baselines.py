"""
Train & evaluate baseline models to compare with the Deep Autoencoder.

Unsupervised (same setting as AE: fit on Benign only, threshold = percentile of val-Benign scores):
    PCA reconstruction, Isolation Forest, One-Class SVM
Supervised (uses attack labels, reported as a separate group):
    Random Forest, Logistic Regression, LightGBM (if installed)

Data, scaler and metrics come from common/ (same as the DL models):
    common.data.load_split  -> same capped test sample (random_state=42) for every model
    models/scaler.json      -> fitted on train Benign by preprocessing (fallback: fit here with common.fit_scaler)

Usage:
    python train/baselines.py --data /path/to/parquet_root --out results
    python train/baselines.py --data ... --holdout-attack "DDOS attack-HOIC"   # unseen-attack experiment
    python train/baselines.py --smoke --out .cache/smoke                       # fake data, CPU, ~10 s
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import config as C  # noqa: E402
from common.data import SEED, fit_scaler, load_scaler, load_split, read_split_frame  # noqa: E402
from common.metrics import METRIC_COLUMNS, evaluate, per_attack_recall, save_scores  # noqa: E402


# ---------------------------------------------------------------- data
def get_scaler(data_dir, scaler_path):
    """Scaler saved by preprocessing (same one AE uses); if absent, fit on train Benign and save."""
    if os.path.exists(scaler_path):
        return load_scaler(scaler_path)
    print(f"[warn] {scaler_path} not found -> fitting scaler on train Benign and saving it")
    train = read_split_frame(data_dir, "train")
    feats = [c for c in train.columns if c not in ("flow_id", "orig_ts", "Label")]
    scaler = fit_scaler(train, feats)
    scaler.to_json(scaler_path)
    return scaler


# ---------------------------------------------------------------- unsupervised
def pca_recon(n_components=0.95):
    class PCARecon:
        def fit(self, X):
            self.p = PCA(n_components=n_components, random_state=SEED).fit(X)
            return self

        def score(self, X):
            R = self.p.inverse_transform(self.p.transform(X))
            return ((X - R) ** 2).mean(axis=1)
    return PCARecon()


class SkWrap:
    """Turn sklearn novelty detectors into 'higher score = more anomalous'."""
    def __init__(self, model):
        self.m = model

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


# ---------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", "--data-dir", dest="data", default=None,
                    help="root with train/ val/ test/ parquet (default: data.sample_local_dir)")
    ap.add_argument("--out", "--results-dir", dest="out", default="results")
    ap.add_argument("--scaler", default=None, help="scaler.json (default: models.scaler)")
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="threshold = this percentile of val-Benign scores (use same as AE)")
    ap.add_argument("--train-cap", type=int, default=500_000)
    ap.add_argument("--test-cap", type=int, default=None,
                    help="same test sample for every model (default: data.test_cap = 300000)")
    ap.add_argument("--ocsvm-cap", type=int, default=20_000)
    ap.add_argument("--holdout-attack", default=None,
                    help="drop this attack from supervised training to test unseen attacks")
    ap.add_argument("--smoke", action="store_true", help="fake data (common/fake.py), quick CPU run")
    a = ap.parse_args(argv)
    cfg = C.load_config(a.config)
    a.data = a.data or C.get(cfg, "data.sample_local_dir")
    a.scaler = a.scaler or C.get(cfg, "models.scaler", "models/scaler.json")
    a.test_cap = a.test_cap or C.get(cfg, "data.test_cap", 300_000)
    if a.smoke:
        from common.fake import write_fake_dataset
        tmp = Path(tempfile.mkdtemp(prefix="baselines_smoke_"))
        write_fake_dataset(tmp / "data", n=4000, scaler_path=tmp / "scaler.json")
        a.data, a.scaler = str(tmp / "data"), str(tmp / "scaler.json")
    return a


def main(argv=None):
    a = parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    scaler = get_scaler(a.data, a.scaler)
    tr = load_split(a.data, "train", scaler, cap=a.train_cap)
    va = load_split(a.data, "val", scaler)
    te = load_split(a.data, "test", scaler, cap=a.test_cap)
    Xtr_s, ytr, atr = tr.X, tr.y, tr.labels
    Xte_s, yte, ate = te.X, te.y, te.labels
    print(f"train={len(ytr):,} val={len(va.y):,} test={len(yte):,} "
          f"test attack ratio={yte.mean():.3f}")

    Xtr_benign = Xtr_s[ytr == 0]
    Xva_benign = va.X[va.y == 0]
    np.save(os.path.join(a.out, "test_index_labels.npy"), np.stack([yte]))  # for AE alignment

    rows, per_attack_all = [], []
    rng = np.random.default_rng(SEED)

    def run(name, group, fit_fn, score_fn, thr_fn):
        t0 = time.time(); fit_fn(); t1 = time.time()
        s = score_fn(Xte_s); t2 = time.time()
        thr = thr_fn()
        row = {"model": name, "group": group, **evaluate(yte, s, thr),
               "train_time_s": round(t1 - t0, 2),
               "infer_ms_per_1k": round((t2 - t1) / len(yte) * 1e6, 3)}
        rows.append(row)
        per_attack_all.append(per_attack_recall(name, ate, s >= thr))
        save_scores(os.path.join(a.out, f"scores_{name}.npz"), yte, s, thr)
        print(f"{name:<20} F1={row['f1']:.4f} AUC={row['roc_auc']:.4f} "
              f"FPR={row['fpr']:.4f} train={row['train_time_s']}s")

    # --- unsupervised: fit on Benign only
    unsup = {
        "PCA": pca_recon(),
        "IsolationForest": SkWrap(IsolationForest(n_estimators=200, max_samples=256,
                                                  n_jobs=-1, random_state=SEED)),
        "OneClassSVM": SkWrap(OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")),
    }
    for name, m in unsup.items():
        Xfit = Xtr_benign
        if name == "OneClassSVM" and len(Xfit) > a.ocsvm_cap:
            Xfit = Xfit[rng.choice(len(Xfit), a.ocsvm_cap, replace=False)]
        run(name, "unsupervised",
            lambda m=m, X=Xfit: m.fit(X),
            lambda X, m=m: m.score(X),
            lambda m=m: np.percentile(m.score(Xva_benign), a.percentile))

    # --- supervised: uses attack labels
    mask = np.ones(len(ytr), bool)
    if a.holdout_attack:
        mask = atr != a.holdout_attack
        print(f"holding out '{a.holdout_attack}': removed {(~mask).sum():,} train rows")
        if (~mask).sum() == 0:
            print(f"[warn] '{a.holdout_attack}' not present in train data")
        with open(os.path.join(a.out, "holdout.json"), "w", encoding="utf-8") as f:
            json.dump({"attack": a.holdout_attack}, f)
    Xs, ys = Xtr_s[mask], ytr[mask]

    sup = {
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                               class_weight="balanced_subsample",
                                               random_state=SEED),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }
    try:
        from lightgbm import LGBMClassifier
        sup["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.1,
                                         class_weight="balanced", random_state=SEED,
                                         verbose=-1)
    except ImportError:
        print("lightgbm not installed -> skip (pip install lightgbm)")

    for name, m in sup.items():
        run(name, "supervised",
            lambda m=m: m.fit(Xs, ys),
            lambda X, m=m: m.predict_proba(X)[:, 1],
            lambda: 0.5)

    suffix = f"_holdout-{a.holdout_attack.replace(' ', '_')}" if a.holdout_attack else ""
    pd.DataFrame(rows, columns=METRIC_COLUMNS).to_csv(
        os.path.join(a.out, f"baselines_metrics{suffix}.csv"), index=False)
    pd.concat(per_attack_all).to_csv(
        os.path.join(a.out, f"baselines_per_attack{suffix}.csv"), index=False)
    print(f"\nSaved to {a.out}/")
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


if __name__ == "__main__":
    main()
