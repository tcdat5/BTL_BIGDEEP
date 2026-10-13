"""
Train & evaluate baseline models to compare with the Deep Autoencoder.

Unsupervised (same setting as AE: fit on Benign only, threshold = percentile of val-Benign scores):
    PCA reconstruction, Isolation Forest, One-Class SVM
Supervised (uses attack labels, reported as a separate group):
    Random Forest, Logistic Regression, LightGBM (if installed)

Inputs: Parquet produced by the Spark preprocessing job, split into train/ val/ test/.
        Each has numeric feature columns + a 'Label' column ('Benign' or attack name).

Usage:
    python baselines.py --data /path/to/parquet_root --out results
    python baselines.py --data ... --holdout-attack "DDOS attack-HOIC"   # unseen-attack experiment
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.svm import OneClassSVM

LABEL = "Label"
BENIGN = "Benign"
SEED = 42


# ---------------------------------------------------------------- data
def load_split(path, cap=None):
    df = pd.read_parquet(path)
    if cap and len(df) > cap:
        df = df.sample(n=cap, random_state=SEED)
    attack = df[LABEL].astype(str).values
    y = (attack != BENIGN).astype(int)
    X = df.drop(columns=[LABEL]).astype("float32")
    return X, y, attack


def get_scaler(X_train_benign, scaler_path):
    """Reuse the scaler saved by preprocessing (same one AE uses); fit+save if absent."""
    if os.path.exists(scaler_path):
        with open(scaler_path) as f:
            s = json.load(f)
        cols = s["features"]
        mean, std = np.array(s["mean"], "float32"), np.array(s["std"], "float32")
    else:
        cols = list(X_train_benign.columns)
        mean = X_train_benign.mean().values.astype("float32")
        std = X_train_benign.std().values.astype("float32")
        with open(scaler_path, "w") as f:
            json.dump({"features": cols, "mean": mean.tolist(), "std": std.tolist()}, f)
    std[std == 0] = 1.0

    def transform(X):
        return np.clip((X[cols].values - mean) / std, -10, 10).astype("float32")

    return transform


# ---------------------------------------------------------------- metrics
def evaluate(name, group, y, score, thr, attack, train_s, infer_s):
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    row = {
        "model": name, "group": group,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "threshold": float(thr),
        "train_time_s": round(train_s, 2),
        "infer_ms_per_1k": round(infer_s / len(y) * 1e6, 3),
    }
    per_attack = (pd.DataFrame({"attack": attack, "y": y, "pred": pred})
                  .query("y == 1").groupby("attack")["pred"]
                  .agg(recall="mean", n="size").reset_index())
    per_attack.insert(0, "model", name)
    return row, per_attack, pred


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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="root with train/ val/ test/ parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="threshold = this percentile of val-Benign scores (use same as AE)")
    ap.add_argument("--train-cap", type=int, default=500_000)
    ap.add_argument("--test-cap", type=int, default=300_000,
                    help="same test sample for every model; also evaluate AE on it")
    ap.add_argument("--ocsvm-cap", type=int, default=20_000)
    ap.add_argument("--holdout-attack", default=None,
                    help="drop this attack from supervised training to test unseen attacks")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    Xtr, ytr, atr = load_split(os.path.join(a.data, "train"), a.train_cap)
    Xva, yva, _ = load_split(os.path.join(a.data, "val"))
    Xte, yte, ate = load_split(os.path.join(a.data, "test"), a.test_cap)
    print(f"train={len(Xtr):,} val={len(Xva):,} test={len(Xte):,} "
          f"test attack ratio={yte.mean():.3f}")

    tf = get_scaler(Xtr[ytr == 0], os.path.join(a.out, "scaler.json"))
    Xtr_s, Xva_s, Xte_s = tf(Xtr), tf(Xva), tf(Xte)
    Xtr_benign = Xtr_s[ytr == 0]
    Xva_benign = Xva_s[yva == 0]
    np.save(os.path.join(a.out, "test_index_labels.npy"), np.stack([yte]))  # for AE alignment

    rows, per_attack_all = [], []
    rng = np.random.default_rng(SEED)

    def run(name, group, fit_fn, score_fn, thr_fn):
        t0 = time.time(); fit_fn(); t1 = time.time()
        s = score_fn(Xte_s); t2 = time.time()
        thr = thr_fn()
        row, pa, _ = evaluate(name, group, yte, s, thr, ate, t1 - t0, t2 - t1)
        rows.append(row); per_attack_all.append(pa)
        np.savez(os.path.join(a.out, f"scores_{name}.npz"), y=yte, score=s, thr=thr)
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
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"baselines_metrics{suffix}.csv"), index=False)
    pd.concat(per_attack_all).to_csv(
        os.path.join(a.out, f"baselines_per_attack{suffix}.csv"), index=False)
    print(f"\nSaved to {a.out}/")


if __name__ == "__main__":
    main()
