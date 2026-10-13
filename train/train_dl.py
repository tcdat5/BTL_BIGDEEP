"""Train 1 model DL (ae|vae|dae|mlp), đánh giá trên test chung, export ONNX trả thẳng score.

Kaggle: python train/train_dl.py --model ae --data-dir /kaggle/working/data/kaggle_sample
Smoke:  python train/train_dl.py --model ae --smoke      (dữ liệu giả, 2 epoch, CPU)
VAE: early stopping chỉ tính từ epoch --kl-warmup (trước đó β đang tăng nên val loss không so được).
ONNX: opset 17, ir_version <= 8 (onnxruntime cũ trên cụm); so khớp ORT vs PyTorch rtol=atol=1e-4 trên 1000 mẫu.
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
        print("[%s] epoch %3d train %.5f val %.5f (%.0fs)" % (kind, epoch, row["train_loss"], v_loss,
                                                              time.time() - t0), flush=True)
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
        n_before = len(tr.y)
        tr = tr.subset(tr.labels != args.holdout_attack)
        va = va.subset(va.labels != args.holdout_attack)
        if supervised and n_before == len(tr.y):
            print("[warn] '%s' không có trong train" % args.holdout_attack, flush=True)
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
        if kind == "ae":      # streaming/dashboard mặc định dùng models/ae.onnx + models/threshold.json
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
