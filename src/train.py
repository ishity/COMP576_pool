"""Train a two-headed PoolCoach model.

Usage:
    python -m src.train --config configs/train_set_transformer.yaml [--max_train_states N]
                        [--use_skill 0|1] [--seed S] [--tag NAME]

Writes to results/<run_name>/: config.yaml, metrics.csv (per epoch), best.pt,
test_metrics.json, test_predictions.npz, and TensorBoard logs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import batch, load_states, split_masks, subsample_games, to_tensors
from src.models.nets import build_model


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def losses(win_logit, shot_logit, b):
    """BCE on the eventual winner + masked soft-label BCE on candidate success rates."""
    l_win = F.binary_cross_entropy_with_logits(win_logit, b["win"])
    m = b["cand_mask"]
    l_shot = (F.binary_cross_entropy_with_logits(shot_logit, b["cand_label"], reduction="none") * m).sum() / m.sum()
    return l_win, l_shot


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for k in range(n_bins):
        sel = bins == k
        if sel.any():
            e += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(e)


def shot_metrics(score: np.ndarray, label: np.ndarray, mask: np.ndarray) -> dict:
    """Agreement of a candidate scorer with the teacher labels.

    top1: argmax matches teacher argmax; teacher_value: teacher success estimate of the
    picked candidate (higher is better); regret: teacher max minus that value.
    """
    s = np.where(mask > 0, score, -np.inf)
    pick = s.argmax(1)
    lab = np.where(mask > 0, label, -np.inf)
    best = lab.max(1)
    val = label[np.arange(len(pick)), pick]
    return {
        "top1": float((pick == lab.argmax(1)).mean()),
        "teacher_value": float(val.mean()),
        "regret": float((best - val).mean()),
    }


@torch.no_grad()
def predict(model, t, bs=1024):
    """Return (win_prob, shot_prob) numpy arrays for every row of tensor dict ``t``."""
    model.eval()
    n = len(t["win"])
    wins, shots = [], []
    for i in range(0, n, bs):
        ids = torch.arange(i, min(i + bs, n), device=t["win"].device)
        w, s = model(batch(t, ids))
        wins.append(torch.sigmoid(w).cpu())
        shots.append(torch.sigmoid(s).cpu())
    return torch.cat(wins).numpy(), torch.cat(shots).numpy()


def evaluate(model, t) -> dict:
    """Win-probability and shot-head metrics on a tensor split."""
    pw, ps = predict(model, t)
    y = t["win"].cpu().numpy()
    eps = 1e-6
    m = t["cand_mask"].cpu().numpy()
    lab = t["cand_label"].cpu().numpy()
    late = t["shots_to_end"].cpu().numpy() < 3
    out = {
        "win_logloss": float(-np.mean(y * np.log(pw + eps) + (1 - y) * np.log(1 - pw + eps))),
        "win_brier": float(np.mean((pw - y) ** 2)),
        "win_ece": ece(pw, y),
        "win_acc": float(((pw > 0.5) == y).mean()),
        "win_acc_last3": float(((pw[late] > 0.5) == y[late]).mean()),
        "shot_mse": float((((ps - lab) ** 2) * m).sum() / m.sum()),
    }
    out.update({f"shot_{k}": v for k, v in shot_metrics(ps, lab, m).items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_train_states", type=int, default=None)
    ap.add_argument("--use_skill", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for k in ["max_train_states", "use_skill", "seed", "data", "epochs"]:
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    cfg.setdefault("max_train_states", None)
    cfg["use_skill"] = bool(cfg.get("use_skill", True))
    run = a.tag or cfg["name"]
    out_dir = os.path.join("results", run)
    os.makedirs(out_dir, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out_dir, "config.yaml"), "w"))
    set_seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = load_states(cfg["data"])
    sp = split_masks(d["game_id"])
    tr_mask = subsample_games(d["game_id"], sp["train"], cfg["max_train_states"], cfg["seed"])
    T = {k: to_tensors(d, np.where(m)[0], dev) for k, m in [("train", tr_mask), ("val", sp["val"]), ("test", sp["test"])]}
    print({k: len(v["win"]) for k, v in T.items()}, flush=True)

    cand_f = T["train"]["cand_feats"].shape[-1]
    model = build_model(cfg["model"], cand_f, use_skill=cfg["use_skill"], **cfg.get("model_kwargs", {})).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n_tr = len(T["train"]["win"])
    steps_per_epoch = max(1, n_tr // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * steps_per_epoch, pct_start=0.05)
    tb = SummaryWriter(os.path.join(out_dir, "tb"))
    f = open(os.path.join(out_dir, "metrics.csv"), "w", newline="")
    writer = None
    best, best_ep = np.inf, -1
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n_tr, device=dev)
        tot = np.zeros(2)
        for i in range(steps_per_epoch):
            b = batch(T["train"], perm[i * cfg["batch_size"] : (i + 1) * cfg["batch_size"]])
            lw, ls = losses(*model(b), b)
            loss = lw + cfg["shot_weight"] * ls
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += [lw.item(), ls.item()]
        tot /= steps_per_epoch
        val = evaluate(model, T["val"])
        # model selection: win log loss + shot mse (both are "lower is better")
        sel = val["win_logloss"] + cfg["shot_weight"] * 10 * val["shot_mse"]
        row = {"epoch": ep, "train_win_bce": tot[0], "train_shot_bce": tot[1], **{f"val_{k}": v for k, v in val.items()},
               "select": sel, "time_min": (time.time() - t0) / 60}
        if writer is None:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        for k, v in row.items():
            tb.add_scalar(k, v, ep)
        if sel < best:
            best, best_ep = sel, ep
            torch.save({"model": model.state_dict(), "cfg": cfg, "cand_f": cand_f}, os.path.join(out_dir, "best.pt"))
        print(f"ep {ep:3d} win_bce {tot[0]:.4f} shot_bce {tot[1]:.4f} | val logloss {val['win_logloss']:.4f} "
              f"brier {val['win_brier']:.4f} shot_mse {val['shot_mse']:.4f} top1 {val['shot_top1']:.3f} "
              f"regret {val['shot_regret']:.4f}", flush=True)

    ck = torch.load(os.path.join(out_dir, "best.pt"), map_location=dev)
    model.load_state_dict(ck["model"])
    test = evaluate(model, T["test"])
    test.update({"best_epoch": best_ep, "n_train_states": n_tr, "n_params": n_params,
                 "train_minutes": (time.time() - t0) / 60})
    # inference speed (batch of 1, GPU) for the speed comparison
    b1 = batch(T["test"], torch.arange(1, device=dev))
    with torch.no_grad():
        for _ in range(10):
            model(b1)
        torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(100):
            model(b1)
        torch.cuda.synchronize()
    test["infer_ms_per_state"] = (time.time() - t1) * 10
    json.dump(test, open(os.path.join(out_dir, "test_metrics.json"), "w"), indent=2)
    pw, ps = predict(model, T["test"])
    np.savez(os.path.join(out_dir, "test_predictions.npz"), win_prob=pw, shot_prob=ps,
             idx=np.where(sp["test"])[0])
    print("TEST", json.dumps(test, indent=1))


if __name__ == "__main__":
    main()
