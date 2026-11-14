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
    """BCE on the eventual winner + masked soft-label BCE on candidate success rates.

    A (B,K) shot logit is trained on the stored label at the shooter's skill; a (B,K,3) logit
    (three-output head) on the labels at all three skills (``cand_label3``).
    """
    l_win = F.binary_cross_entropy_with_logits(win_logit, b["win"])
    if shot_logit.dim() == 3:
        m = b["cand_mask"][..., None].expand_as(shot_logit)
        target = b["cand_label3"]
    else:
        m, target = b["cand_mask"], b["cand_label"]
    l_shot = (F.binary_cross_entropy_with_logits(shot_logit, target, reduction="none") * m).sum() / m.sum()
    return l_win, l_shot


def rank_loss(shot_logit, b, tau: float):
    """Listwise softmax ranking loss (ListNet top-one): cross-entropy between the target
    distribution softmax(label / tau) and softmax(logit) over each state's valid candidates."""
    neg = torch.finfo(shot_logit.dtype).min / 2
    m = b["cand_mask"] > 0
    target = torch.softmax(torch.where(m, b["cand_label"] / tau, torch.full_like(shot_logit, neg)), 1)
    logp = torch.log_softmax(torch.where(m, shot_logit, torch.full_like(shot_logit, neg)), 1)
    return -(target * torch.where(m, logp, torch.zeros_like(logp))).sum(1).mean()


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
    """Return (win_prob, shot_prob) numpy arrays for every row of tensor dict ``t``.

    shot_prob is (N,K), or (N,K,3) for a three-output shot head.
    """
    model.eval()
    n = len(t["win"])
    wins, shots = [], []
    for i in range(0, n, bs):
        ids = torch.arange(i, min(i + bs, n), device=t["win"].device)
        w, s = model(batch(t, ids))
        wins.append(torch.sigmoid(w).cpu())
        shots.append(torch.sigmoid(s).cpu())
    return torch.cat(wins).numpy(), torch.cat(shots).numpy()


def shot_at_skill(model, t, s: int, ps=None) -> np.ndarray:
    """(N,K) predicted success at skill ``s``: the s-th output of a three-output head, otherwise
    the single output with the shooter's skill input forced to ``s``."""
    if ps is not None and ps.ndim == 3:
        return ps[..., s]
    t2 = dict(t)
    t2["skill_self"] = torch.full_like(t["skill_self"], s)
    return predict(model, t2)[1]


def evaluate(model, t, all_skills: bool = False) -> dict:
    """Win-probability and shot-head metrics on a tensor split.

    shot_* metrics compare the prediction at the shooter's own skill with the stored label.
    If ``all_skills`` and the split has ``cand_label3``, shot3_<skill>_* metrics compare the
    prediction at each skill with the relabeled success rates at that skill.
    """
    pw, ps = predict(model, t)
    y = t["win"].cpu().numpy()
    eps = 1e-6
    m = t["cand_mask"].cpu().numpy()
    lab = t["cand_label"].cpu().numpy()
    late = t["shots_to_end"].cpu().numpy() < 3
    if ps.ndim == 3:
        own = t["skill_self"].cpu().numpy()
        ps_own = np.take_along_axis(ps, own[:, None, None].repeat(ps.shape[1], 1), 2)[..., 0]
    else:
        ps_own = ps
    out = {
        "win_logloss": float(-np.mean(y * np.log(pw + eps) + (1 - y) * np.log(1 - pw + eps))),
        "win_brier": float(np.mean((pw - y) ** 2)),
        "win_ece": ece(pw, y),
        "win_acc": float(((pw > 0.5) == y).mean()),
        "win_acc_last3": float(((pw[late] > 0.5) == y[late]).mean()),
        "shot_mse": float((((ps_own - lab) ** 2) * m).sum() / m.sum()),
    }
    out.update({f"shot_{k}": v for k, v in shot_metrics(ps_own, lab, m).items()})
    if ps.ndim == 3:  # training target of the three-output head
        lab3 = t["cand_label3"].cpu().numpy()
        out["shot3_mse"] = float((((ps - lab3) ** 2) * m[..., None]).sum() / (3 * m.sum()))
    if all_skills and "cand_label3" in t:
        lab3 = t["cand_label3"].cpu().numpy()
        for s, name in enumerate(["beginner", "intermediate", "expert"]):
            p_s = shot_at_skill(model, t, s, ps)
            out[f"shot3_{name}_mse"] = float((((p_s - lab3[..., s]) ** 2) * m).sum() / m.sum())
            out.update({f"shot3_{name}_{k}": v for k, v in shot_metrics(p_s, lab3[..., s], m).items()})
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
    ap.add_argument("--labels3", default=None, help="labels3.npz from src.data.relabel (all-skill labels)")
    ap.add_argument("--shot_outputs", type=int, default=None, help="3 = one shot output per skill (needs labels3)")
    ap.add_argument("--shot_label", default=None, choices=["stored", "m32"],
                    help="m32: average of the stored label and the relabel3 label at the shooter's skill (needs labels3)")
    ap.add_argument("--rank_weight", type=float, default=None, help="weight of the listwise ranking loss (0 = off)")
    ap.add_argument("--rank_tau", type=float, default=None, help="target softmax temperature of the ranking loss")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for k in ["max_train_states", "use_skill", "seed", "data", "epochs", "labels3", "shot_outputs", "shot_label", "rank_weight", "rank_tau"]:
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    cfg.setdefault("max_train_states", None)
    cfg["use_skill"] = bool(cfg.get("use_skill", True))
    run = a.tag or cfg["name"]
    out_dir = os.path.join("results", run)
    os.makedirs(out_dir, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out_dir, "config.yaml"), "w"), sort_keys=False)
    set_seed(cfg["seed"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.setdefault("labels3", None)
    cfg["shot_outputs"] = int(cfg.get("shot_outputs", 1))
    assert cfg["shot_outputs"] == 1 or cfg["labels3"], "three-output head needs --labels3"
    cfg["shot_label"] = cfg.get("shot_label") or "stored"
    cfg["rank_weight"] = float(cfg.get("rank_weight") or 0.0)
    cfg["rank_tau"] = float(cfg.get("rank_tau") or 0.05)
    d = load_states(cfg["data"], cfg["labels3"])
    if cfg["shot_label"] == "m32":
        # two independent M=16 estimates at the shooter's own skill -> one M=32 estimate
        assert cfg["labels3"], "--shot_label m32 needs --labels3"
        own = d["cand_label3"][np.arange(len(d["win"])), :, d["skill_self"]]
        d["cand_label"] = (0.5 * (d["cand_label"] + own)).astype(np.float32)
    sp = split_masks(d["game_id"])
    tr_mask = subsample_games(d["game_id"], sp["train"], cfg["max_train_states"], cfg["seed"])
    T = {k: to_tensors(d, np.where(m)[0], dev) for k, m in [("train", tr_mask), ("val", sp["val"]), ("test", sp["test"])]}
    print({k: len(v["win"]) for k, v in T.items()}, flush=True)

    cand_f = T["train"]["cand_feats"].shape[-1]
    model = build_model(cfg["model"], cand_f, use_skill=cfg["use_skill"], shot_outputs=cfg["shot_outputs"],
                        **cfg.get("model_kwargs", {})).to(dev)
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
            win_logit, shot_logit = model(b)
            lw, ls = losses(win_logit, shot_logit, b)
            loss = lw + cfg["shot_weight"] * ls
            if cfg["rank_weight"] > 0:
                loss = loss + cfg["rank_weight"] * rank_loss(shot_logit, b, cfg["rank_tau"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += [lw.item(), ls.item()]
        tot /= steps_per_epoch
        val = evaluate(model, T["val"])
        # model selection: win log loss + shot mse (both are "lower is better")
        # (three-output head: mse against its own training target, the all-skill labels)
        sel = val["win_logloss"] + cfg["shot_weight"] * 10 * val.get("shot3_mse", val["shot_mse"])
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
    test = evaluate(model, T["test"], all_skills=True)
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
    extra = {}
    if "cand_label3" in T["test"]:  # prediction at every skill (forced input or per-skill output)
        extra["shot_prob3"] = np.stack([shot_at_skill(model, T["test"], s, ps) for s in range(3)], -1)
    np.savez(os.path.join(out_dir, "test_predictions.npz"), win_prob=pw, shot_prob=ps,
             idx=np.where(sp["test"])[0], **extra)
    print("TEST", json.dumps(test, indent=1))


if __name__ == "__main__":
    main()
