"""Measure teacher label noise: re-estimate candidate success with M_ref rollouts.

Usage: python -m src.data.label_noise --data data/main/states.npz --n_states 100 --m_ref 64
Writes results/label_noise/summary.json (MAE/corr between M-rollout labels and M_ref labels,
and how often the M-rollout argmax matches the M_ref argmax).
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import yaml

from src.data.dataset import load_states


def _job(args):
    i, st, cfg = args
    from src.sim.game import system_from_state
    from src.sim.shots import BALL_IDS, POCKET_IDS, Candidate, estimate_success, skills_from_config
    skill = skills_from_config(cfg["skills"])[int(st["skill_self"])]
    system = system_from_state(st["pos"], st["on_table"])
    hit = tuple(b for b, h in zip(BALL_IDS, st["hittable"]) if h > 0.5)
    n = int(st["cand_mask"].sum())
    cands = [Candidate(BALL_IDS[st["cand_meta"][k, 0]], POCKET_IDS[st["cand_meta"][k, 1]], int(st["cand_meta"][k, 2]),
                       *map(float, st["cand_params"][k]), feats=(), heuristic=0.0) for k in range(n)]
    ref = estimate_success(system, cands, hit, skill, cfg["m_ref"], np.random.default_rng(10_000 + i))
    return st["cand_label"][:n], ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_states", type=int, default=100)
    ap.add_argument("--m_ref", type=int, default=64)
    ap.add_argument("--n_workers", type=int, default=96)
    a = ap.parse_args()
    d = load_states(a.data)
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(a.data), "config.yaml")))
    cfg["m_ref"] = a.m_ref
    idx = np.random.default_rng(1).choice(len(d["win"]), a.n_states, replace=False)
    keys = ["pos", "on_table", "hittable", "cand_meta", "cand_params", "cand_mask", "cand_label", "skill_self"]
    with Pool(a.n_workers) as p:
        res = p.map(_job, [(int(i), {k: d[k][i] for k in keys}, cfg) for i in idx])
    lab = np.concatenate([r[0] for r in res])
    ref = np.concatenate([r[1] for r in res])
    out = {"M": int(cfg["game"]["n_rollouts"]), "M_ref": a.m_ref, "n_states": a.n_states, "n_candidates": int(len(lab)),
           "mae": float(np.abs(lab - ref).mean()), "corr": float(np.corrcoef(lab, ref)[0, 1]),
           "argmax_agree": float(np.mean([r[0].argmax() == r[1].argmax() for r in res])),
           "regret_of_M_argmax_under_ref": float(np.mean([r[1].max() - r[1][r[0].argmax()] for r in res]))}
    os.makedirs("results/label_noise", exist_ok=True)
    json.dump(out, open("results/label_noise/summary.json", "w"), indent=2)
    print(out)


if __name__ == "__main__":
    main()
