"""Does a wider candidate set make the best shot more skill-dependent? (teacher only)

For the 300 shot-evaluation test states, compare two candidate sets:
  * current: the stored candidates (8 easiest ball-pocket pairs x 3 speeds, center ball)
  * wide:    ALL viable ball-pocket pairs (no ball in the way, no jaw, cut <= 85 deg)
             x 3 speeds x 3 spins (draw b=-0.3, center, follow b=+0.3)
Every candidate is simulated M=64 times at each of the three skills, as two independent halves
of 32 rollouts (A and B). Picking the best shot on half A and scoring it on half B gives
unbiased success rates (no winner's curse), and A-vs-B argmax disagreement at the SAME skill is
the noise floor for "the best shot changes with skill".

Usage:
    python -m src.wide_candidates --data data/main/states.npz --pilot 16      # runtime estimate
    python -m src.wide_candidates --data data/main/states.npz                  # full run
    python -m src.wide_candidates --data data/main/states.npz --analyse        # numbers only
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import yaml

OUT = "results/wide_candidates"
SPINS = (-0.3, 0.0, 0.3)  # b offset: draw, center, follow
SK = ["beginner", "intermediate", "expert"]


def eval_states(data: str, n_states: int, seed: int = 0) -> np.ndarray:
    """The same 300 test states as src.evaluate (same seed and sampling)."""
    from src.data.dataset import split_masks

    gid = np.load(data)["game_id"]
    test = np.where(split_masks(gid)["test"])[0]
    return np.sort(np.random.default_rng(seed).choice(test, size=min(n_states, len(test)), replace=False))


def _job(args):
    """Worker: half-A and half-B hit counts for both candidate sets at all skills."""
    i, st, cfg = args
    from dataclasses import replace

    from src.sim.game import system_from_state
    from src.sim.shots import (BALL_IDS, POCKET_IDS, Candidate, add_noise, execute, generate_candidates,
                               is_success, skills_from_config)

    t0 = time.time()
    skills = skills_from_config(cfg["skills"])
    system = system_from_state(st["pos"], st["on_table"])
    hit = tuple(b for b, h in zip(BALL_IDS, st["hittable"]) if h > 0.5)
    n = int(st["cand_mask"].sum())
    cur = [Candidate(BALL_IDS[st["cand_meta"][k, 0]], POCKET_IDS[st["cand_meta"][k, 1]], int(st["cand_meta"][k, 2]),
                     *map(float, st["cand_params"][k]), feats=(), heuristic=0.0) for k in range(n)]
    allc = generate_candidates(system, hit, cfg["speeds"], top_k=10_000)
    viable = [c for c in allc if c.feats[3] > 0.5]
    base = viable if viable else cur  # no viable pair: fall back to the current set
    wide = [replace(c, b=b) for c in base for b in SPINS]
    half = cfg["M"] // 2
    res = {}
    for name, cands in [("cur", cur), ("wide", wide)]:
        hits = np.zeros((len(cands), 3, 2), np.int16)
        for s, skill in enumerate(skills):
            for h in range(2):
                rng = np.random.default_rng([cfg["seed"], i, s, h, name == "wide"])
                for k, c in enumerate(cands):
                    hits[k, s, h] = sum(is_success(execute(system, *add_noise(c, skill, rng)), c.ball, c.pocket, hit)
                                        for _ in range(half))
        res[name] = {"hits": hits, "ball_pocket": np.array([[BALL_IDS.index(c.ball), POCKET_IDS.index(c.pocket)]
                                                            for c in cands]),
                     "speed": np.array([c.speed_idx for c in cands]), "spin": np.array([c.b for c in cands])}
    return i, res, len(viable), time.time() - t0


def analyse(res: dict, M: int) -> dict:
    """Cross-skill loss and ball/pocket disagreement vs. the noise floor, per candidate set."""
    half = M // 2
    out = {}
    for name in ["cur", "wide"]:
        succ_b, n_cands = [], []
        loss = np.zeros((3, 3))
        diff_bp = {f"{SK[x]}-{SK[y]}": [] for x, y in itertools.combinations(range(3), 2)}
        diff_any = {k: [] for k in diff_bp}
        floor_bp, floor_any = [], []
        for i, r in res.items():
            h = r[name]["hits"].astype(float) / half  # (K,3,2)
            bp = r[name]["ball_pocket"]
            n_cands.append(len(h))
            best_a = h[:, :, 0].argmax(0)  # pick on half A, per skill
            # success of skill X playing skill Y's best shot, scored on half B
            for x in range(3):
                for y in range(3):
                    loss[x, y] += h[best_a[x], x, 1] - h[best_a[y], x, 1]
            for x, y in itertools.combinations(range(3), 2):
                diff_bp[f"{SK[x]}-{SK[y]}"].append(tuple(bp[best_a[x]]) != tuple(bp[best_a[y]]))
                diff_any[f"{SK[x]}-{SK[y]}"].append(best_a[x] != best_a[y])
            best_b = h[:, :, 1].argmax(0)
            for s in range(3):  # noise floor: same skill, half A vs half B
                floor_bp.append(tuple(bp[best_a[s]]) != tuple(bp[best_b[s]]))
                floor_any.append(best_a[s] != best_b[s])
            succ_b.append([h[best_a[s], s, 1] for s in range(3)])
        n = len(res)
        out[name] = {
            "n_states": n, "mean_candidates": float(np.mean(n_cands)),
            "own_best_success_halfB": np.mean(succ_b, 0).tolist(),
            "success_lost": (loss / n).tolist(),  # [exec X][best of Y], positive = lost
            "frac_ball_or_pocket_differs": {k: float(np.mean(v)) for k, v in diff_bp.items()},
            "frac_best_differs": {k: float(np.mean(v)) for k, v in diff_any.items()},
            "noise_floor_ball_or_pocket": float(np.mean(floor_bp)),
            "noise_floor_best": float(np.mean(floor_any)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_states", type=int, default=300)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_workers", type=int, default=96)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    raw = os.path.join(OUT, "raw.npz")
    if a.analyse:
        z = np.load(raw, allow_pickle=True)
        res = z["res"].item()
        out = analyse(res, int(z["M"]))
        json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
        print(json.dumps(out, indent=1))
        return
    d = dict(np.load(a.data))
    data_cfg = yaml.safe_load(open(os.path.join(os.path.dirname(a.data), "config.yaml")))
    cfg = {"skills": data_cfg["skills"], "speeds": data_cfg["game"]["speeds"], "M": a.M, "seed": a.seed}
    idx = eval_states(a.data, a.n_states)
    if a.pilot:
        idx = np.random.default_rng(1).choice(idx, a.pilot, replace=False)
    keys = ["pos", "on_table", "hittable", "cand_meta", "cand_params", "cand_mask"]
    jobs = [(int(i), {k: d[k][i] for k in keys}, cfg) for i in idx]
    t0 = time.time()
    res, nv, secs = {}, {}, {}
    with Pool(min(a.n_workers, len(jobs))) as p:
        for j, (i, r, n_viable, sec) in enumerate(p.imap_unordered(_job, jobs, chunksize=1)):
            res[i], nv[i], secs[i] = r, n_viable, sec
            if (j + 1) % 25 == 0:
                print(f"[{j+1}/{len(jobs)}] {(time.time()-t0)/60:.1f} min", flush=True)
    wall = time.time() - t0
    s = np.array(list(secs.values()))
    info = {"n_states": len(idx), "wall_min": wall / 60, "cpu_s_per_state_mean": float(s.mean()),
            "cpu_s_per_state_max": float(s.max()), "mean_viable_pairs": float(np.mean(list(nv.values()))),
            "est_full_hours_96_workers": float(s.mean() * a.n_states / a.n_workers / 3600)}
    if a.pilot:
        json.dump(info, open(os.path.join(OUT, "pilot.json"), "w"), indent=2)
        print(info)
        return
    np.savez(raw, res=np.array(res, dtype=object), M=a.M, idx=idx)
    out = {"run": info, **analyse(res, a.M)}
    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
