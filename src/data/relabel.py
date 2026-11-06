"""Relabel stored states at ALL three skill levels (same candidates, M rollouts each).

The generated data only has labels at the shooter's own skill. This script re-estimates the
success rate of every stored candidate under beginner, intermediate and expert noise, so a
model can be trained with one output per skill (and the teacher's skill dependence can be
measured on every state). The shooter's own-skill label is recomputed too (fresh rollouts),
which doubles as a label-noise check against the stored label.

Finished states are written in chunks of --chunk states to <out>/chunk_XXXX.npz, so the job
is resumable (states already present in a chunk file are skipped). --merge writes <out>/labels3.npz with
cand_label3 (N, K, 3) aligned with the rows of the input states.npz.

Usage:
    python -m src.data.relabel --data data/main/states.npz --out data/main/relabel3 [--pilot 192]
    python -m src.data.relabel --data data/main/states.npz --out data/main/relabel3 --merge
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import yaml

from src.data.dataset import load_states

KEYS = ["pos", "on_table", "hittable", "cand_meta", "cand_params", "cand_mask"]


def _job(args):
    """Worker: (row, (K,3) labels, seconds) for one state."""
    i, st, cfg = args
    from src.sim.game import system_from_state
    from src.sim.shots import BALL_IDS, POCKET_IDS, Candidate, estimate_success, skills_from_config

    skills = skills_from_config(cfg["skills"])
    system = system_from_state(st["pos"], st["on_table"])
    hit = tuple(b for b, h in zip(BALL_IDS, st["hittable"]) if h > 0.5)
    n = int(st["cand_mask"].sum())
    cands = [Candidate(BALL_IDS[st["cand_meta"][k, 0]], POCKET_IDS[st["cand_meta"][k, 1]], int(st["cand_meta"][k, 2]),
                       *map(float, st["cand_params"][k]), feats=(), heuristic=0.0) for k in range(n)]
    out = np.zeros((len(st["cand_mask"]), 3), np.float32)
    t0 = time.time()
    for s, skill in enumerate(skills):
        rng = np.random.default_rng([cfg["seed"], i, s])
        out[:n, s] = estimate_success(system, cands, hit, skill, cfg["M"], rng)
    return i, out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--n_workers", type=int, default=96)
    ap.add_argument("--pilot", type=int, default=0, help="time this many random states and exit")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load_states(a.data)
    N = len(d["win"])
    data_cfg = yaml.safe_load(open(os.path.join(os.path.dirname(a.data), "config.yaml")))
    cfg = {"skills": data_cfg["skills"], "M": a.M, "seed": a.seed}

    if a.merge:
        lab = np.full(d["cand_label"].shape + (3,), np.nan, np.float32)
        secs = np.full(N, np.nan)
        for f in sorted(glob.glob(os.path.join(a.out, "chunk_*.npz"))):
            z = np.load(f)
            lab[z["idx"]] = z["label3"]
            secs[z["idx"]] = z["seconds"]
        assert not np.isnan(lab).any(), f"{np.isnan(secs).sum()} states missing"
        np.savez_compressed(os.path.join(a.out, "labels3.npz"), cand_label3=lab, seconds=secs)
        json.dump({**cfg, "n_states": N, "cpu_hours": float(secs.sum() / 3600)},
                  open(os.path.join(a.out, "meta.json"), "w"), indent=2, sort_keys=False)
        print("merged", lab.shape, "cpu hours", secs.sum() / 3600)
        return

    if a.pilot:
        idx = np.sort(np.random.default_rng(0).choice(N, a.pilot, replace=False))
        t0 = time.time()
        with Pool(a.n_workers) as p:
            res = p.map(_job, [(int(i), {k: d[k][i] for k in KEYS}, cfg) for i in idx])
        wall = time.time() - t0
        secs = np.array([r[2] for r in res])
        est = {"pilot_states": a.pilot, "wall_s": wall, "cpu_s_per_state_mean": float(secs.mean()),
               "cpu_s_per_state_p99": float(np.percentile(secs, 99)), "n_workers": a.n_workers,
               "est_full_hours": float(secs.mean() * N / a.n_workers / 3600)}
        json.dump(est, open(os.path.join(a.out, "pilot.json"), "w"), indent=2)
        print(est)
        return

    done_idx = set()
    for f in glob.glob(os.path.join(a.out, "chunk_*.npz")):
        done_idx.update(np.load(f)["idx"].tolist())
    todo = [i for i in range(N) if i not in done_idx]
    c = len(glob.glob(os.path.join(a.out, "chunk_*.npz")))
    print(f"{N} states, {len(done_idx)} done, {len(todo)} to do", flush=True)
    t0 = time.time()
    buf, done = [], 0

    def flush(buf, c):
        np.savez(os.path.join(a.out, f"chunk_{c:04d}.npz"), idx=np.array([r[0] for r in buf]),
                 label3=np.stack([r[1] for r in buf]), seconds=np.array([r[2] for r in buf]))

    with Pool(a.n_workers) as p:
        jobs = ((int(i), {k: d[k][i] for k in KEYS}, cfg) for i in todo)
        for r in p.imap_unordered(_job, jobs, chunksize=1):
            buf.append(r)
            done += 1
            if len(buf) == a.chunk:
                flush(buf, c)
                c, buf = c + 1, []
                el = (time.time() - t0) / 3600
                print(f"{done}/{len(todo)} states | {el:.2f} h | ETA {el / done * (len(todo) - done):.2f} h", flush=True)
    if buf:
        flush(buf, c)
    print("finished", (time.time() - t0) / 3600, "h", flush=True)

if __name__ == "__main__":
    main()
