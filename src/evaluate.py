"""Shot-recommendation evaluation with fresh simulator rollouts.

For N test states and every skill level s, each method picks one candidate shot:
  * teacher:   simulator search, argmax of M noisy rollouts per candidate at skill s (timed)
  * heuristic: straightest, shortest candidate at medium speed
  * random:    uniformly random candidate
  * <model>:   argmax of a trained model's shot head, conditioned on skill s
The picked shot is then executed R times with skill-s noise (independent rollouts) and we
report the success rate (called ball in called pocket, no foul).

Usage:
    python -m src.evaluate --data data/<set>/states.npz --runs results/a results/b \
        --n_states 300 --out results/shot_eval_<tag>
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from src.data.dataset import load_states, split_masks, to_tensors
from src.models.heuristic import heuristic_scores
from src.models.nets import build_model
from src.train import predict

SKILL_NAMES = ["beginner", "intermediate", "expert"]


def model_picks(run_dir: str, d: dict, idx: np.ndarray, dev: str) -> np.ndarray:
    """(3, N) candidate index picked by a trained model for each forced skill level."""
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=dev)
    cfg = ck["cfg"]
    model = build_model(cfg["model"], ck["cand_f"], use_skill=cfg["use_skill"], **cfg.get("model_kwargs", {})).to(dev)
    model.load_state_dict(ck["model"])
    t = to_tensors(d, idx, dev)
    out = []
    for s in range(3):
        t["skill_self"] = torch.full_like(t["skill_self"], s)
        _, ps = predict(model, t)
        ps = np.where(t["cand_mask"].cpu().numpy() > 0, ps, -1.0)
        out.append(ps.argmax(1))
    return np.stack(out)


def _eval_state(args):
    """Worker: teacher search + fresh rollouts for every method's pick, all skill levels."""
    i, st, picks, cfg = args
    from src.sim.game import system_from_state
    from src.sim.shots import BALL_IDS, POCKET_IDS, Candidate, estimate_success, skills_from_config

    skills = skills_from_config(cfg["skills"])
    system = system_from_state(st["pos"], st["on_table"])
    hittable = tuple(b for b, h in zip(BALL_IDS, st["hittable"]) if h > 0.5)
    n = int(st["cand_mask"].sum())
    cands = [
        Candidate(BALL_IDS[st["cand_meta"][k, 0]], POCKET_IDS[st["cand_meta"][k, 1]], int(st["cand_meta"][k, 2]),
                  *map(float, st["cand_params"][k]), feats=(), heuristic=0.0)
        for k in range(n)
    ]
    rng = np.random.default_rng(cfg["seed"] * 7919 + i)
    rows = []
    for s, skill in enumerate(skills):
        t0 = time.time()
        p_teacher = estimate_success(system, cands, hittable, skill, cfg["teacher_rollouts"], rng)
        t_teacher = time.time() - t0
        method_pick = {m: int(p[s]) for m, p in picks.items()}
        method_pick["teacher"] = int(np.argmax(p_teacher))
        method_pick["random"] = int(rng.integers(n))
        cache = {}
        for m, k in method_pick.items():
            if k not in cache:
                cache[k] = float(estimate_success(system, [cands[k]], hittable, skill, cfg["eval_rollouts"], rng)[0])
            rows.append({"state": i, "skill": SKILL_NAMES[s], "method": m, "pick": k, "success": cache[k],
                         "teacher_est": float(p_teacher[k]), "teacher_seconds": t_teacher,
                         "is_teacher_pick": k == method_pick["teacher"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--n_states", type=int, default=300)
    ap.add_argument("--teacher_rollouts", type=int, default=16)
    ap.add_argument("--eval_rollouts", type=int, default=64)
    ap.add_argument("--n_workers", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import yaml

    data_cfg = yaml.safe_load(open(os.path.join(os.path.dirname(a.data), "config.yaml")))
    d = load_states(a.data)
    test = np.where(split_masks(d["game_id"])["test"])[0]
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(test, size=min(a.n_states, len(test)), replace=False))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    picks = {"heuristic": np.tile(heuristic_scores(d["cand_heur"][idx], d["cand_meta"][idx]).argmax(1), (3, 1))}
    for r in a.runs:
        picks[os.path.basename(r.rstrip("/"))] = model_picks(r, d, idx, dev)

    cfg = {"skills": data_cfg["skills"], "teacher_rollouts": a.teacher_rollouts, "eval_rollouts": a.eval_rollouts, "seed": a.seed}
    keys = ["pos", "on_table", "hittable", "cand_meta", "cand_params", "cand_mask"]
    jobs = [(int(i), {k: d[k][i] for k in keys}, {m: p[:, j] for m, p in picks.items()}, cfg) for j, i in enumerate(idx)]
    rows = []
    t0 = time.time()
    with Pool(a.n_workers) as pool:
        for j, r in enumerate(pool.imap_unordered(_eval_state, jobs)):
            rows += r
            if (j + 1) % max(1, len(jobs) // 10) == 0:
                print(f"[{j+1}/{len(jobs)}] {(time.time()-t0)/60:.1f} min", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "per_state.csv"), index=False)
    g = df.groupby(["method", "skill"])["success"]
    summ = pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n": g.count()}).reset_index()
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    tt = df.drop_duplicates(["state", "skill"])["teacher_seconds"]
    meta = {"n_states": len(idx), "teacher_rollouts": a.teacher_rollouts, "eval_rollouts": a.eval_rollouts,
            "teacher_seconds_per_state_mean": float(tt.mean()), "wall_minutes": (time.time() - t0) / 60,
            "runs": a.runs}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print(summ.pivot(index="method", columns="skill", values="mean").round(3))
    print(meta)


if __name__ == "__main__":
    main()
