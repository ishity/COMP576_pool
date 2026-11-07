"""Mean and std over seeds of test metrics (seed 0 = untagged run, seed s = run tag + "_s<s>").

Usage: python -m src.aggregate_seeds --runs set_transformer mlp ... --seeds 0 1 2 --out results/seeds_summary.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.data.dataset import split_masks

METRICS = ["win_logloss", "win_brier", "win_ece", "win_acc", "shot_mse", "shot_top1", "shot_regret", "best_epoch"]


def run_dir(run: str, seed: int) -> str:
    """Directory of one run at one seed (seed 0 may be untagged, from train_all.sh)."""
    if seed == 0 and os.path.exists(os.path.join("results", run, "test_metrics.json")):
        return os.path.join("results", run)
    return os.path.join("results", f"{run}_s{seed}")


def turnover_frac_up(pred_path: str, d: dict) -> float:
    """Fraction of test turnovers after which the former shooter's predicted win prob went up."""
    te = np.where(split_masks(d["game_id"])["test"])[0]
    z = np.load(pred_path)
    pw = z["win_prob"]
    g, sh = d["game_id"][te], d["shooter"][te]
    turn = (g[1:] == g[:-1]) & (sh[1:] != sh[:-1])
    return float((((1 - pw[1:]) - pw[:-1])[turn] > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--data", default="data/main/states.npz")
    ap.add_argument("--out", default="results/seeds_summary.csv")
    a = ap.parse_args()
    d = dict(np.load(a.data))
    rows = []
    for r in a.runs:
        for s in a.seeds:
            p = run_dir(r, s)
            if not os.path.exists(os.path.join(p, "test_metrics.json")):
                print("missing", p)
                continue
            m = json.load(open(os.path.join(p, "test_metrics.json")))
            row = {"run": r, "seed": s, "n_train_states": m.get("n_train_states", np.nan), **{k: m.get(k, np.nan) for k in METRICS}}
            row["turnover_frac_up"] = turnover_frac_up(os.path.join(p, "test_predictions.npz"), d)
            row.update({k: v for k, v in m.items() if k.startswith("shot3_") and isinstance(v, float)})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(a.out.replace(".csv", "_per_seed.csv"), index=False)
    num = [c for c in df.columns if c not in ("run", "seed")]
    agg = df.groupby("run", sort=False)[num].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg.insert(0, "n_seeds", df.groupby("run", sort=False).size())
    agg.to_csv(a.out)
    show = ["win_logloss", "win_brier", "win_ece", "shot_regret", "shot_top1", "turnover_frac_up"]
    for r, x in agg.iterrows():
        print(f"{r:28s} n={x.n_seeds} " + " ".join(f"{k}={x[k+'_mean']:.3f}±{x[k+'_std']:.3f}" for k in show))


if __name__ == "__main__":
    main()
