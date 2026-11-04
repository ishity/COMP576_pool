"""Model 1: geometric heuristic (no deep learning).

Shot: the straightest, shortest candidate (lowest geometric difficulty) at medium speed.
Win probability: logistic regression on ball counts.

Usage:  python -m src.models.heuristic --data data/<set>/states.npz --tag heuristic_<set>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.dataset import load_states, split_masks
from src.train import ece, shot_metrics

MEDIUM_SPEED_IDX = 1


def count_features(d: dict, idx: np.ndarray) -> np.ndarray:
    """[own balls left, opponent balls left, open table, on the 8]."""
    roles = d["roles"][idx] * d["on_table"][idx][..., None]
    hit = d["hittable"][idx]
    on8 = (hit.sum(1) == 1) & (hit[:, 8] == 1)
    return np.stack([roles[:, :, 2].sum(1), roles[:, :, 3].sum(1), roles[:, :, 4].sum(1) > 0, on8], 1).astype(float)


def heuristic_scores(cand_heur: np.ndarray, cand_meta: np.ndarray) -> np.ndarray:
    """Higher = preferred. Only medium-speed candidates are considered."""
    s = -np.where(np.isfinite(cand_heur), cand_heur, 1e9)
    return np.where(cand_meta[..., 2] == MEDIUM_SPEED_IDX, s, -1e12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    d = load_states(a.data)
    sp = split_masks(d["game_id"])
    tr, te = np.where(sp["train"])[0], np.where(sp["test"])[0]
    lr = LogisticRegression(max_iter=1000).fit(count_features(d, tr), d["win"][tr])
    pw = lr.predict_proba(count_features(d, te))[:, 1]
    y = d["win"][te]
    late = d["shots_to_end"][te] < 3
    sc = heuristic_scores(d["cand_heur"][te], d["cand_meta"][te])
    m = d["cand_mask"][te] * (d["cand_meta"][te][..., 2] == MEDIUM_SPEED_IDX)
    out = {
        "win_logloss": float(-np.mean(y * np.log(pw) + (1 - y) * np.log(1 - pw))),
        "win_brier": float(np.mean((pw - y) ** 2)),
        "win_ece": ece(pw, y),
        "win_acc": float(((pw > 0.5) == y).mean()),
        "win_acc_last3": float(((pw[late] > 0.5) == y[late]).mean()),
        "lr_coef": lr.coef_.tolist(), "lr_intercept": lr.intercept_.tolist(),
    }
    # shot metrics are computed against teacher labels over ALL candidates (pick is medium-speed only)
    out.update({f"shot_{k}": v for k, v in shot_metrics(np.where(m > 0, sc, -1e12), d["cand_label"][te], d["cand_mask"][te]).items()})
    os.makedirs(f"results/{a.tag}", exist_ok=True)
    json.dump(out, open(f"results/{a.tag}/test_metrics.json", "w"), indent=2)
    np.savez(f"results/{a.tag}/test_predictions.npz", win_prob=pw, idx=te)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
