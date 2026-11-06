"""Simple win-probability baselines on the test split (no shot head).

  prior        : constant = training-set win rate of the current shooter
  lr_skills    : logistic regression on one-hot(skill_self) + one-hot(skill_opp)
  lr_skills_counts : the same plus the ball-count features of the heuristic baseline
                 (own left, opponent left, open table, on the 8)

Fit on train, report on test. Usage:
    python -m src.models.win_baselines --data data/main/states.npz --out results/win_baselines
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data.dataset import load_states, split_masks
from src.models.heuristic import count_features
from src.train import ece


def skill_features(d: dict, idx: np.ndarray) -> np.ndarray:
    """One-hot shooter skill and one-hot opponent skill (6 columns)."""
    eye = np.eye(3)
    return np.concatenate([eye[d["skill_self"][idx]], eye[d["skill_opp"][idx]]], 1)


def win_metrics(pw: np.ndarray, y: np.ndarray, late: np.ndarray) -> dict:
    """Same win metrics as src.train.evaluate."""
    eps = 1e-6
    return {
        "win_logloss": float(-np.mean(y * np.log(pw + eps) + (1 - y) * np.log(1 - pw + eps))),
        "win_brier": float(np.mean((pw - y) ** 2)),
        "win_ece": ece(pw, y),
        "win_acc": float(((pw > 0.5) == y).mean()),
        "win_acc_last3": float(((pw[late] > 0.5) == y[late]).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/win_baselines")
    a = ap.parse_args()
    d = load_states(a.data)
    sp = split_masks(d["game_id"])
    tr, te = np.where(sp["train"])[0], np.where(sp["test"])[0]
    y, ytr = d["win"][te], d["win"][tr]
    late = d["shots_to_end"][te] < 3
    feats = {
        "lr_skills": skill_features,
        "lr_skills_counts": lambda d, i: np.concatenate([skill_features(d, i), count_features(d, i)], 1),
    }
    res = {"prior": {**win_metrics(np.full(len(te), ytr.mean()), y, late), "train_rate": float(ytr.mean())}}
    preds = {"prior": np.full(len(te), ytr.mean())}
    for name, f in feats.items():
        lr = LogisticRegression(max_iter=2000).fit(f(d, tr), ytr)
        preds[name] = lr.predict_proba(f(d, te))[:, 1]
        res[name] = {**win_metrics(preds[name], y, late), "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist()}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "test_metrics.json"), "w"), indent=2)
    np.savez(os.path.join(a.out, "test_predictions.npz"), idx=te, **preds)
    for k, v in res.items():
        print(k, {m: round(v[m], 4) for m in ["win_logloss", "win_brier", "win_ece", "win_acc", "win_acc_last3"]})


if __name__ == "__main__":
    main()
