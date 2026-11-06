"""How much does the teacher's best shot depend on skill? (shot-evaluation states)

Inputs (from src.evaluate): teacher_vectors.npz (M-rollout teacher estimate for every candidate
at every skill) and per_state_cross.csv (fresh R-rollout success of each skill's teacher pick
executed at every skill).

Reports, per pair of skills:
  * fraction of states whose teacher argmax differs (any difference, and different ball or pocket)
  * noise floor: fraction of states whose argmax differs between two independent M-rollout
    estimates at the SAME skill (stored dataset label vs. the evaluation's teacher estimate at
    the shooter's skill), i.e. what pure Monte-Carlo noise produces
  * success lost when a skill-X player plays skill-Y's best shot, from the fresh rollouts:
    success_X(best_Y) - success_X(best_X) (both sides fresh, so no winner's-curse bias), and
    the same quantity from the M-rollout teacher vectors for reference.

Usage:
    python -m src.skill_dependence --data data/main/states.npz --shot_eval results/shot_eval_main_v2
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.figures import SERIES, TEXT, TEXT2, save, style  # noqa: E402

SK = ["beginner", "intermediate", "expert"]


def analyse(d: dict, shot_eval: str) -> dict:
    """All numbers of the skill-dependence analysis (see module docstring)."""
    z = np.load(os.path.join(shot_eval, "teacher_vectors.npz"))
    idx, T, mask = z["state"], z["teacher"], z["cand_mask"]
    Tm = np.where(mask[..., None] > 0, T, -1.0)
    best = Tm.argmax(1)  # (N,3)
    meta = d["cand_meta"][idx]  # (N,K,3): ball, pocket, speed
    n = len(idx)
    out = {"n_states": int(n), "pairs": {}}
    for a, b in itertools.combinations(range(3), 2):
        ba, bb = best[:, a], best[:, b]
        bp_a = meta[np.arange(n), ba, :2]
        bp_b = meta[np.arange(n), bb, :2]
        out["pairs"][f"{SK[a]}-{SK[b]}"] = {
            "frac_best_differs": float((ba != bb).mean()),
            "frac_ball_or_pocket_differs": float((bp_a != bp_b).any(1).mean()),
        }
    # noise floor at the same skill: stored label (at shooter skill) vs this teacher estimate
    own = d["skill_self"][idx]
    stored = np.where(mask > 0, d["cand_label"][idx], -1.0)
    t_own = Tm[np.arange(n), :, own]
    sa, ta = stored.argmax(1), t_own.argmax(1)
    out["noise_floor_same_skill"] = {
        "frac_best_differs": float((sa != ta).mean()),
        "frac_ball_or_pocket_differs": float((meta[np.arange(n), sa, :2] != meta[np.arange(n), ta, :2]).any(1).mean()),
    }
    # success lost, fresh rollouts
    cr = pd.read_csv(os.path.join(shot_eval, "per_state_cross.csv"))
    t = cr[cr.method == "teacher"].pivot_table(index="state", columns=["exec_skill", "cond_skill"], values="success")
    fresh = np.zeros((3, 3))
    fresh_sem = np.zeros((3, 3))
    for x in range(3):
        for y in range(3):
            diff = t[(SK[x], SK[y])] - t[(SK[x], SK[x])]
            fresh[x, y], fresh_sem[x, y] = diff.mean(), diff.sem()
    teach = np.zeros((3, 3))
    for x in range(3):
        for y in range(3):
            teach[x, y] = float((T[np.arange(n), best[:, y], x] - T[np.arange(n), best[:, x], x]).mean())
    out["success_lost_fresh"] = fresh.tolist()  # [exec skill X][best shot of skill Y]
    out["success_lost_fresh_sem"] = fresh_sem.tolist()
    out["success_lost_teacher_M"] = teach.tolist()
    out["own_best_success_fresh"] = [float(t[(SK[x], SK[x])].mean()) for x in range(3)]
    out["n_states_cross"] = int(len(t))
    return out


def figure(res: dict, name="fig_skill_dependence"):
    """Left: how often the best shot changes between skills; right: success lost matrix."""
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.25, 1]})
    pairs = list(res["pairs"])
    x = np.arange(len(pairs))
    w = 0.36
    for j, (k, lab) in enumerate([("frac_best_differs", "Any change (ball, pocket or speed)"),
                                  ("frac_ball_or_pocket_differs", "Different ball or pocket")]):
        v = [res["pairs"][p][k] for p in pairs]
        ax[0].bar(x + (j - 0.5) * w, v, w * 0.92, color=SERIES[j], label=lab)
        for xi, vi in zip(x, v):
            ax[0].text(xi + (j - 0.5) * w, vi + 0.015, f"{vi:.2f}", ha="center", fontsize=7, color=TEXT)
        nf = res["noise_floor_same_skill"][k]
        ax[0].axhline(nf, color=SERIES[j], lw=1, ls="--")
    from matplotlib.lines import Line2D

    h, l = ax[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=TEXT2, lw=1, ls="--"))
    l.append("Noise floor: same skill, two independent\n16-rollout labels (color = measure)")
    ax[0].set_xticks(x, [p.replace("-", " vs\n") for p in pairs], fontsize=8)
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("Fraction of states")
    ax[0].set_title(f"Teacher's best shot changes with skill ({res['n_states']} test states)", fontsize=9, color=TEXT)
    style(ax[0])
    ax[0].legend(h, l, frameon=False, fontsize=7, loc="upper left")

    m = -np.array(res["success_lost_fresh"])
    se = np.array(res["success_lost_fresh_sem"])
    im = ax[1].imshow(m, cmap="Purples", vmin=0, vmax=1.15 * m.max())
    for i in range(3):
        for j in range(3):
            txt = "own best\n" + f"{res['own_best_success_fresh'][i]:.3f}" if i == j else f"{m[i, j]:.3f}\n± {se[i, j]:.3f}"
            ax[1].text(j, i, txt, ha="center", va="center", fontsize=7.5,
                       color="#ffffff" if m[i, j] > 0.6 * im.get_clim()[1] else (TEXT2 if i == j else TEXT))
    ax[1].set_xticks(range(3), [s.capitalize() for s in SK], fontsize=8)
    ax[1].set_yticks(range(3), [s.capitalize() for s in SK], fontsize=8)
    ax[1].set_xlabel("Plays the best shot of a ...", fontsize=8)
    ax[1].set_ylabel("Player executing", fontsize=8)
    ax[1].set_title("Success-rate drop vs. own best shot\n(fresh 64-rollout estimates, mean ± s.e.)", fontsize=9, color=TEXT)
    for s in ax[1].spines.values():
        s.set_visible(False)
    fig.tight_layout()
    save(fig, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--shot_eval", required=True)
    ap.add_argument("--out", default="results/skill_dependence")
    a = ap.parse_args()
    d = dict(np.load(a.data))
    res = analyse(d, a.shot_eval)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))
    figure(res)


if __name__ == "__main__":
    main()
