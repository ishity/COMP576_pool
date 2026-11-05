"""Generate result figures (PNG + PDF) from logged runs in results/.

Usage:
    python -m src.figures --data data/main/states.npz --shot_eval results/shot_eval_main \
        --win_runs set_transformer cnn mlp deepsets heuristic_main --ablation_prefix size_
Every number plotted is read from results/; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.dataset import split_masks  # noqa: E402

# Reference categorical palette (fixed order, validated; see dataviz skill palette.md)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SKILLS = ["beginner", "intermediate", "expert"]
PRETTY = {
    "teacher": "Simulator search (teacher)",
    "set_transformer": "Set Transformer (ours)",
    "set_transformer_noskill": "Set Transformer, no skill input",
    "cnn": "CNN (ResNet-18, image)",
    "mlp": "MLP (flat set)",
    "deepsets": "DeepSets",
    "heuristic": "Geometric heuristic",
    "random": "Random candidate",
}


def style(ax):
    """Recessive axes: light grid, no top/right spines, secondary-ink ticks."""
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(TEXT2)
    ax.tick_params(colors=TEXT2)


def save(fig, name):
    os.makedirs("figures", exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(f"figures/{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/" + name)


def fig_shot_success(shot_eval: str, methods: list[str], name="fig3_shot_success_by_skill"):
    """Grouped bars: success rate of each method's recommended shot per skill level."""
    s = pd.read_csv(os.path.join(shot_eval, "summary.csv"))
    methods = [m for m in methods if m in set(s.method)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.8 / len(methods)
    for j, m in enumerate(methods):
        r = s[s.method == m].set_index("skill").loc[SKILLS]
        x = np.arange(3) + (j - (len(methods) - 1) / 2) * w
        ax.bar(x, r["mean"], w * 0.9, yerr=r["sem"], color=SERIES[j], label=PRETTY.get(m, m),
               error_kw=dict(ecolor=TEXT2, lw=0.8, capsize=2))
    ax.set_xticks(range(3), [k.capitalize() for k in SKILLS])
    ax.set_ylabel("Success rate of recommended shot")
    ax.set_ylim(0, 1)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    n = json.load(open(os.path.join(shot_eval, "meta.json")))
    ax.set_title(f"Recommended-shot success under skill noise ({n['n_states']} test states, "
                 f"{n['eval_rollouts']} fresh rollouts each)", fontsize=9, color=TEXT)
    save(fig, name)


def fig_win(data: str, runs: list[str]):
    """Win-probability curve of one test game + reliability diagram for each run."""
    d = dict(np.load(data))
    te = np.where(split_masks(d["game_id"])["test"])[0]
    preds = {}
    for r in runs:
        p = os.path.join("results", r, "test_predictions.npz")
        if os.path.exists(p):
            z = np.load(p)
            assert np.array_equal(z["idx"], te)
            preds[r] = z["win_prob"]
    y = d["win"][te]
    # (a) one example game: longest test game, plotted from player P0's perspective
    gids, cnt = np.unique(d["game_id"][te], return_counts=True)
    g = gids[np.argmax(cnt)]
    sel = d["game_id"][te] == g
    shooter = d["shooter"][te][sel]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for j, (r, p) in enumerate(preds.items()):
        p0 = np.where(shooter == 0, p[sel], 1 - p[sel])
        ax[0].plot(np.arange(sel.sum()), p0, "-o", ms=3, lw=2, color=SERIES[j], label=PRETTY.get(r, r))
    winner0 = y[sel][0] if shooter[0] == 0 else 1 - y[sel][0]
    sk = d["skill_self"][te][sel][shooter == 0][:1].tolist() + d["skill_opp"][te][sel][shooter == 0][:1].tolist()
    ax[0].axhline(0.5, color=TEXT2, lw=0.8, ls="--")
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("Shot index in rack")
    ax[0].set_ylabel("P(player 0 wins)")
    title = f"Test game {g}: P0={SKILLS[sk[0]] if sk else '?'} vs P1={SKILLS[sk[1]] if len(sk)>1 else '?'}, " \
            f"P0 {'won' if winner0 else 'lost'}"
    ax[0].set_title(title, fontsize=9)
    # (b) reliability diagram
    bins = np.linspace(0, 1, 11)
    for j, (r, p) in enumerate(preds.items()):
        b = np.minimum((p * 10).astype(int), 9)
        xs = [p[b == k].mean() for k in range(10) if (b == k).sum() >= 20]
        ys = [y[b == k].mean() for k in range(10) if (b == k).sum() >= 20]
        ax[1].plot(xs, ys, "-o", ms=4, lw=2, color=SERIES[j], label=PRETTY.get(r, r))
    ax[1].plot([0, 1], [0, 1], color=TEXT2, lw=0.8, ls="--")
    ax[1].set_xlabel("Predicted win probability")
    ax[1].set_ylabel("Observed win frequency")
    ax[1].set_title(f"Reliability ({len(y)} test states, bins with n>=20)", fontsize=9)
    for a in ax:
        style(a)
    ax[1].legend(frameon=False, fontsize=7)
    save(fig, "fig4_win_probability")


def fig_ablation(prefix: str, extra=(), name="fig5_data_size_ablation"):
    """Test metrics vs number of training states, from runs named <prefix>* plus ``extra`` runs."""
    rows = []
    paths = glob.glob(f"results/{prefix}*/test_metrics.json") + [f"results/{r}/test_metrics.json" for r in extra]
    for p in paths:
        m = json.load(open(p))
        m["run"] = os.path.basename(os.path.dirname(p))
        rows.append(m)
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("n_train_states")
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    for a, (k, lab) in zip(ax, [("win_logloss", "Test win log loss (lower is better)"),
                                ("shot_regret", "Test shot regret vs teacher (lower is better)")]):
        a.plot(df["n_train_states"], df[k], "-o", color=SERIES[0], lw=2, ms=6)
        a.set_xscale("log")
        a.set_xlabel("Training states")
        a.set_title(lab, fontsize=9)
        style(a)
    save(fig, name)


ROLE_COLORS = ["#ffffff", "#0b0b0b", SERIES[0], SERIES[1], "#a9a8a2"]  # cue, eight, own, opp, open
POCKETS_XY = {0: (0, 0), 1: (0, 0.9906), 2: (0, 1.9812), 3: (0.9906, 0), 4: (0.9906, 0.9906), 5: (0.9906, 1.9812)}


def draw_table(ax, pos, on, roles):
    """Top-down table with balls colored by role relative to the shooter."""
    from matplotlib.patches import Circle, Rectangle
    ax.add_patch(Rectangle((0, 0), 0.9906, 1.9812, fc="#e8f1ea", ec=TEXT2, lw=1))
    for x, y in POCKETS_XY.values():
        ax.add_patch(Circle((x, y), 0.06, fc="#52514e", ec="none"))
    for i in range(16):
        if on[i] > 0.5:
            r = int(np.argmax(roles[i]))
            ax.add_patch(Circle(pos[i], 0.028575, fc=ROLE_COLORS[r], ec=TEXT, lw=0.6, zorder=3))
            if i > 0:
                ax.text(*pos[i], str(i), fontsize=4.5, ha="center", va="center", zorder=4,
                        color="#ffffff" if r in (1, 2, 3) else TEXT)
    ax.set_xlim(-0.08, 1.07)
    ax.set_ylim(-0.08, 2.06)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_examples(data: str, run: str, shot_eval: str, n: int = 4, name="fig6_qualitative"):
    """Test states where the model's beginner and expert recommendations differ.

    Arrows: cue -> ghost ball and object ball -> pocket for each skill's recommended shot.
    Titles report fresh-rollout success rates logged in shot_eval/per_state.csv.
    """
    d = dict(np.load(data))
    df = pd.read_csv(os.path.join(shot_eval, "per_state.csv"))
    sub = df[df.method == run].pivot_table(index="state", columns="skill", values=["pick", "success"])
    diff = sub[sub[("pick", "beginner")] != sub[("pick", "expert")]].index.tolist()[:n]
    if not diff:
        return
    fig, axes = plt.subplots(1, len(diff), figsize=(2.3 * len(diff), 4.2))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, diff):
        draw_table(ax, d["pos"][i], d["on_table"][i], d["roles"][i])
        lines = []
        for sk, col, ls in [("beginner", SERIES[2], "-"), ("expert", SERIES[6], "--")]:
            k = int(sub.loc[i, ("pick", sk)])
            f = d["cand_feats"][i, k]
            gx, gy = f[-3] * 0.9906, f[-2] * 1.9812
            ox, oy = f[-7] * 0.9906, f[-6] * 1.9812
            px, py = f[-5] * 0.9906, f[-4] * 1.9812
            cx, cy = d["pos"][i][0]
            ax.annotate("", (gx, gy), (cx, cy), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, ls=ls), zorder=5)
            ax.annotate("", (px, py), (ox, oy), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, ls=ls), zorder=5)
            spd = ["soft", "med", "firm"][int(d["cand_meta"][i, k, 2])]
            lines.append(f"{sk[:3]}: ball {d['cand_meta'][i,k,0]} {spd}, succ {sub.loc[i, ('success', sk)]:.2f}")
        ax.set_title("\n".join(lines), fontsize=6, color=TEXT)
    fig.suptitle(f"{PRETTY.get(run, run)}: beginner (solid) vs expert (dashed) recommendation", fontsize=8)
    save(fig, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--shot_eval", default=None)
    ap.add_argument("--methods", nargs="*", default=list(PRETTY))
    ap.add_argument("--win_runs", nargs="*", default=[])
    ap.add_argument("--ablation_prefix", default=None)
    ap.add_argument("--ablation_extra", nargs="*", default=[])
    ap.add_argument("--example_run", default=None)
    a = ap.parse_args()
    if a.shot_eval:
        fig_shot_success(a.shot_eval, a.methods)
    if a.win_runs:
        fig_win(a.data, a.win_runs)
    if a.ablation_prefix:
        fig_ablation(a.ablation_prefix, a.ablation_extra)
    if a.example_run and a.shot_eval:
        fig_examples(a.data, a.example_run, a.shot_eval)


if __name__ == "__main__":
    main()
