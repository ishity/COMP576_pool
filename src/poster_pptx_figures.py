"""Figures sized for the 36 x 24 in poster (pool_coach_poster.pptx), in the poster's green/orange palette.

Each figure is drawn at its exact size on the poster (inches), so 20 pt here is 20 pt on the printed
poster. Numbers are read from results/ exactly as in src.poster_figures.

  poster_slot_concept           4.9 x 5.6 in   same table, best shot for a beginner vs. an expert
  poster_slot_skill_dependence  10.8 x 3.8 in  best ball/pocket changes with skill vs. noise floor; cost of ignoring skill
  poster_slot_win               10.8 x 3.7 in  win log loss: networks vs. logistic regression

Usage: python -m src.poster_pptx_figures
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.figures import TEXT, TEXT2, draw_table  # noqa: E402

GREEN, LIGHT, ORANGE, GRAY, PURPLE = "#1D5C45", "#8CC4AA", "#D9731F", "#A9A8A2", "#6250D6"
GRID = "#e4e3df"
SK = ["beginner", "intermediate", "expert"]
plt.rcParams.update({"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 17, "ytick.labelsize": 17,
                     "legend.fontsize": 16, "axes.labelcolor": TEXT, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(f"figures/{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved figures/" + name)


def clean(ax, grid="y"):
    ax.grid(axis=grid, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=TEXT2)


def concept():
    from src.data.dataset import TABLE_L, TABLE_W

    z = np.load("results/pipeline_figs/state_teacher.npz")
    i = int(z["state"])
    d = np.load("data/main/states.npz")
    est = {0: z["beginner"], 2: z["expert"]}
    pos, feats = d["pos"][i], d["cand_feats"][i]
    fig, axes = plt.subplots(1, 2, figsize=(4.9, 5.6))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.16, wspace=0.04)
    for ax, s, col in [(axes[0], 0, GREEN), (axes[1], 2, PURPLE)]:
        draw_table(ax, pos, d["on_table"][i], d["roles"][i], num_fs=6.5)
        k = int(est[s].argmax())
        other = int(est[2 - s].argmax())
        f = feats[k]
        segs = [(tuple(pos[0]), (f[14] * TABLE_W, f[15] * TABLE_L)), ((f[10] * TABLE_W, f[11] * TABLE_L),
                                                                     (f[12] * TABLE_W, f[13] * TABLE_L))]
        for a, b in segs:
            ax.annotate("", b, a, zorder=6, arrowprops=dict(arrowstyle="-|>", color=col, lw=4, mutation_scale=22))
        ax.text(0.5, -0.01, f"{SK[s].capitalize()}'s best", transform=ax.transAxes, ha="center", va="top",
                fontsize=17, fontweight="bold", color=col)
        ax.text(0.5, -0.075, f"{est[s][k]:.0%} for {'a beginner' if s == 0 else 'an expert'}\n"
                f"({SK[2 - s]}'s best: {est[s][other]:.0%})", transform=ax.transAxes, ha="center", va="top",
                fontsize=15, color=TEXT, linespacing=1.3)
    save(fig, "poster_slot_concept")


def skill_dependence():
    s = json.load(open("results/wide_candidates/summary.json"))["cur"]
    dep = json.load(open("results/poster_skill_dependence.json"))  # written by src.poster_figures
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.105, right=0.955, top=0.97, bottom=0.27, wspace=0.38)
    pairs = list(s["frac_ball_or_pocket_differs"])
    v = [s["frac_ball_or_pocket_differs"][p] for p in pairs]
    x = np.arange(len(pairs))
    ax[0].bar(x, v, 0.6, color=GREEN)
    for xi, vi in zip(x, v):
        ax[0].text(xi, vi + 0.008, f"{vi:.0%}", ha="center", va="bottom", fontsize=17, color=TEXT)
    nf = s["noise_floor_ball_or_pocket"]
    ax[0].axhline(nf, color=ORANGE, lw=3, ls="--", label=f"noise floor ({nf:.0%})")
    ax[0].legend(frameon=False, loc="upper left", fontsize=16, handlelength=1.6, borderaxespad=0.1)
    ax[0].set_xticks(x, ["beg vs\ninterm", "beg vs\nexpert", "interm vs\nexpert"])
    ax[0].set_ylabel("Ball/pocket differs")
    ax[0].set_ylim(0, 0.27)
    ax[0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    clean(ax[0])
    m = [dep["mean"]["b_on_e"], dep["mean"]["e_on_b"]]
    e = [dep["se"]["b_on_e"], dep["se"]["e_on_b"]]
    ax[1].bar([0, 1], m, 0.55, color=[GREEN, PURPLE], yerr=e, ecolor=TEXT, capsize=5, error_kw={"lw": 1.5})
    for xi, (mi, ei) in enumerate(zip(m, e)):
        ax[1].text(xi, mi + ei + 0.003, f"-{mi * 100:.1f} pts", ha="center", va="bottom", fontsize=17, color=TEXT)
    ax[1].set_xticks([0, 1], ["beginner plays\nexpert's best", "expert plays\nbeginner's\nbest"])
    ax[1].set_ylabel("Success lost")
    ax[1].set_ylim(0, 0.072)
    ax[1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y * 100:.0f} pts"))
    clean(ax[1])
    save(fig, "poster_slot_skill_dependence")


def win():
    lr = json.load(open("results/win_baselines/test_metrics.json"))
    heur = json.load(open("results/heuristic/test_metrics.json"))
    d = pd.read_csv("results/seeds_summary_per_seed.csv")
    rows = [("LR: skills + ball counts", lr["lr_skills_counts"]["win_logloss"], 0, ORANGE),
            ("LR: skills only", lr["lr_skills"]["win_logloss"], 0, GRAY),
            ("LR: ball counts only", heur["win_logloss"], 0, GRAY)]
    for r, lab in [("deepsets", "DeepSets"), ("set_transformer", "Set Transformer"), ("mlp", "MLP"), ("cnn", "CNN")]:
        x = d[d.run == r].win_logloss
        rows.append((lab, x.mean(), x.std(), GREEN))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(10.8, 3.7))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.98, bottom=0.2)
    y = np.arange(len(rows))[::-1]
    for yi, (lab, m, sd, c) in zip(y, rows):
        ax.barh(yi, m, height=0.66, color=c)
        if sd > 0:
            ax.errorbar(m, yi, xerr=sd, fmt="none", ecolor=TEXT, elinewidth=1.5, capsize=4)
        ax.text(m + sd + 0.002, yi, f"{m:.3f}", va="center", fontsize=16, color=TEXT)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=16)
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(r[3] if r[3] != GRAY else TEXT2)
    ax.set_xlim(0.5, 0.665)
    ax.set_xlabel("Test win log loss (lower is better)")
    clean(ax, grid="x")
    save(fig, "poster_slot_win")


if __name__ == "__main__":
    concept()
    skill_dependence()
    win()
