"""Final figure set for the poster and report (PNG + PDF, 300 dpi, no titles inside figures).

Every number is read from results/ (nothing hard-coded). Captions live in EXPERIMENTS.md.

  final_shot_success     success of the recommended shot by skill, all methods (seed mean ± std)
  final_size_ablation    shot regret and win log loss vs. training states (seed mean ± std)
  final_win_baselines    win log loss of learned models vs. simple baselines (seed mean ± std)
  final_skill_dependence best-shot disagreement across skills vs. noise floor, current vs. wide set
  final_label_noise      stored M=16 label vs. independent M=16 relabel at the same skill
  final_win_curve        win probability over one held-out rack

Usage: python -m src.final_figures [--only name ...]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.figures import SERIES, TEXT, TEXT2, style  # noqa: E402

SK = ["beginner", "intermediate", "expert"]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5, "axes.labelcolor": TEXT, "pdf.fonttype": 42})


def save(fig, name):
    """Write figures/<name>.png and .pdf at 300 dpi."""
    os.makedirs("figures", exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(f"figures/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/" + name)


def seed_stats(df: pd.DataFrame, group_col: str, value_col: str, by: list) -> pd.DataFrame:
    """Mean and std over seeds of per-seed means."""
    per_seed = df.groupby(by + [group_col])[value_col].mean().reset_index()
    return per_seed.groupby(by)[value_col].agg(["mean", "std", "count"]).reset_index()


# ------------------------------------------------------------------ 1. shot success by skill
SHOT_METHODS = [  # (label, eval dir, method prefix or exact name)
    ("Simulator search (teacher)", "results/shot_eval_abc", "teacher"),
    ("Set Transformer, skill input", "results/shot_eval_abc", "shot_a_skill"),
    ("Set Transformer, no skill input", "results/shot_eval_abc", "shot_b_noskill"),
    ("Set Transformer, 3-skill head", "results/shot_eval_abc", "shot_c_3head"),
    ("Set Transformer, M=32 labels", "results/shot_eval_final", "shot_a_m32"),
    ("+ ranking loss, M=16", "results/shot_eval_final", "shot_rank_m16"),
    ("+ ranking loss, M=32", "results/shot_eval_final", "shot_rank_m32"),
    ("DeepSets (seed 0)", "results/shot_eval_main_v2", "deepsets"),
    ("MLP (seed 0)", "results/shot_eval_main_v2", "mlp"),
    ("CNN on image (seed 0)", "results/shot_eval_main_v2", "cnn"),
    ("Geometric heuristic", "results/shot_eval_abc", "heuristic"),
    ("Random candidate", "results/shot_eval_abc", "random"),
]


def shot_table() -> pd.DataFrame:
    """Rows: method label, skill, mean, std over seeds (0 for single runs), n seeds."""
    rows = []
    cache = {}
    for label, ev, m in SHOT_METHODS:
        if ev not in cache:
            p = os.path.join(ev, "per_state.csv")
            if not os.path.exists(p):
                continue
            cache[ev] = pd.read_csv(p)
        df = cache[ev]
        sub = df[(df.method == m) | df.method.str.fullmatch(m + r"_s\d")].copy()
        if sub.empty:
            continue
        sub["seed"] = sub.method
        st = seed_stats(sub, "seed", "success", ["skill"])
        for _, r in st.iterrows():
            rows.append({"label": label, "skill": r.skill, "mean": r["mean"],
                         "std": 0.0 if r["count"] < 2 else r["std"], "n_seeds": int(r["count"])})
    return pd.DataFrame(rows)


def fig_shot_success():
    t = shot_table()
    t.to_csv("results/final_figures_shot_success.csv", index=False)
    labels = [m[0] for m in SHOT_METHODS if m[0] in set(t.label) and not m[0].startswith("Random")]
    rand = t[t.label.str.startswith("Random")].set_index("skill")["mean"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), sharey=True)
    y = np.arange(len(labels))[::-1]
    for ax, sk in zip(axes, SK):
        r = t[t.skill == sk].set_index("label").loc[labels]
        col = [SERIES[1] if lab.startswith("Simulator") else (TEXT2 if lab.startswith("Geometric") else SERIES[0])
               for lab in labels]
        ax.errorbar(r["mean"], y, xerr=r["std"], fmt="none", ecolor=TEXT2, elinewidth=1, capsize=2, zorder=2)
        ax.scatter(r["mean"], y, s=24, c=col, zorder=3, edgecolor="white", lw=0.8)
        teacher = r.loc[labels[0], "mean"]
        ax.axvline(teacher, color=SERIES[1], lw=0.8, ls=":", zorder=0)
        lo = (r["mean"] - r["std"]).min()
        ax.set_xlim(lo - 0.02, teacher + 0.02)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.set_xlabel(f"Success rate, {sk}\n(random candidate: {rand[sk]:.2f})")
        ax.grid(axis="x", color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=TEXT2)
    axes[0].set_yticks(y, labels)
    fig.tight_layout()
    save(fig, "final_shot_success")


# ------------------------------------------------------------------ 2. size ablation
def fig_size_ablation():
    d = pd.read_csv("results/seeds_summary_per_seed.csv")
    runs = {"size_set_transformer_1k": 1, "size_set_transformer_3k": 3, "size_set_transformer_10k": 10,
            "set_transformer": 17}
    d = d[d.run.isin(runs)]
    g = d.groupby("run").agg(n=("n_train_states", "mean"), reg=("shot_regret", "mean"), reg_sd=("shot_regret", "std"),
                             ll=("win_logloss", "mean"), ll_sd=("win_logloss", "std"), k=("seed", "count"))
    g = g.sort_values("n")
    lr = json.load(open("results/win_baselines/test_metrics.json"))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax[0].errorbar(g.n, g.reg, yerr=g.reg_sd, fmt="-o", color=SERIES[0], ms=4, lw=1.8, capsize=3)
    ax[0].set_ylabel("Shot regret vs. teacher label\n(lower is better)")
    ax[1].errorbar(g.n, g.ll, yerr=g.ll_sd, fmt="-o", color=SERIES[0], ms=4, lw=1.8, capsize=3,
                   label="Set Transformer (3 seeds)")
    for k, lab, c in [("lr_skills", "LR on skills only", SERIES[2]), ("lr_skills_counts", "LR on skills + ball counts", SERIES[3])]:
        ax[1].axhline(lr[k]["win_logloss"], color=c, lw=1.2, ls="--", label=lab)
    ax[1].set_ylabel("Win log loss (lower is better)")
    ax[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    for a in ax:
        a.set_xscale("log")
        a.set_xticks(g.n, [f"{int(round(v / 1000))}k" for v in g.n])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.set_xlabel("Training states")
        style(a)
    fig.tight_layout()
    save(fig, "final_size_ablation")


# ------------------------------------------------------------------ 3. win log loss vs baselines
def fig_win_baselines():
    lr = json.load(open("results/win_baselines/test_metrics.json"))
    heur = json.load(open("results/heuristic/test_metrics.json"))
    d = pd.read_csv("results/seeds_summary_per_seed.csv")
    rows = [("Constant prior", lr["prior"]["win_logloss"], 0, "base"),
            ("LR on ball counts", heur["win_logloss"], 0, "base"),
            ("LR on skills only", lr["lr_skills"]["win_logloss"], 0, "base"),
            ("LR on skills + ball counts", lr["lr_skills_counts"]["win_logloss"], 0, "base")]
    for r, lab in [("mlp", "MLP"), ("deepsets", "DeepSets"), ("set_transformer", "Set Transformer"),
                   ("set_transformer_noskill", "Set Transformer, no skill input"), ("cnn", "CNN on image")]:
        x = d[d.run == r].win_logloss
        rows.append((lab, x.mean(), x.std(), "net"))
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, m, sd, kind) in zip(y, rows):
        c = TEXT2 if kind == "base" else SERIES[0]
        ax.barh(yi, m, height=0.62, color=c)
        if sd > 0:
            ax.errorbar(m, yi, xerr=sd, fmt="none", ecolor=TEXT, elinewidth=1, capsize=2)
        ax.text(m + sd + 0.004, yi, f"{m:.3f}", va="center", fontsize=7, color=TEXT)
    best = lr["lr_skills_counts"]["win_logloss"]
    ax.axvline(best, color=TEXT2, lw=0.8, ls=":")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0.5, 0.72)
    ax.set_xlabel("Test win log loss (lower is better); gray = baseline, blue = neural (3 seeds)")
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=TEXT2)
    save(fig, "final_win_baselines")


# ------------------------------------------------------------------ 4. skill dependence
def fig_skill_dependence():
    s = json.load(open("results/wide_candidates/summary.json"))
    pairs = list(s["cur"]["frac_ball_or_pocket_differs"])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(len(pairs))
    w = 0.36
    for j, (k, lab) in enumerate([("cur", "Current set (8 pairs x 3 speeds)"),
                                  ("wide", "Wide set (all viable pairs x 3 speeds x 3 spins)")]):
        v = [s[k]["frac_ball_or_pocket_differs"][p] for p in pairs]
        ax[0].bar(x + (j - 0.5) * w, v, w * 0.92, color=SERIES[j], label=lab)
        ax[0].hlines(s[k]["noise_floor_ball_or_pocket"], x[0] - 0.5, x[-1] + 0.5, color=SERIES[j], lw=1.2, ls="--")
    ax[0].plot([], [], color=TEXT2, ls="--", lw=1.2, label="Noise floor (same skill, independent rollouts)")
    ax[0].set_xticks(x, [p.replace("-", " vs\n") for p in pairs])
    ax[0].set_ylabel("States where the best shot uses\na different ball or pocket")
    ax[0].set_ylim(0, max(0.6, ax[0].get_ylim()[1]))
    ax[0].legend(frameon=False, loc="upper left", fontsize=6.8)
    style(ax[0])
    # cross-skill loss: beginner plays expert's best, expert plays beginner's best
    cases = [((0, 2), "Beginner plays\nexpert's best"), ((2, 0), "Expert plays\nbeginner's best")]
    xx = np.arange(len(cases))
    for j, k in enumerate(["cur", "wide"]):
        L = np.array(s[k]["success_lost"])
        v = [L[a, b] for (a, b), _ in cases]
        ax[1].bar(xx + (j - 0.5) * w, v, w * 0.92, color=SERIES[j])
        for xi, vi in zip(xx, v):
            ax[1].text(xi + (j - 0.5) * w, vi + 0.002, f"{vi:.3f}", ha="center", fontsize=6.8, color=TEXT)
    ax[1].set_xticks(xx, [c[1] for c in cases])
    ax[1].set_ylabel("Success lost vs. own best shot")
    style(ax[1])
    fig.tight_layout()
    save(fig, "final_skill_dependence")


# ------------------------------------------------------------------ 5. label noise
def fig_label_noise():
    d = np.load("data/main/states.npz")
    l3 = np.load("data/main/relabel3/labels3.npz")["cand_label3"]
    m = d["cand_mask"] > 0
    own = l3[np.arange(len(m)), :, d["skill_self"]]
    a, b = d["cand_label"][m], own[m]
    M = 16
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    h = np.zeros((M + 1, M + 1))
    np.add.at(h, (np.rint(a * M).astype(int), np.rint(b * M).astype(int)), 1)
    im = ax[0].imshow(np.log10(h.T + 1), origin="lower", extent=(-0.5 / M, 1 + 0.5 / M, -0.5 / M, 1 + 0.5 / M),
                      cmap="Purples", aspect="equal")
    ax[0].plot([0, 1], [0, 1], color=TEXT2, lw=0.8, ls="--")
    ax[0].set_xlabel("Stored label (M=16)")
    ax[0].set_ylabel("Independent relabel, same skill (M=16)")
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    cb.set_label("log10(candidates + 1)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, colors=TEXT2)
    # spread of the difference vs. the binomial prediction for two independent M=16 estimates
    p = 0.5 * (a + b)
    bins = np.linspace(0, 1, 11)
    k = np.clip(np.digitize(p, bins) - 1, 0, 9)
    centers = 0.5 * (bins[1:] + bins[:-1])
    obs = [np.std((a - b)[k == i]) for i in range(10)]
    pp = np.linspace(0, 1, 200)
    ax[1].plot(pp, np.sqrt(2 * pp * (1 - pp) / M), color=TEXT2, lw=1.2, ls="--", label="Binomial, two M=16 estimates")
    ax[1].plot(centers, obs, "-o", color=SERIES[0], ms=4, lw=1.8, label="Observed")
    ax[1].set_xlabel("Mean of the two labels")
    ax[1].set_ylabel("Std of label difference")
    ax[1].set_ylim(0, 0.2)
    ax[1].legend(frameon=False, loc="lower center")
    style(ax[1])
    fig.tight_layout()
    save(fig, "final_label_noise")
    return {"n_candidates": int(m.sum()), "observed_std_by_bin": dict(zip(np.round(centers, 2).tolist(), map(float, obs)))}


# ------------------------------------------------------------------ 6. win curve
def fig_win_curve():
    g = dict(np.load("results/pipeline_figs/heldout_game.npz"))
    wp = dict(np.load("results/pipeline_figs/heldout_game_winprobs.npz"))
    codes = list(g["codes"])
    sh = g["shooter"]
    to0 = lambda p: np.where(sh == 0, p, 1 - p)  # noqa: E731
    y0 = float(g["win"][0] if sh[0] == 0 else 1 - g["win"][0])
    x = np.arange(len(sh))
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for i in x:
        if sh[i] == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#f0efec", lw=0, zorder=0)
    main = to0(wp["Set Transformer"])
    ax.plot(x, main, "-", color=SERIES[0], lw=2, zorder=2, label="Set Transformer")
    ax.plot(x, to0(wp["DeepSets"]), "-", color=SERIES[4], lw=1.1, zorder=1.5, label="DeepSets")
    ax.plot(x, to0(wp["LR on skills + ball counts"]), "-", color=SERIES[5], lw=1.1, zorder=1.5,
            label="LR on skills + ball counts")
    ax.axhline(y0, color=TEXT, lw=1.1, ls="--", label=f"Outcome (player 0 {'won' if y0 else 'lost'})")
    ax.axhline(0.5, color=TEXT2, lw=0.6, ls=":")
    for c, (mk, col, lab) in {"pot": ("o", SERIES[2], "Pot"), "miss": ("X", SERIES[3], "Miss"),
                              "foul": ("s", SERIES[7], "Foul")}.items():
        sel = np.array([k == c for k in codes])
        ax.scatter(x[sel], main[sel], marker=mk, s=40, color=col, edgecolor="white", lw=1.2, zorder=3, label=lab)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.7, len(x) - 0.3)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Shot number (shaded: player 0 at the table; value at i = estimate before shot i)")
    ax.set_ylabel("P(player 0 wins)")
    style(ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "final_win_curve")


FIGS = {"shot_success": fig_shot_success, "size_ablation": fig_size_ablation, "win_baselines": fig_win_baselines,
        "skill_dependence": fig_skill_dependence, "label_noise": fig_label_noise, "win_curve": fig_win_curve}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(FIGS))
    a = ap.parse_args()
    for k in a.only:
        out = FIGS[k]()
        if out:
            print(k, json.dumps(out))


if __name__ == "__main__":
    main()
