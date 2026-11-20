"""Four large figures for the poster (PNG + PDF, 300 dpi, no titles, large fonts).

Every number is read from results/ (nothing hard-coded).

  poster_concept          same test table: the teacher's best shot for a beginner vs. an expert
                          (results/pipeline_figs/state_teacher.npz, 64 fresh rollouts per candidate)
  poster_shot_success     success of the recommended shot by skill: teacher, Set Transformer with and
                          without skill input, three-output head, heuristic (results/shot_eval_single)
  poster_skill_dependence how often the best ball/pocket changes with skill vs. the noise floor, and the
                          success lost by playing another skill's best shot (results/wide_candidates, current set)
  poster_win              win log loss: neural models (3 seeds) vs. logistic regression baselines

Usage: python -m src.poster_figures [--only name ...]
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
from matplotlib.lines import Line2D  # noqa: E402

from src.figures import ROLE_COLORS, SERIES, TEXT, TEXT2, draw_table  # noqa: E402

SK = ["beginner", "intermediate", "expert"]
GRID = "#e4e3df"
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 11.5, "axes.labelcolor": TEXT, "pdf.fonttype": 42})


def save(fig, name):
    """Write figures/<name>.png and .pdf at 300 dpi."""
    for ext in ["png", "pdf"]:
        fig.savefig(f"figures/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved figures/" + name)


def clean(ax, grid="y"):
    """Light grid, no top/right spines, secondary-ink ticks."""
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=TEXT2)


# ------------------------------------------------------------------ 1. concept
def fig_concept():
    from src.data.dataset import TABLE_L, TABLE_W

    z = np.load("results/pipeline_figs/state_teacher.npz")
    i = int(z["state"])
    d = np.load("data/main/states.npz")
    est = {0: z["beginner"], 2: z["expert"]}
    pos, on, roles, feats = d["pos"][i], d["on_table"][i], d["roles"][i], d["cand_feats"][i]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 6.6))
    for ax, s, col in [(axes[0], 0, SERIES[2]), (axes[1], 2, SERIES[6])]:
        draw_table(ax, pos, on, roles, num_fs=6)
        k = int(est[s].argmax())
        other = int(est[2 - s].argmax())
        f = feats[k]
        cue, ghost = tuple(pos[0]), (f[14] * TABLE_W, f[15] * TABLE_L)
        obj, pp = (f[10] * TABLE_W, f[11] * TABLE_L), (f[12] * TABLE_W, f[13] * TABLE_L)
        for a, b in [(cue, ghost), (obj, pp)]:
            ax.annotate("", b, a, zorder=6, arrowprops=dict(arrowstyle="-|>", color=col, lw=3.5, mutation_scale=18))
        who, them = SK[s].capitalize(), SK[2 - s]
        ax.text(0.5, -0.03, f"{who}'s best shot", transform=ax.transAxes, ha="center", va="top",
                fontsize=14, fontweight="bold", color=col)
        ax.text(0.5, -0.09, f"{est[s][k]:.0%} for a{'n' if s == 2 else ''} {SK[s]}\n"
                f"(the {them}'s best shot: {est[s][other]:.0%})", transform=ax.transAxes, ha="center", va="top",
                fontsize=12.5, color=TEXT, linespacing=1.4)
    names = ["cue ball", "8-ball", "shooter's group", "opponent's group"]
    h = [Line2D([], [], marker="o", ls="", mfc=c, mec=TEXT, mew=0.6, ms=9, label=n) for c, n in zip(ROLE_COLORS, names)]
    fig.legend(handles=h, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), fontsize=11.5,
               handletextpad=0.2, columnspacing=1.0)
    fig.subplots_adjust(wspace=0.05)
    save(fig, "poster_concept")


# ------------------------------------------------------------------ 2. shot success
SHOT = [("Simulator search (teacher)", "teacher", SERIES[1]),
        ("Set Transformer, skill input", "shot_a_skill", SERIES[0]),
        ("Set Transformer, no skill input", "shot_b_noskill", "#8fb8ea"),
        ("Set Transformer, 3-output head", "shot_c_3head", SERIES[6]),
        ("Geometric heuristic", "heuristic", "#a9a8a2")]


def shot_table() -> pd.DataFrame:
    df = pd.read_csv("results/shot_eval_single/per_state.csv")
    rows = []
    for label, m, _ in SHOT:
        sub = df[(df.method == m) | df.method.str.fullmatch(m + r"_s\d")]
        per = sub.groupby(["skill", "method"]).success.mean().reset_index()
        st = per.groupby("skill").success.agg(["mean", "std", "count"]).reset_index()
        for _, r in st.iterrows():
            rows.append({"label": label, "skill": r.skill, "mean": r["mean"],
                         "std": 0.0 if r["count"] < 2 else r["std"], "n_seeds": int(r["count"])})
    return pd.DataFrame(rows)


def fig_shot_success():
    t = shot_table()
    t.to_csv("results/poster_shot_success.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    n = len(SHOT)
    w = 0.8 / n
    x = np.arange(3)
    for j, (label, _, col) in enumerate(SHOT):
        r = t[t.label == label].set_index("skill").loc[SK]
        xx = x + (j - (n - 1) / 2) * w
        ax.bar(xx, r["mean"], w * 0.9, color=col, label=label)
        if (r["std"] > 0).any():
            ax.errorbar(xx, r["mean"], yerr=r["std"], fmt="none", ecolor=TEXT, elinewidth=1, capsize=2)
        for xi, m, sd in zip(xx, r["mean"], r["std"]):
            ax.text(xi, m + sd + 0.012, f"{m:.2f}", ha="center", va="bottom", fontsize=9.5, color=TEXT)
    ax.set_xticks(x, [s.capitalize() for s in SK], fontsize=13)
    ax.set_ylabel("Success rate of the recommended shot\n(64 fresh noisy executions)")
    ax.set_ylim(0, 0.85)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    clean(ax)
    save(fig, "poster_shot_success")


# ------------------------------------------------------------------ 3. skill dependence
def fig_skill_dependence():
    s = json.load(open("results/wide_candidates/summary.json"))["cur"]
    z = np.load("results/wide_candidates/raw.npz", allow_pickle=True)
    res, half = z["res"].item(), int(z["M"]) // 2
    loss = {"b_on_e": [], "e_on_b": []}  # per-state success lost, scored on half B
    for r in res.values():
        h = r["cur"]["hits"] / half
        a = h[:, :, 0].argmax(0)
        loss["b_on_e"].append(h[a[0], 0, 1] - h[a[2], 0, 1])
        loss["e_on_b"].append(h[a[2], 2, 1] - h[a[0], 2, 1])
    se = {k: np.std(v, ddof=1) / np.sqrt(len(v)) for k, v in loss.items()}
    mean = {k: float(np.mean(v)) for k, v in loss.items()}
    json.dump({"mean": mean, "se": se, "n_states": len(res)}, open("results/poster_skill_dependence.json", "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
    pairs = list(s["frac_ball_or_pocket_differs"])
    v = [s["frac_ball_or_pocket_differs"][p] for p in pairs]
    x = np.arange(len(pairs))
    ax[0].bar(x, v, 0.6, color=SERIES[0], label="Between two skill levels")
    for xi, vi in zip(x, v):
        ax[0].text(xi, vi + 0.006, f"{vi:.0%}", ha="center", va="bottom", fontsize=12, color=TEXT)
    nf = s["noise_floor_ball_or_pocket"]
    ax[0].axhline(nf, color=SERIES[1], lw=2, ls="--", label=f"Noise floor, same skill ({nf:.0%})")
    ax[0].set_xticks(x, [p.replace("intermediate", "interm.").replace("-", " vs\n") for p in pairs])
    ax[0].set_ylabel("Tables where the best shot\nuses a different ball or pocket")
    ax[0].set_ylim(0, 0.3)
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax[0].legend(frameon=False, loc="upper left")
    clean(ax[0])
    labels = ["Beginner plays\nthe expert's\nbest shot", "Expert plays\nthe beginner's\nbest shot"]
    m = [mean["b_on_e"], mean["e_on_b"]]
    e = [se["b_on_e"], se["e_on_b"]]
    ax[1].bar([0, 1], m, 0.55, color=[SERIES[2], SERIES[6]], yerr=e, ecolor=TEXT, capsize=4)
    for xi, (mi, ei) in enumerate(zip(m, e)):
        ax[1].text(xi, mi + ei + 0.002, f"-{mi * 100:.1f} pts", ha="center", va="bottom", fontsize=12, color=TEXT)
    ax[1].set_xticks([0, 1], labels)
    ax[1].set_ylabel("Success lost vs. own best shot")
    ax[1].set_ylim(0, 0.07)
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y * 100:.0f} pts"))
    clean(ax[1])
    fig.tight_layout(w_pad=2.5)
    save(fig, "poster_skill_dependence")


# ------------------------------------------------------------------ 4. win prediction
def fig_win():
    lr = json.load(open("results/win_baselines/test_metrics.json"))
    heur = json.load(open("results/heuristic/test_metrics.json"))
    d = pd.read_csv("results/seeds_summary_per_seed.csv")
    rows = [("LR on skills + ball counts", lr["lr_skills_counts"]["win_logloss"], 0, "best"),
            ("LR on skills only", lr["lr_skills"]["win_logloss"], 0, "base"),
            ("LR on ball counts only", heur["win_logloss"], 0, "base")]
    for r, lab in [("deepsets", "DeepSets"), ("set_transformer", "Set Transformer"), ("mlp", "MLP"),
                   ("cnn", "CNN on image")]:
        x = d[d.run == r].win_logloss
        rows.append((lab, x.mean(), x.std(), "net"))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    y = np.arange(len(rows))[::-1]
    col = {"best": SERIES[1], "base": "#a9a8a2", "net": SERIES[0]}
    for yi, (lab, m, sd, kind) in zip(y, rows):
        ax.barh(yi, m, height=0.62, color=col[kind])
        if sd > 0:
            ax.errorbar(m, yi, xerr=sd, fmt="none", ecolor=TEXT, elinewidth=1.2, capsize=3)
        ax.text(m + sd + 0.003, yi, f"{m:.3f}", va="center", fontsize=12, color=TEXT)
    ax.axvline(lr["lr_skills_counts"]["win_logloss"], color=SERIES[1], lw=1.2, ls=":")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=12.5)
    ax.set_xlim(0.5, 0.66)
    ax.set_xlabel("Test win log loss (lower is better)")
    h = [plt.Rectangle((0, 0), 1, 1, color=col[k]) for k in ["net", "best", "base"]]
    ax.legend(h, ["Neural network (mean ± std, 3 seeds)", "Best baseline", "Other logistic regression (LR)"],
              frameon=False, loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2)
    clean(ax, grid="x")
    save(fig, "poster_win")


FIGS = {"concept": fig_concept, "shot_success": fig_shot_success, "skill_dependence": fig_skill_dependence,
        "win": fig_win}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(FIGS), default=list(FIGS))
    for name in ap.parse_args().only:
        FIGS[name]()


if __name__ == "__main__":
    main()
