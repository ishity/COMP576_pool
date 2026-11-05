"""Pipeline-explainer figures from real data (saved to figures/, data to results/pipeline_figs/).

  fig_pipeline_candidates : one test state, every candidate shot drawn as a line and colored
                            by its teacher success rate (stored M-rollout label), best highlighted.
  fig_pipeline_win_curve  : win probability over one full held-out rack (model vs. true outcome),
                            with pot / miss / foul markers from pooltool's per-shot verdict.
                            Racks stored before 2026-11-05 cannot be replayed (python `random` was
                            unseeded), so we generate new held-out racks with the same pipeline and
                            config, game ids far outside the dataset, each played twice to verify
                            determinism.
  fig_pipeline_skill_pair : the same state, teacher's best shot at beginner vs. expert skill
                            (fresh M_ref-rollout teacher estimates at each skill).

Usage:
    python -m src.pipeline_figures --data data/main/states.npz --shot_eval results/shot_eval_main \
        --run set_transformer
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.data.dataset import TABLE_L, TABLE_W, split_masks  # noqa: E402
from src.figures import SERIES, TEXT, TEXT2, draw_table, save, style  # noqa: E402

OUT = "results/pipeline_figs"
SKILLS = ["beginner", "intermediate", "expert"]
SPEEDS = ["soft", "medium", "firm"]
# Sequential, single hue (violet) so it does not clash with the blue/orange ball groups. The
# palest 25% is cut so that low-success lines stay visible on the felt.
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CMAP = LinearSegmentedColormap.from_list("succ", plt.get_cmap("Purples")(np.linspace(0.25, 1.0, 256)))


# ---------------------------------------------------------------- held-out racks (fig 2)
HELDOUT_GID0 = 10_000_000  # far outside the dataset's game ids; multiples of 10 = "test" bucket


def _play(args):
    """Play held-out rack ``gid`` (same seeding rule as src.data.generate) with a shot log."""
    gid, cfg = args
    from src.sim.game import play_game
    from src.sim.shots import skills_from_config

    skills = skills_from_config(cfg["skills"])
    seed = int(cfg["seed"]) * 1_000_003 + gid
    skill_idx = tuple(int(x) for x in np.random.default_rng(seed).integers(0, len(skills), size=2))
    log: list = []
    out = play_game(skills, skill_idx, cfg["game"], seed=seed, shot_log=log)
    return gid, out, log


def outcome_codes(log: list, chosen_success: np.ndarray) -> list:
    """pot = called ball made legally; foul = pooltool marked the shot illegal; else miss."""
    return ["foul" if not e["legal"] else ("pot" if ok > 0.5 else "miss") for e, ok in zip(log, chosen_success)]


def heldout_game(cfg, n_games: int):
    """Generate n_games held-out racks twice each; keep the longest with a pot, a miss and a foul."""
    gids = [HELDOUT_GID0 + 10 * k for k in range(n_games)]
    with Pool(2 * n_games) as p:
        res = p.map(_play, [(g, cfg) for g in gids] * 2)
    first, second = res[:n_games], res[n_games:]
    summary, cands = [], []
    for (g, out, log), (_, out2, _) in zip(first, second):
        if out is None:
            summary.append({"game_id": g, "discarded": True})
            continue
        det = out2 is not None and np.array_equal(out["pos"], out2["pos"]) and np.array_equal(out["cand_label"], out2["cand_label"])
        codes = outcome_codes(log, out["chosen_success"])
        rec = {"game_id": g, "discarded": False, "deterministic": bool(det), "n_shots": len(codes),
               "n_pot": codes.count("pot"), "n_miss": codes.count("miss"), "n_foul": codes.count("foul"),
               "skills": [int(out["skill_self"][0]), int(out["skill_opp"][0])], "shooter0": int(out["shooter"][0])}
        summary.append(rec)
        if det and min(rec["n_pot"], rec["n_miss"], rec["n_foul"]) > 0:
            cands.append((rec, out, log, codes))
    chosen = max(cands, key=lambda c: c[0]["n_shots"]) if cands else None
    return chosen, summary


def model_win_prob(run: str, out: dict) -> np.ndarray:
    """Win probabilities of a trained run on every state of one generated rack."""
    import torch

    from src.data.dataset import to_tensors
    from src.models.nets import build_model
    from src.train import predict

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join("results", run, "best.pt"), map_location=dev)
    c = ck["cfg"]
    m = build_model(c["model"], ck["cand_f"], use_skill=c["use_skill"], **c.get("model_kwargs", {})).to(dev)
    m.load_state_dict(ck["model"])
    pw, _ = predict(m, to_tensors(out, np.arange(len(out["win"])), dev))
    return pw


def fig_win_curve(g, codes, win_prob, gid, name="fig_pipeline_win_curve"):
    """P(player 0 wins) after each shot of one rack, model vs. eventual outcome."""
    shooter = g["shooter"]
    p0 = np.where(shooter == 0, win_prob, 1 - win_prob)
    y0 = float(g["win"][0] if shooter[0] == 0 else 1 - g["win"][0])
    sk0 = int(g["skill_self"][0] if shooter[0] == 0 else g["skill_opp"][0])
    sk1 = int(g["skill_opp"][0] if shooter[0] == 0 else g["skill_self"][0])
    x = np.arange(len(p0))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    # shade turns of player 0 so the reader sees who is at the table
    for i in x:
        if shooter[i] == 0:
            ax.axvspan(i - 0.5, i + 0.5, color="#f0efec", lw=0, zorder=0)
    ax.plot(x, p0, "-", color=SERIES[0], lw=2, zorder=2, label="Model: P(player 0 wins)")
    ax.axhline(y0, color=TEXT, lw=1.2, ls="--", zorder=1,
               label=f"True outcome (player 0 {'won' if y0 else 'lost'})")
    ax.axhline(0.5, color=TEXT2, lw=0.6, ls=":", zorder=1)
    mk = {"pot": ("o", SERIES[2]), "miss": ("X", SERIES[3]), "foul": ("s", SERIES[7])}
    for c, (m, col) in mk.items():
        sel = np.array([k == c for k in codes])
        ax.scatter(x[sel], p0[sel], marker=m, s=60, color=col, edgecolor="white", lw=1.5, zorder=3,
                   label={"pot": "Pot (called ball made)", "miss": "Miss (legal, no pot)", "foul": "Foul"}[c])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.7, len(p0) - 0.3)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("Shot number after the break (shaded = player 0 at the table)")
    ax.set_ylabel("P(player 0 wins)")
    ax.set_title(f"Held-out rack (not in training data): player 0 = {SKILLS[sk0]}, player 1 = {SKILLS[sk1]}\n"
                 "Line = model estimate before each shot; marker = outcome of that shot", fontsize=9, color=TEXT)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, name)


# ---------------------------------------------------------------- state figures (fig 1, 3)
def _teacher_full(args):
    """Fresh teacher estimates for every candidate of one state at one skill."""
    i, st, skill_i, cfg, m_ref = args
    from src.sim.game import system_from_state
    from src.sim.shots import BALL_IDS, POCKET_IDS, Candidate, estimate_success, skills_from_config

    skill = skills_from_config(cfg["skills"])[skill_i]
    system = system_from_state(st["pos"], st["on_table"])
    hit = tuple(b for b, h in zip(BALL_IDS, st["hittable"]) if h > 0.5)
    n = int(st["cand_mask"].sum())
    cands = [Candidate(BALL_IDS[st["cand_meta"][k, 0]], POCKET_IDS[st["cand_meta"][k, 1]], int(st["cand_meta"][k, 2]),
                       *map(float, st["cand_params"][k]), feats=(), heuristic=0.0) for k in range(n)]
    rng = np.random.default_rng(424242 + 1000 * i + skill_i)
    return i, skill_i, estimate_success(system, cands, hit, skill, m_ref, rng)


def pick_state(d, shot_eval, cfg, m_ref, n_try):
    """State where the teacher's beginner and expert picks differ in ball or pocket.

    Candidates come from the shot-evaluation states (teacher picks logged there); the full
    label vectors are then re-estimated with m_ref fresh rollouts at both skills, and the
    first state whose argmax still differs (different ball or pocket) is kept.
    """
    df = pd.read_csv(os.path.join(shot_eval, "per_state.csv"))
    t = df[df.method == "teacher"].pivot_table(index="state", columns="skill", values="pick")
    meta = d["cand_meta"]
    diff = [int(i) for i in t.index
            if tuple(meta[i, int(t.loc[i, "beginner"]), :2]) != tuple(meta[i, int(t.loc[i, "expert"]), :2])]
    diff = sorted(diff, key=lambda i: -d["cand_mask"][i].sum())[:n_try]
    keys = ["pos", "on_table", "hittable", "cand_meta", "cand_params", "cand_mask"]
    jobs = [(i, {k: d[k][i] for k in keys}, s, cfg, m_ref) for i in diff for s in (0, 2)]
    with Pool(min(96, len(jobs))) as p:
        res = p.map(_teacher_full, jobs)
    est = {}
    for i, s, v in res:
        est.setdefault(i, {})[s] = v
    for i in diff:
        b, e = est[i][0], est[i][2]
        if tuple(meta[i, b.argmax(), :2]) != tuple(meta[i, e.argmax(), :2]):
            return i, est[i], len(diff)
    return None, None, len(diff)


def _shot_lines(d, i, k):
    """Cue -> ghost ball and object ball -> potting point segments for candidate k."""
    f = d["cand_feats"][i, k]
    cue = d["pos"][i][0]
    ghost = (f[14] * TABLE_W, f[15] * TABLE_L)
    obj = (f[10] * TABLE_W, f[11] * TABLE_L)
    pp = (f[12] * TABLE_W, f[13] * TABLE_L)
    return [(tuple(cue), ghost), (obj, pp)]


def _shot_text(d, i, k):
    ball, _, spd = d["cand_meta"][i, k]
    pocket = ["left bottom", "left side", "left top", "right bottom", "right side", "right top"][d["cand_meta"][i, k, 1]]
    return f"ball {ball} to {pocket} pocket, {SPEEDS[spd]}"


def fig_candidates(d, i, name="fig_pipeline_candidates"):
    """All candidates of state i; a (ball, pocket) line is colored by its best speed's label."""
    n = int(d["cand_mask"][i].sum())
    lab = d["cand_label"][i, :n]
    pairs = {}
    for k in range(n):
        key = tuple(d["cand_meta"][i, k, :2])
        if key not in pairs or lab[k] > lab[pairs[key]]:
            pairs[key] = k
    best = int(lab.argmax())
    fig, ax = plt.subplots(figsize=(4.6, 6.6))
    draw_table(ax, d["pos"][i], d["on_table"][i], d["roles"][i])
    norm = plt.Normalize(0, 1)
    segs, vals = [], []
    for k in sorted(pairs.values(), key=lambda k: lab[k]):  # draw best last
        if k == best:
            continue
        for s in _shot_lines(d, i, k):
            segs.append(s)
            vals.append(lab[k])
    lc = LineCollection(segs, cmap=CMAP, norm=norm, lw=1.8, zorder=4)
    lc.set_array(np.array(vals))
    ax.add_collection(lc)
    for s in _shot_lines(d, i, best):
        ax.annotate("", s[1], s[0], zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color=CMAP(norm(lab[best])), lw=3.2))
        ax.plot(*zip(*s), color=TEXT, lw=5.2, zorder=5, solid_capstyle="round")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"Teacher success rate (M={int(d['_M'])} rollouts)", fontsize=8, color=TEXT2)
    cb.outline.set_visible(False)
    sk = SKILLS[int(d["skill_self"][i])]
    ax.set_title(f"Test state {i}: {len(pairs)} (ball, pocket) pairs x 3 speeds = {n} candidates\n"
                 f"line color = best speed of that pair, skill = {sk}\n"
                 f"best (outlined): {_shot_text(d, i, best)}, success {lab[best]:.2f}", fontsize=7.5, color=TEXT)
    legend_roles(ax)
    save(fig, name)


def legend_roles(ax):
    from src.figures import ROLE_COLORS
    names = ["cue ball", "8-ball", "shooter's group", "opponent's group", "open table"]
    h = [Line2D([], [], marker="o", ls="", mfc=c, mec=TEXT, mew=0.6, ms=6, label=nm) for c, nm in zip(ROLE_COLORS, names)]
    ax.legend(handles=h, frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)


def fig_skill_pair(d, i, est, m_ref, name="fig_pipeline_skill_pair"):
    """Same state, teacher's best shot at beginner (left) and expert (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 6.4))
    for ax, s, col in [(axes[0], 0, SERIES[2]), (axes[1], 2, SERIES[6])]:
        draw_table(ax, d["pos"][i], d["on_table"][i], d["roles"][i])
        v = est[s]
        k = int(v.argmax())
        for seg in _shot_lines(d, i, k):
            ax.annotate("", seg[1], seg[0], zorder=6, arrowprops=dict(arrowstyle="-|>", color=col, lw=3))
        other = int(est[2 - s].argmax())
        ax.set_title(f"{SKILLS[s].capitalize()}: teacher's best shot\n{_shot_text(d, i, k)}\n"
                     f"success at this skill {v[k]:.2f}; {SKILLS[2 - s]}'s pick here {v[other]:.2f}",
                     fontsize=7.5, color=TEXT)
    legend_roles(axes[0])
    fig.suptitle(f"Test state {i}: the best shot depends on skill (teacher re-estimated with {m_ref} fresh rollouts "
                 "per candidate)", fontsize=8.5, color=TEXT)
    save(fig, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--shot_eval", required=True)
    ap.add_argument("--run", default="set_transformer")
    ap.add_argument("--m_ref", type=int, default=64)
    ap.add_argument("--n_heldout_games", type=int, default=8)
    ap.add_argument("--n_try_states", type=int, default=12)
    ap.add_argument("--only", choices=["all", "state", "win", "win_plot"], default="all")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(a.data), "config.yaml")))
    meta_path = os.path.join(OUT, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    if a.only in ("all", "state"):  # Fig 1 + 3
        d = dict(np.load(a.data))
        d["_M"] = cfg["game"]["n_rollouts"]
        i, est, n_cands = pick_state(d, a.shot_eval, cfg, a.m_ref, a.n_try_states)
        assert i is not None, "no state with skill-dependent teacher pick"
        assert split_masks(d["game_id"][i : i + 1])["test"][0]
        np.savez(os.path.join(OUT, "state_teacher.npz"), state=i, beginner=est[0], expert=est[2],
                 stored_label=d["cand_label"][i, : int(d["cand_mask"][i].sum())])
        fig_candidates(d, i)
        fig_skill_pair(d, i, est, a.m_ref)
        meta.update({"state_for_fig1_fig3": int(i), "state_game_id": int(d["game_id"][i]),
                     "state_skill": SKILLS[int(d["skill_self"][i])], "n_candidate_states_checked": n_cands,
                     "m_ref": a.m_ref, "beginner_best": int(est[0].argmax()), "expert_best": int(est[2].argmax()),
                     "beginner_best_succ_beginner": float(est[0].max()), "expert_best_succ_expert": float(est[2].max()),
                     "expert_pick_succ_beginner": float(est[0][est[2].argmax()]),
                     "beginner_pick_succ_expert": float(est[2][est[0].argmax()])})

    if a.only in ("all", "win"):  # Fig 2
        chosen, summary = heldout_game(cfg, a.n_heldout_games)
        json.dump(summary, open(os.path.join(OUT, "heldout_games.json"), "w"), indent=1)
        assert chosen is not None, "no held-out rack with pot, miss and foul"
        rec, out, log, codes = chosen
        pw = model_win_prob(a.run, out)
        np.savez(os.path.join(OUT, "heldout_game.npz"), win_prob=pw, codes=np.array(codes), **out)
        json.dump(log, open(os.path.join(OUT, "heldout_game_shotlog.json"), "w"), indent=1)
        fig_win_curve(out, codes, pw, rec["game_id"])
        p0 = np.where(out["shooter"] == 0, pw, 1 - pw)
        meta.update({"game_for_fig2": rec, "win_model": a.run,
                     "fig2_p0_first": float(p0[0]), "fig2_p0_last": float(p0[-1]),
                     "foul_reasons": [e["reason"] for e in log if not e["legal"]]})

    if a.only == "win_plot":  # redraw Fig 2 from the saved held-out rack (no simulation)
        z = dict(np.load(os.path.join(OUT, "heldout_game.npz")))
        fig_win_curve(z, list(z["codes"]), z["win_prob"], int(z["game_id"][0]) if "game_id" in z else -1)

    json.dump(meta, open(meta_path, "w"), indent=2)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
