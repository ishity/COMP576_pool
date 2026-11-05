"""Parallel dataset generation: play many racks, save every pre-shot state.

Usage:
    python -m src.data.generate --config configs/data_smoke.yaml

Each game is written to ``<out_dir>/games/game_XXXXXX.npz`` as soon as it finishes, so an
interrupted run can be resumed (finished games are skipped). ``merge`` concatenates
all games into ``<out_dir>/states.npz`` with a ``game_id`` array for game-level splits.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import yaml


def _worker(args):
    gid, cfg, out_dir = args
    from src.sim.game import play_game
    from src.sim.shots import skills_from_config

    path = os.path.join(out_dir, "games", f"game_{gid:06d}.npz")
    if os.path.exists(path) or os.path.exists(path + ".skip"):
        return gid, "exists", 0, 0.0
    skills = skills_from_config(cfg["skills"])
    seed = int(cfg["seed"]) * 1_000_003 + gid
    rng = np.random.default_rng(seed)
    skill_idx = tuple(int(x) for x in rng.integers(0, len(skills), size=2))
    t0 = time.time()
    out = play_game(skills, skill_idx, cfg["game"], seed=seed)
    dt = time.time() - t0
    if out is None:
        open(path + ".skip", "w").close()
        return gid, "discarded", 0, dt
    out["game_id"] = np.full(len(out["win"]), gid, np.int64)
    np.savez_compressed(path, **out)
    return gid, "ok", len(out["win"]), dt


def merge(out_dir: str, contiguous: bool = False) -> dict:
    """Concatenate per-game files into one states.npz and return summary stats.

    With ``contiguous=True`` only games 0..g-1 are kept, where g is the first game id that
    has not finished. Use this after stopping a run early: keeping every finished game would
    over-represent short racks among the games that were still in flight.
    """
    files = sorted(glob.glob(os.path.join(out_dir, "games", "game_*.npz")))
    if contiguous:
        done = {int(os.path.basename(f)[5:11]) for f in glob.glob(os.path.join(out_dir, "games", "game_*"))}
        g = 0
        while g in done:
            g += 1
        files = [f for f in files if int(os.path.basename(f)[5:11]) < g]
    parts = [dict(np.load(f)) for f in files]
    keys = set.intersection(*[set(p) for p in parts])  # older games lack the shot_legal fields
    data = {k: np.concatenate([p[k] for p in parts]) for k in sorted(keys)}
    np.savez(os.path.join(out_dir, "states.npz"), **data)
    n_skip = len(glob.glob(os.path.join(out_dir, "games", "*.skip")))
    return {"n_games": len(files), "n_discarded": n_skip, "n_states": int(len(data["win"])),
            "max_game_id": int(data["game_id"].max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_games", type=int, default=None, help="override config")
    ap.add_argument("--n_workers", type=int, default=None)
    ap.add_argument("--merge_only", action="store_true", help="merge finished games (contiguous prefix) and exit")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.merge_only:
        summary = merge(cfg["out_dir"], contiguous=True)
        json.dump(summary, open(os.path.join(cfg["out_dir"], "summary.json"), "w"), indent=2)
        print(summary)
        return
    n_games = a.n_games or cfg["n_games"]
    n_workers = a.n_workers or cfg["n_workers"]
    out_dir = cfg["out_dir"]
    os.makedirs(os.path.join(out_dir, "games"), exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    t0 = time.time()
    n_states = 0
    log = open(os.path.join(out_dir, "generate_log.csv"), "a")
    with Pool(n_workers, maxtasksperchild=50) as pool:
        jobs = [(g, cfg, out_dir) for g in range(n_games)]
        for i, (gid, status, n, dt) in enumerate(pool.imap_unordered(_worker, jobs)):
            n_states += n
            log.write(f"{gid},{status},{n},{dt:.1f}\n")
            log.flush()
            if (i + 1) % max(1, n_games // 20) == 0 or i + 1 == n_games:
                el = time.time() - t0
                print(f"[{i+1}/{n_games}] states={n_states} elapsed={el/60:.1f}min "
                      f"({n_states/max(el,1):.2f} states/s)", flush=True)
    summary = merge(out_dir)
    summary["wall_minutes"] = (time.time() - t0) / 60
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(summary)


if __name__ == "__main__":
    main()
