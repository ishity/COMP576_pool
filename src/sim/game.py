"""Full 8-ball racks between two noisy simulated players.

Rules come from pooltool's built-in 8-ball ruleset (fouls, ball-in-hand, open table,
8-ball win/loss). Our own simplifications (documented in EXPERIMENTS.md):
  * Ball in hand: the shooter samples random legal cue positions and keeps the one with
    the easiest geometric shot (pooltool itself only respots the cue at the head spot).
  * No safeties, no bank/kick shots: every shot is a called ghost-ball pot attempt.
  * The break is a fixed hard hit on the apex ball (with skill noise) and is not saved.
  * Racks longer than ``max_shots`` are discarded (counted in the generation log).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

import src.sim  # noqa: F401
import pooltool as pt
import pooltool.constants as const
from pooltool.ruleset.datatypes import BallInHandOptions
from pooltool.ruleset.eight_ball import BallGroup

from src.sim.shots import (
    BALL_IDS,
    BALL_INDEX,
    POCKET_INDEX,
    Candidate,
    Skill,
    add_noise,
    estimate_success,
    generate_candidates,
    is_success,
    on_table,
)

SOLIDS = set(BallGroup.SOLIDS.balls)
STRIPES = set(BallGroup.STRIPES.balls)
# Role channels per ball (relative to the shooter): cue, eight, own, opponent, open-table
N_ROLES = 5


def ball_roles(system: pt.System, hittable: Sequence[str]) -> np.ndarray:
    """One-hot role of every ball from the shooter's perspective, shape (16, 5)."""
    roles = np.zeros((16, N_ROLES), dtype=np.float32)
    hit = set(hittable)
    if hit == SOLIDS:
        own, opp, open_ = SOLIDS, STRIPES, set()
    elif hit == STRIPES:
        own, opp, open_ = STRIPES, SOLIDS, set()
    elif hit == {"8"}:
        solids_done = all(not on_table(system, b) for b in SOLIDS)
        stripes_done = all(not on_table(system, b) for b in STRIPES)
        if solids_done and not stripes_done:
            own, opp = SOLIDS, STRIPES
        elif stripes_done and not solids_done:
            own, opp = STRIPES, SOLIDS
        else:
            own, opp = set(), set()
        open_ = set()
    else:  # open table
        own, opp, open_ = set(), set(), SOLIDS | STRIPES
    for i, b in enumerate(BALL_IDS):
        if b == "cue":
            roles[i, 0] = 1
        elif b == "8":
            roles[i, 1] = 1
        elif b in own:
            roles[i, 2] = 1
        elif b in opp:
            roles[i, 3] = 1
        elif b in open_:
            roles[i, 4] = 1
    return roles


def encode_state(system: pt.System, hittable: Sequence[str]) -> Dict[str, np.ndarray]:
    """Raw state arrays saved to disk (positions, on-table mask, roles, hittable mask)."""
    pos = np.array([system.balls[b].xyz[:2] for b in BALL_IDS], dtype=np.float32)
    on = np.array([on_table(system, b) for b in BALL_IDS], dtype=np.float32)
    hit = np.array([b in hittable for b in BALL_IDS], dtype=np.float32)
    return {"pos": pos, "on_table": on, "roles": ball_roles(system, hittable), "hittable": hit}


def system_from_state(pos: np.ndarray, on: np.ndarray) -> pt.System:
    """Rebuild a pooltool System from saved positions (used for offline evaluation)."""
    table = pt.Table.default()
    balls = pt.get_rack(pt.GameType.EIGHTBALL, table)
    for i, b in enumerate(BALL_IDS):
        ball = balls[b]
        ball.state.rvw[0] = [float(pos[i, 0]), float(pos[i, 1]), ball.params.R]
        ball.state.rvw[1] = 0.0
        ball.state.rvw[2] = 0.0
        ball.state.s = const.stationary if on[i] > 0.5 else const.pocketed
    return pt.System(table=table, balls=balls, cue=pt.Cue(cue_ball_id="cue"))


def _overlaps(system: pt.System, xy: np.ndarray, R: float) -> bool:
    for b in system.balls.values():
        if b.id != "cue" and b.state.s != const.pocketed:
            if np.linalg.norm(b.xyz[:2] - xy) < 2 * R + 1e-3:
                return True
    return False


def place_cue_ball(system, hittable, speeds, top_k, rng, n_samples=30) -> None:
    """Ball in hand: try random legal spots, keep the one with the easiest shot."""
    cue = system.balls["cue"]
    R = cue.params.R
    W, L = system.table.w, system.table.l
    best, best_h = None, np.inf
    for _ in range(n_samples):
        xy = np.array([rng.uniform(R + 0.02, W - R - 0.02), rng.uniform(R + 0.02, L - R - 0.02)])
        if _overlaps(system, xy, R):
            continue
        cue.state.rvw[0] = [xy[0], xy[1], R]
        cue.state.s = const.stationary
        cands = generate_candidates(system, hittable, speeds[:1], top_k=1)
        h = cands[0].heuristic if cands else np.inf
        if best is None or h < best_h:
            best, best_h = xy, h
    if best is None:  # extremely crowded table: fall back to head spot
        best = np.array([W / 2, L / 4])
    cue.state.rvw[0] = [best[0], best[1], R]
    cue.state.rvw[1] = 0.0
    cue.state.rvw[2] = 0.0
    cue.state.s = const.stationary


def choose(p_hat: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Softmax over estimated success probabilities (extra randomness for diversity)."""
    z = p_hat / max(temperature, 1e-6)
    z = np.exp(z - z.max())
    return int(rng.choice(len(p_hat), p=z / z.sum()))


def _break_shot(system: pt.System, skill: Skill, rng: np.random.Generator):
    cue = system.balls["cue"]
    objs = [b for b in system.balls.values() if b.id != "cue"]
    apex = min(objs, key=lambda b: np.linalg.norm(b.xyz[:2] - cue.xyz[:2]))
    phi = pt.aim.at_ball(system, apex.id) + rng.normal(0.0, skill.phi_std)
    V0 = 7.0 * (1.0 + rng.normal(0.0, skill.v_rel_std))
    return phi, V0, 0.0, 0.0


def play_game(
    skills: Sequence[Skill],
    skill_idx: Sequence[int],
    cfg: dict,
    seed: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Play one rack and return stacked arrays of all saved pre-shot states.

    Returns None if the rack exceeded ``max_shots`` or ended before any saved shot.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed % (2**32))  # pooltool's rack generator uses the global RNG
    speeds = cfg["speeds"]
    top_k = cfg["top_k_pairs"]
    M = cfg["n_rollouts"]
    K_max = top_k * len(speeds)

    table = pt.Table.default()
    balls = pt.get_rack(pt.GameType.EIGHTBALL, table)
    system = pt.System(table=table, balls=balls, cue=pt.Cue(cue_ball_id="cue"))
    game = pt.get_ruleset(pt.GameType.EIGHTBALL)(players=[pt.Player("P0"), pt.Player("P1")])

    rec: List[dict] = []
    while True:
        if game.shot_number >= cfg["max_shots"]:
            return None
        shooter = game.active_idx
        skill = skills[skill_idx[shooter]]
        cons = game.shot_constraints
        hittable = tuple(cons.hittable)
        system.reset_history()

        if game.shot_number == 0:
            phi, V0, a, b = _break_shot(system, skill, rng)
            entry = None
        else:
            if cons.ball_in_hand == BallInHandOptions.ANYWHERE:
                place_cue_ball(system, hittable, speeds, top_k, rng)
            cands = generate_candidates(system, hittable, speeds, top_k)
            if not cands:
                return None
            p_hat = estimate_success(system, cands, hittable, skill, M, rng)
            k = choose(p_hat, cfg["choice_temperature"], rng)
            c: Candidate = cands[k]
            cons.ball_call, cons.pocket_call = c.ball, c.pocket
            phi, V0, a, b = add_noise(c, skill, rng)

            n = len(cands)
            F = len(c.feats)
            feats = np.zeros((K_max, F), np.float32)
            feats[:n] = np.array([x.feats for x in cands], np.float32)
            meta = np.zeros((K_max, 3), np.int64)  # ball idx, pocket idx, speed idx
            meta[:n] = [[BALL_INDEX[x.ball], POCKET_INDEX[x.pocket], x.speed_idx] for x in cands]
            params = np.zeros((K_max, 4), np.float32)  # intended phi, V0, a, b
            params[:n] = [[x.phi, x.V0, x.a, x.b] for x in cands]
            label = np.zeros(K_max, np.float32)
            label[:n] = p_hat
            mask = np.zeros(K_max, np.float32)
            mask[:n] = 1
            heur = np.full(K_max, np.inf, np.float32)
            heur[:n] = [x.heuristic for x in cands]
            entry = dict(
                **encode_state(system, hittable),
                shooter=shooter,
                skill_self=skill_idx[shooter],
                skill_opp=skill_idx[1 - shooter],
                shot_index=game.shot_number,
                cand_feats=feats,
                cand_meta=meta,
                cand_params=params,
                cand_label=label,
                cand_mask=mask,
                cand_heur=heur,
                chosen=k,
            )

        shot = system.copy()
        shot.cue.set_state(V0=V0, phi=phi, a=a, b=b, theta=0.0, cue_ball_id="cue")
        pt.simulate(shot, inplace=True)
        if entry is not None:
            entry["chosen_success"] = float(is_success(shot, cands[k].ball, cands[k].pocket, hittable))
            rec.append(entry)
        game.process_and_advance(shot)
        system = shot
        if game.shot_info.game_over:
            winner = game.players.index(game.shot_info.winner)
            break

    if not rec:
        return None
    out = {k: np.stack([np.asarray(r[k]) for r in rec]) for k in rec[0]}
    out["win"] = (out["shooter"] == winner).astype(np.float32)
    n_shots = len(rec)
    out["shots_to_end"] = n_shots - 1 - np.arange(n_shots)
    return out
