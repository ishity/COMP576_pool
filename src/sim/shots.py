"""Shot primitives: skill noise, candidate generation, execution and success checks.

All geometry uses pooltool helpers (ghost-ball / potting-point / occlusion checks from
``pooltool.ai.pot``). Angles are in degrees, distances in meters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

import src.sim  # noqa: F401  (numpy compat shim must run before pooltool import)
import pooltool as pt
import pooltool.constants as const
from pooltool.ai.pot import core as pot
from pooltool.ruleset import utils as rutils

BALL_IDS: Tuple[str, ...] = ("cue",) + tuple(str(i) for i in range(1, 16))
BALL_INDEX: Dict[str, int] = {b: i for i, b in enumerate(BALL_IDS)}
POCKET_IDS: Tuple[str, ...] = ("lb", "lc", "lt", "rb", "rc", "rt")
POCKET_INDEX: Dict[str, int] = {p: i for i, p in enumerate(POCKET_IDS)}
# Event cap per simulated shot. A few configurations never terminate in pooltool 0.3.3
# (observed: 2 racks hung for 16 h). Capped shots count as failures / discard the rack.
MAX_EVENTS = 2000


@dataclass(frozen=True)
class Skill:
    """Execution noise of a simulated player (Gaussian std per shot parameter)."""

    name: str
    phi_std: float  # degrees
    v_rel_std: float  # relative std of V0
    spin_std: float  # std of a and b (units of ball radius)


@dataclass(frozen=True)
class Candidate:
    """A candidate called shot: object ball, pocket, and intended cue parameters."""

    ball: str
    pocket: str
    speed_idx: int
    phi: float
    V0: float
    a: float
    b: float
    feats: Tuple[float, ...]
    heuristic: float  # lower = easier (straighter, shorter)


SKILL_ORDER: Tuple[str, ...] = ("beginner", "intermediate", "expert")


def skills_from_config(cfg: dict) -> List[Skill]:
    """Build the skill list in the fixed order beginner, intermediate, expert (index 0, 1, 2).

    The order is fixed here instead of taken from the dict, because a YAML round trip with
    sorted keys silently reorders the levels (this happened once; see EXPERIMENTS.md).
    """
    assert set(cfg) == set(SKILL_ORDER), f"skills must be exactly {SKILL_ORDER}, got {list(cfg)}"
    return [Skill(name=k, **cfg[k]) for k in SKILL_ORDER]


def on_table(system: pt.System, ball_id: str) -> bool:
    """True if the ball exists and is not pocketed."""
    return ball_id in system.balls and system.balls[ball_id].state.s != const.pocketed


def _heuristic_score(cut: float, d_cg: float, d_op: float, viable: bool, L: float) -> float:
    """Geometric difficulty: straight + short shots are easy. Non-viable pairs are penalized."""
    return abs(cut) / 90.0 + 0.5 * (d_cg + d_op) / L + (0.0 if viable else 10.0)


def generate_candidates(
    system: pt.System,
    hittable: Sequence[str],
    speeds: Sequence[float],
    top_k: int,
    max_cut: float = 85.0,
) -> List[Candidate]:
    """Enumerate ghost-ball shots for every legal object ball and pocket.

    For each (object ball, pocket) pair we compute the ghost-ball aim angle and simple
    geometric features. Pairs are ranked by a geometric difficulty heuristic and the
    ``top_k`` easiest pairs are combined with every speed level. If no pair has a cut
    angle below ``max_cut`` a direct hit on the nearest legal ball is returned
    (a "safety" fallback), so the list is never empty unless no legal ball exists.
    """
    table = system.table
    W, L = table.w, table.l
    cue = system.balls["cue"]
    balls_on = [b for b in system.balls.values() if b.state.s != const.pocketed]
    cxy = cue.xyz[:2]

    pairs = []
    for bid in hittable:
        if not on_table(system, bid):
            continue
        ball = system.balls[bid]
        oxy = ball.xyz[:2]
        for pid in POCKET_IDS:
            pocket = table.pockets[pid]
            pp = pot.get_potting_point(ball, table, pocket)
            ghost = pot.calc_shadow_ball_center(ball, table, pocket)
            cut = pot.calc_cut_angle(cxy, ghost, pp)
            if abs(cut) > max_cut:
                continue
            viable = (
                not pot.is_pocket_occluded(ball, table, pocket, balls_on)
                and not pot.is_object_ball_occluded(cue, ball, table, pocket, balls_on)
                and not pot.is_jaw_in_way(ball, table, pocket)
            )
            d_cg = float(np.linalg.norm(ghost - cxy))
            d_op = float(np.linalg.norm(pp - oxy))
            phi = float(pt.aim.at_pos(cue, np.array([ghost[0], ghost[1], cue.xyz[2]])))
            h = _heuristic_score(cut, d_cg, d_op, viable, L)
            pairs.append((h, bid, pid, cut, d_cg, d_op, viable, phi, oxy, pp, ghost))

    if not pairs:
        # Fallback: aim straight at the closest legal ball, call its nearest pocket.
        legal = [b for b in hittable if on_table(system, b)]
        if not legal:
            return []
        bid = min(legal, key=lambda b: np.linalg.norm(system.balls[b].xyz[:2] - cxy))
        ball = system.balls[bid]
        oxy = ball.xyz[:2]
        pid = min(POCKET_IDS, key=lambda p: np.linalg.norm(table.pockets[p].center[:2] - oxy))
        pp = pot.get_potting_point(ball, table, table.pockets[pid])
        phi = float(pt.aim.at_ball(system, bid))
        d_cg = float(np.linalg.norm(oxy - cxy))
        d_op = float(np.linalg.norm(pp - oxy))
        pairs.append((99.0, bid, pid, 89.0, d_cg, d_op, False, phi, oxy, pp, oxy))

    pairs.sort(key=lambda x: x[0])
    cands = []
    for h, bid, pid, cut, d_cg, d_op, viable, phi, oxy, pp, ghost in pairs[:top_k]:
        corner = pot.pocket_jaw_map[pid].corner
        for si, v in enumerate(speeds):
            onehot = [0.0] * len(speeds)
            onehot[si] = 1.0
            rad = np.deg2rad(phi)
            feats = (
                abs(cut) / 90.0,
                d_cg / L,
                d_op / L,
                float(viable),
                float(corner),
                *onehot,
                float(np.sin(rad)),
                float(np.cos(rad)),
                oxy[0] / W,
                oxy[1] / L,
                pp[0] / W,
                pp[1] / L,
                ghost[0] / W,
                ghost[1] / L,
                float(bid == "8"),
            )
            cands.append(Candidate(bid, pid, si, phi, float(v), 0.0, 0.0, feats, h))
    return cands


def add_noise(c: Candidate, skill: Skill, rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """Return executed (phi, V0, a, b) = intended + Gaussian noise for this skill."""
    phi = (c.phi + rng.normal(0.0, skill.phi_std)) % 360.0
    V0 = float(np.clip(c.V0 * (1.0 + rng.normal(0.0, skill.v_rel_std)), 0.2, 8.0))
    a = c.a + rng.normal(0.0, skill.spin_std)
    b = c.b + rng.normal(0.0, skill.spin_std)
    r = np.hypot(a, b)
    if r > 0.6:  # keep the tip on the ball (no miscue model)
        a, b = a * 0.6 / r, b * 0.6 / r
    return phi, V0, float(a), float(b)


def execute(system: pt.System, phi: float, V0: float, a: float, b: float) -> pt.System:
    """Simulate a shot on a copy of ``system`` (which must have empty history)."""
    shot = system.copy()
    shot.cue.set_state(V0=V0, phi=phi, a=a, b=b, theta=0.0, cue_ball_id="cue")
    pt.simulate(shot, inplace=True, max_events=MAX_EVENTS)
    return shot


def hit_event_cap(shot: pt.System) -> bool:
    """True if the simulation was cut off by MAX_EVENTS (ball states are then unreliable)."""
    return len(shot.events) >= MAX_EVENTS


def is_success(shot: pt.System, ball: str, pocket: str, hittable: Sequence[str]) -> bool:
    """Called ball in called pocket, no scratch, legal first contact, no early 8-ball."""
    if hit_event_cap(shot):
        return False
    if rutils.is_ball_pocketed(shot, "cue"):
        return False
    if rutils.get_id_of_first_ball_hit(shot, cue="cue") not in hittable:
        return False
    if ball != "8" and rutils.is_ball_pocketed(shot, "8"):
        return False
    return rutils.is_ball_pocketed_in_pocket(shot, ball, pocket)


def estimate_success(
    system: pt.System,
    cands: Sequence[Candidate],
    hittable: Sequence[str],
    skill: Skill,
    n_rollouts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Monte-Carlo success probability of every candidate under ``skill`` noise."""
    out = np.zeros(len(cands), dtype=np.float32)
    for i, c in enumerate(cands):
        hits = 0
        for _ in range(n_rollouts):
            shot = execute(system, *add_noise(c, skill, rng))
            hits += is_success(shot, c.ball, c.pocket, hittable)
        out[i] = hits / n_rollouts
    return out
