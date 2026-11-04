"""Load generated states into GPU tensors and build game-level train/val/test splits."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

TABLE_W, TABLE_L = 0.9906, 1.9812  # pooltool default 7ft table (meters)
BALL_R = 0.028575


def load_states(path: str) -> Dict[str, np.ndarray]:
    """Load a merged states.npz into a dict of numpy arrays."""
    return dict(np.load(path))


def split_masks(game_id: np.ndarray) -> Dict[str, np.ndarray]:
    """Deterministic game-level split: 80% train, 10% val, 10% test (no state leakage)."""
    r = game_id % 10
    return {"train": r >= 2, "val": r == 1, "test": r == 0}


def subsample_games(game_id: np.ndarray, mask: np.ndarray, max_states: Optional[int], seed: int) -> np.ndarray:
    """Keep whole games (random order) until ``max_states`` states are reached."""
    if max_states is None or mask.sum() <= max_states:
        return mask
    rng = np.random.default_rng(seed)
    games = np.unique(game_id[mask])
    rng.shuffle(games)
    keep, n = [], 0
    counts = {g: c for g, c in zip(*np.unique(game_id[mask], return_counts=True))}
    for g in games:
        if n >= max_states:
            break
        keep.append(g)
        n += counts[g]
    return mask & np.isin(game_id, keep)


def to_tensors(d: Dict[str, np.ndarray], idx: np.ndarray, device: str) -> Dict[str, torch.Tensor]:
    """Convert raw arrays (subset ``idx``) into model-ready tensors on ``device``.

    ball_feats: (N,16,7) = [x/W, y/L, 5 role one-hot], zeroed for pocketed balls.
    glob:       (N,4)    = [on_eight, open_table, own_left/7, opp_left/7].
    """
    pos = d["pos"][idx]
    on = d["on_table"][idx]
    roles = d["roles"][idx] * on[..., None]
    xy = pos / np.array([TABLE_W, TABLE_L], np.float32)
    ball_feats = np.concatenate([xy * on[..., None], roles], -1).astype(np.float32)
    hit = d["hittable"][idx]
    on_eight = (hit.sum(1) == 1) & (hit[:, 8] == 1)
    open_table = roles[:, :, 4].sum(1) > 0
    glob = np.stack(
        [on_eight, open_table, roles[:, :, 2].sum(1) / 7.0, roles[:, :, 3].sum(1) / 7.0], 1
    ).astype(np.float32)
    meta = d["cand_meta"][idx]
    t = {
        "ball_feats": ball_feats,
        "ball_mask": on.astype(np.float32),
        "glob": glob,
        "pos": pos.astype(np.float32),
        "skill_self": d["skill_self"][idx].astype(np.int64),
        "skill_opp": d["skill_opp"][idx].astype(np.int64),
        "cand_feats": d["cand_feats"][idx],
        "cand_ball": meta[..., 0],
        "cand_pocket": meta[..., 1],
        "cand_label": d["cand_label"][idx],
        "cand_mask": d["cand_mask"][idx],
        "cand_heur": np.where(np.isfinite(d["cand_heur"][idx]), d["cand_heur"][idx], 1e9).astype(np.float32),
        "win": d["win"][idx].astype(np.float32),
        "shots_to_end": d["shots_to_end"][idx].astype(np.int64),
    }
    return {k: torch.as_tensor(v).to(device) for k, v in t.items()}


def batch(t: Dict[str, torch.Tensor], ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Index every tensor with the same batch indices."""
    return {k: v[ids] for k, v in t.items()}
