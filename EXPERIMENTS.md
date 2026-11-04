# EXPERIMENTS.md

Running log of every experiment (including failures). Format follows CLAUDE.md Section 9.

## Decisions

### [2026-11-04] Environment
- Python 3.11 venv via `uv` (system python is 3.6). Versions are pinned in `requirements.txt`.
- pooltool-billiards 0.3.3 calls `np.math.atan2`, which numpy 2.x removed. Downgrading numpy
  breaks pandas/scipy, so `src/sim/__init__.py` restores the `np.math = math` alias before pooltool is imported.

### [2026-11-04] Rules (simplified 8-ball)
- We use pooltool's built-in 8-ball ruleset (`pt.get_ruleset(GameType.EIGHTBALL)`): open table,
  group assignment by first called pot, fouls (scratch, wrong first contact, no rail after contact),
  ball in hand after a foul, losing on an early or illegal 8-ball.
- Our simplifications:
  - Ball in hand: we sample 30 random legal cue positions and keep the one whose easiest candidate
    has the best geometric score (pooltool only respots the cue at the head spot).
  - No safeties and no bank/kick shots: every shot is a called ghost-ball pot attempt.
  - Break: fixed V0=7 m/s aimed at the apex ball, with skill noise. Break states are not saved.
  - Racks longer than `max_shots` (200) are discarded and counted in `generate_log.csv`.

### [2026-11-04] Skill model (execution noise std)
| level | phi std (deg) | V0 relative std | spin a,b std (ball radii) |
|---|---|---|---|
| beginner | 1.5 | 0.20 | 0.12 |
| intermediate | 0.4 | 0.07 | 0.05 |
| expert | 0.12 | 0.03 | 0.02 |
- Spin offset is clipped to a radius of 0.6 (no miscue model).
- First guess for beginner phi std was 1.0 deg. On a 4-rollout smoke game the best candidate's
  estimated success averaged 0.63, which is too high for a first-timer, so we raised it to 1.5 deg.

### [2026-11-04] Candidate set and teacher labels
- For each legal object ball x pocket: ghost-ball aim (pooltool `calc_shadow_ball_center`), cut angle,
  distances, and an occlusion/jaw "viable" flag. We rank pairs by
  `h = |cut|/90 + 0.5*(d_cue_ghost + d_obj_pocket)/L + 10*(not viable)` and keep the 8 easiest pairs.
  Each pair is combined with 3 speeds (1.2 / 2.2 / 3.5 m/s, center ball), so there are at most 24 candidates.
- Teacher label = success rate over M=16 noisy rollouts at the shooter's skill
  (success = called ball in called pocket, no scratch, legal first contact, 8-ball not pocketed early).
- The same rollouts drive the simulated player's choice (softmax over estimates, temperature 0.05),
  so labels come at no extra cost. Labels exist only at the shooter's own skill. Skill pairs are
  sampled uniformly per game, so all levels are covered.
- Splits are by game (game_id % 10: 0=test, 1=val, else train), so states from one rack never cross splits.

## Runs
