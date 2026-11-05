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

### [2026-11-04] smoke_data
- Config: configs/data_smoke.yaml, seed=0
- Goal: check that the generation pipeline works end to end and measure cost.
- Result: 24 games, 0 discarded, 369 states, 10.9 min wall on 24 workers (16.6 s per state per worker).
  Mean teacher success of the chosen shot: beginner 0.42 / intermediate 0.55 / expert 0.70.
  17 of 24 racks ended with a legal 8-ball; the rest ended with an early or illegal 8-ball loss.
- Outcome: success
- Notes: racks are short (median 15 saved shots) because there are no safeties. Even experts
  get only 0.73 best-candidate success because there is no position play, so the cue often ends up in a hard spot.

### [2026-11-04] smoke_train (mlp, deepsets, set_transformer, cnn, heuristic)
- Config: configs/train_*.yaml with --epochs 3, data=data/smoke, seed=0
- Goal: confirm that every model trains and writes metrics/checkpoints.
- Result: all four nets run (GPU inference 0.6 to 3.3 ms/state at batch size 1). Metrics are meaningless (~300 training states, 1 step/epoch).
- Outcome: success (pipeline check only)

### [2026-11-04] smoke_shot_eval
- Command: python -m src.evaluate --data data/smoke/states.npz --runs results/smoke_set_transformer --n_states 8 --eval_rollouts 16
- Goal: check fresh-rollout evaluation of recommended shots.
- Result: runs in 3.0 min on 8 workers. Teacher search (24 candidates x 16 rollouts) takes 22.6 s per state per skill.
- Outcome: success (pipeline check only; 8 states, so the numbers are noise)

### [2026-11-04] main_data generation, throughput issue
- Config: configs/data_main.yaml, seed=0 (2000 games planned)
- Goal: about 30k states for the main training run.
- Result: measured 2.45 states/s on 96 workers (21 s per state per worker; the node has 52 physical
  cores with hyperthreading, so 96 workers do not give 96x speedup). The full 2000 games would take ~3.4 h.
- Outcome: partial. To respect the "ask before >2 h jobs" rule we stopped at the 2 h mark and kept only
  the contiguous prefix of finished game ids (`--merge_only`). This avoids biasing toward short racks.
- Notes: the larger 50k/100k sets need explicit approval (~5.7 h for 50k, ~11 h for 100k at this rate).

### [2026-11-04] dev_partial (Set Transformer / MLP / heuristic on partial main data)
- Config: configs/train_set_transformer.yaml, configs/train_mlp.yaml, data=data/partial (first finished games), seed=0
- Goal: sanity-check learning before the main run.
- Result (test split):
  - 1.3k states: Set Transformer win logloss 0.558, shot regret 0.269. Heuristic shot regret 0.135.
    The model was worse than the heuristic at shot selection and overfit (val logloss rose after epoch 9).
  - 3.7k states: Set Transformer logloss 0.544 / regret 0.116; MLP 0.525 / 0.109; heuristic 0.633 / 0.127.
- Outcome: partial (development only; not reported as results)
- Notes: shot quality improves quickly with data. With ~1k states the nets cannot even match the geometric heuristic.
