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

### [2026-11-05] main_data hang fix and final merge
- Config: configs/data_main.yaml, seed=0
- Goal: finish the in-flight games of the stopped run so the dataset is a contiguous set of game ids.
- Result: the resume run hung for 16 h on games 786 and 849 (one worker each at 100% CPU, inside
  `pt.simulate`). We added an event cap (`MAX_EVENTS=2000` in `src/sim/shots.py`). A capped rollout counts as a
  failure, and a rack whose real shot hits the cap is discarded. The cap uses no randomness,
  so every finished game is unchanged. On replay, game 786 finished and game 849 was discarded.
  Final dataset: 1,291 games, 21,521 states (12 discarded). Split: train 17,127 / val 2,139 / test 2,255.
- Outcome: success (after the fix)

### [2026-11-05] main training (train_all.sh)
- Config: configs/train_{set_transformer,mlp,deepsets,cnn}.yaml, data=data/main, seed=0;
  ablations `--use_skill 0` and `--max_train_states 1000/3000/10000`
- Goal: first full results table.
- Result (test split; shot regret is measured against the stored M=16 teacher labels):

| run | win logloss | Brier | ECE | shot MSE | shot regret | ms/state |
|---|---|---|---|---|---|---|
| heuristic (LR on counts) | 0.621 | 0.216 | 0.046 | n/a | 0.129 | n/a |
| mlp | 0.569 | 0.192 | 0.048 | 0.025 | 0.115 | 0.49 |
| deepsets | 0.548 | 0.183 | 0.033 | 0.023 | 0.111 | 0.69 |
| set_transformer | 0.554 | 0.186 | 0.057 | 0.023 | 0.109 | 2.18 |
| set_transformer_noskill | 0.626 | 0.219 | 0.025 | 0.041 | 0.114 | 2.23 |
| cnn | 0.584 | 0.197 | 0.059 | 0.028 | 0.119 | 2.58 |
| size 1k / 3k / 10k (set_transformer) | 0.549 / 0.623 / 0.565 | | | 0.087 / 0.047 / 0.025 | 0.280 / 0.132 / 0.117 | |

- Outcome: success
- Notes: CNN reached its best epoch at 2 and then overfit. The size ablation is monotonic in shot regret but not in win
  logloss (the 1k run is oddly good); one seed is not enough, so multiple seeds are needed.

### [2026-11-05] shot_eval_main (FAILED: skill-order bug)
- Command: python -m src.evaluate --data data/main/states.npz --runs (5 models) --n_states 300 --eval_rollouts 64
- Goal: success rate of each method's recommended shot under fresh noisy rollouts, per skill.
- Result: invalid. "expert" success (0.52) came out below "intermediate" (0.69). Cause: `generate.py` saved
  `data/<set>/config.yaml` with `yaml.safe_dump` (sorted keys), which reordered the skills to beginner, expert,
  intermediate. `evaluate.py` reads that copy, so a model's index-1 (intermediate) pick was executed with expert
  noise and vice versa. Training data was NOT affected: generation reads `configs/*.yaml` directly
  (confirmed: same content, only the key order differs).
- Outcome: failed. Output kept in results/shot_eval_main_BUGGY_skill_order/ for the record; do not use it.
- Notes: fix: `skills_from_config` now enforces the order beginner, intermediate, expert regardless of dict
  order; configs are saved with `sort_keys=False`; restored data/*/config.yaml from configs/. Re-run below.

### [2026-11-05] shot_eval_main (re-run after skill-order fix)
- Command: python -m src.evaluate --data data/main/states.npz --runs results/{set_transformer,set_transformer_noskill,cnn,mlp,deepsets} --n_states 300 --teacher_rollouts 16 --eval_rollouts 64, seed=0
- Goal: success rate of each method's recommended shot, re-executed with fresh skill noise.
- Result (mean success over 300 test states; results/shot_eval_main/summary.csv):

| method | beginner | intermediate | expert |
|---|---|---|---|
| teacher (sim search, 22.7 s/state/skill) | 0.319 | 0.575 | 0.769 |
| set_transformer | 0.289 | 0.525 | 0.705 |
| set_transformer_noskill | 0.287 | 0.524 | 0.691 |
| mlp | 0.288 | 0.520 | 0.704 |
| deepsets | 0.290 | 0.524 | 0.684 |
| cnn | 0.280 | 0.520 | 0.686 |
| heuristic | 0.267 | 0.504 | 0.695 |
| random | 0.114 | 0.216 | 0.341 |

- Outcome: success
- Notes: skill order is now correct (expert > intermediate > beginner). The learned models beat the heuristic for
  beginner and intermediate, but not clearly for expert, and they remain 0.03 to 0.07 below the teacher. Skill
  conditioning helps only slightly here (0.705 vs 0.691 for expert, ~equal for beginner), so this is not yet the
  strong "skill matters" result. Labels exist only at the shooter's own skill, which limits what the
  shot head can learn about skill; see next steps.

### [2026-11-05] replay of stored test racks (FAILED: rack not reproducible)
- Goal: replay stored test games from their seeds to recover per-shot foul information for the win-curve figure.
- Result: all 8 replayed racks differed from storage from the first saved shot (max position error ~1.6 to 2.0 m).
  Cause: pooltool's `generate_layout` assigns ball ids with python's `random.choice`, and `play_game` seeded only
  numpy. Racks in data/main (and data/smoke) can therefore NOT be regenerated from their seeds, although every
  stored state, label and outcome is real.
- Outcome: failed
- Notes: fix: `play_game` now also calls `random.seed(seed)`, and every new state stores `shot_legal` and
  `shot_turn_over` (pooltool's verdict), so future data does not need replays. Check: 8 new racks, each played twice,
  were identical (results/pipeline_figs/heldout_games.json, "deterministic": true). `merge` keeps only fields
  present in all games, so old and new games can still be merged.

### [2026-11-05] pipeline explainer figures
- Command: python -m src.pipeline_figures --data data/main/states.npz --shot_eval results/shot_eval_main --run set_transformer
  (run as `--only win` then `--only state`; then `--only win_plot` redraws Fig 2 from saved data after a tick fix)
- Goal: three figures that explain the pipeline using only real data. Raw numbers are in results/pipeline_figs/.
- Result:
  - figures/fig_pipeline_candidates: test state 933 (test game 60, shooter = beginner). 8 (ball, pocket) pairs x 3 speeds
    = 24 candidates. Each pair's line is colored by its best-speed stored teacher label (M=16). Best = ball 7 to the
    left-top pocket, medium speed, label 0.69.
  - figures/fig_pipeline_skill_pair: same state, all 24 candidates re-estimated with 64 fresh rollouts at
    beginner and expert. Beginner best = ball 7, left-top, medium (0.58 at beginner; the expert's pick gives a
    beginner only 0.14). Expert best = ball 2, left-bottom, firm (0.94 at expert; the beginner's pick gives an
    expert 0.86). Selection: among the 300 shot-eval states, those whose logged teacher pick differed between
    beginner and expert in ball or pocket, sorted by number of candidates (top 12). We kept the first one whose
    argmax still differed after re-estimation. The re-estimate (0.58) differs from the stored M=16 label (0.69),
    which shows the M=16 label noise.
  - figures/fig_pipeline_win_curve: because stored racks cannot be replayed (above), this uses a NEW held-out rack
    (game id 10000000, same config and pipeline, not in any training data). Player 0 = intermediate, player 1 = beginner.
    22 shots after the break: 11 pots, 6 misses, 5 fouls (3 scratches, 2 wrong first contact). We chose the longest
    of 8 new racks that contains at least one pot, miss and foul. Model = set_transformer (seed 0). P(player 0 wins)
    goes from 0.72 to 0.85, and player 0 won.
- Outcome: success
- Notes: Fig 2's rack is held out, but it is not a member of the stored test split. Say so in any caption.
