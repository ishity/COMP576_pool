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

### [2026-11-06] win-probability baselines (prior, skills-only LR, skills + counts LR)
- Config: src/models/win_baselines.py on data/main/states.npz (fit on train, report on test, 2,255 states); deterministic, no seed
- Goal: know how much of the win-head performance comes from the skill matchup alone.
- Result (test; neural rows copied from results/<run>/test_metrics.json, seed 0):

  | model | log loss | Brier | ECE | acc | acc last 3 |
  |---|---|---|---|---|---|
  | constant prior (train rate 0.468) | 0.689 | 0.248 | 0.008 | 0.547 | 0.578 |
  | LR on skills only | 0.560 | 0.188 | 0.034 | 0.702 | 0.707 |
  | LR on ball counts only (heuristic) | 0.621 | 0.216 | 0.046 | 0.673 | 0.782 |
  | LR on skills + ball counts | **0.541** | **0.180** | 0.070 | **0.761** | 0.785 |
  | MLP | 0.569 | 0.191 | 0.048 | 0.731 | 0.751 |
  | DeepSets | 0.548 | 0.182 | 0.033 | 0.736 | 0.759 |
  | Set Transformer | 0.554 | 0.186 | 0.057 | 0.724 | 0.751 |
  | Set Transformer, no skill | 0.626 | 0.219 | 0.025 | 0.653 | 0.731 |
  | CNN | 0.584 | 0.197 | 0.059 | 0.705 | 0.736 |
- Outcome: success (baseline run), negative finding for the neural win heads
- Notes: the skill matchup alone explains almost all of the neural models' win performance, and
  a 10-feature logistic regression (skills + ball counts) beats every neural model on log loss
  and Brier. With 17k training states the networks do not extract more from ball positions than
  the ball counts already give. Any win-probability claim in the poster must be made against
  this baseline, not only against the counts-only heuristic.

### [2026-11-06] win-curve perspective check (fig_pipeline_win_curve)
- Config: src/pipeline_figures.py --only win_plot (no new simulation; same held-out rack 10000000), models seed 0
- Goal: P(player 0 wins) rose after player 0's own fouls (shots 8 and 18) and dropped after
  player 1's miss at shot 0. Check whether the shooter to player-0 conversion is wrong.
- Result: the conversion is correct. Per state, `win` = 1 iff shooter == winner (player 0
  states have win=1, player 1 states win=0 in this rack, player 0 won), and
  P0 = p if shooter == 0 else 1 - p. The Set Transformer's raw outputs are p=0.70 before shot 8
  (player 0 shooting) and p=0.19 before shot 9 (player 1 shooting, ball in hand), so
  P0 goes 0.70 to 0.81. The jump comes from the model, not the plot.
  Test-set check over 1,042 turnovers (consecutive saved states of the same rack where the
  turn passes): mean change of the former shooter's predicted win probability, and the fraction of
  turnovers after which it went UP:

  | model | mean change | frac. up |
  |---|---|---|
  | Set Transformer | +0.018 | 0.70 |
  | CNN | +0.073 | 0.63 |
  | DeepSets | -0.009 | 0.52 |
  | MLP | -0.045 | 0.41 |
  | LR skills + counts | -0.027 | 0.38 |
  | LR counts only (heuristic) | -0.071 | 0.10 |
  | Set Transformer, no skill | -0.100 | 0.14 |

  The cause is visible in the Set Transformer's calibration by skill pair (test): when an
  intermediate shoots against a beginner it predicts 0.713 (observed 0.831); when the beginner
  shoots it predicts 0.242 for the beginner (observed 0.170). The observed rates show that who is
  at the table barely matters (0.831 vs 1 - 0.170 = 0.830), but the model underrates the
  stronger player when the stronger player is shooting, so every handover from the stronger to
  the weaker player looks like a gain for the stronger player.
- Outcome: partial (no plotting bug; a model inconsistency found)
- Notes: figure redrawn with DeepSets and the skills + counts LR as thin comparison lines and a
  title that states the convention (value at shot i is the estimate before shot i; the effect of
  shot i is the next segment). All three lines rise after shot 8, the Set Transformer the most.
  Possible fixes for later: a perspective-consistency loss on consecutive states, or a
  skill-pair calibration step. Added this turnover statistic to the comparison so it can be
  tracked across seeds.

### [2026-11-06] relabel at all three skills: runtime pilot
- Config: src/data/relabel.py --pilot 384, data/main/states.npz, M=16, same stored candidates, 96 workers, seed=20261106
- Goal: estimate the runtime of relabeling all 21,521 states at beginner, intermediate and expert before launching it.
- Result: 384 random states in 15.2 min wall. 69.1 CPU-s per state for all three skills (p99 476 s, a heavy tail
  from long simulations). Estimate for all states: 69.1 x 21,521 / 96 = 4.3 h (plus straggler tail).
- Outcome: success
- Notes: approved as an overnight job. Every skill is re-estimated, including the shooter's own skill, so
  the stored label vs. the new own-skill label also gives a direct M=16 vs M=16 label-noise measurement.
  The pilot output is not reused; the full run recomputes every state.

### [2026-11-06] seeds 0, 1, 2 for the main models and the size ablation
- Config: scripts/train_seeds.sh (seeds 1, 2; seed 0 = the 2026-11-05 runs), same configs as train_all.sh;
  aggregated with src/aggregate_seeds.py -> results/seeds_summary.csv, results/seeds_summary_per_seed.csv
- Goal: separate real differences from seed noise; resolve the non-monotonic size ablation.
- Result (test, mean ± std over 3 seeds; turnover = fraction of test turnovers after which the former
  shooter's predicted win probability went up, see the win-curve check):

  | run | train states | win log loss | Brier | ECE | shot regret | shot top-1 | turnover up |
  |---|---|---|---|---|---|---|---|
  | LR skills + counts (deterministic) | 17,127 | 0.541 | 0.180 | 0.070 | n/a | n/a | 0.38 |
  | heuristic (deterministic) | 17,127 | 0.621 | 0.216 | 0.046 | 0.129 | 0.182 | 0.10 |
  | Set Transformer | 17,127 | 0.558 ± 0.012 | 0.187 ± 0.005 | 0.046 ± 0.010 | 0.111 ± 0.003 | 0.33 ± 0.04 | 0.46 ± 0.22 |
  | Set Transformer, no skill | 17,127 | 0.623 ± 0.007 | 0.217 ± 0.003 | 0.025 ± 0.002 | 0.112 ± 0.002 | 0.37 ± 0.03 | 0.15 ± 0.01 |
  | MLP | 17,127 | 0.570 ± 0.001 | 0.192 ± 0.000 | 0.053 ± 0.004 | 0.116 ± 0.001 | 0.37 ± 0.06 | 0.31 ± 0.09 |
  | DeepSets | 17,127 | 0.551 ± 0.004 | 0.184 ± 0.001 | 0.035 ± 0.004 | 0.110 ± 0.002 | 0.34 ± 0.01 | 0.46 ± 0.06 |
  | CNN | 17,127 | 0.611 ± 0.043 | 0.208 ± 0.017 | 0.092 ± 0.066 | 0.119 ± 0.001 | 0.42 ± 0.10 | 0.72 ± 0.22 |
  | Set Transformer 1k | ~1,010 | 0.560 ± 0.019 | 0.188 ± 0.008 | 0.064 ± 0.025 | 0.274 ± 0.017 | 0.20 ± 0.07 | 0.34 ± 0.21 |
  | Set Transformer 3k | ~3,006 | 0.607 ± 0.014 | 0.201 ± 0.008 | 0.066 ± 0.007 | 0.126 ± 0.005 | 0.24 ± 0.07 | 0.46 ± 0.14 |
  | Set Transformer 10k | ~10,010 | 0.571 ± 0.006 | 0.193 ± 0.002 | 0.060 ± 0.025 | 0.116 ± 0.001 | 0.29 ± 0.05 | 0.55 ± 0.25 |

- Outcome: success (runs), with failures: cnn_s1 and cnn_s2 first crashed with CUDA OOM because GPUs 2 and 3
  were occupied by another (unrelated) job; rerun on GPUs 4 and 5, results above are from the reruns.
  train_seeds.sh now takes a GPUS list.
- Notes:
  * Shot regret falls monotonically with data (0.274, 0.126, 0.116, 0.111) and the seed std is small, so the
    shot-head size trend is real. Win log loss is not monotonic in any seed pattern: 1k (0.560) is as good as
    the full set. Our reading: at 1k the win head can only learn the skill-matchup prior (LR on skills alone
    gives 0.560), while at 3k to 10k it starts fitting ball positions and overfits. Combined with the LR
    result, the win head gets almost nothing from positions beyond ball counts at this data scale.
  * DeepSets and Set Transformer are within seed noise of each other on every metric; DeepSets is more
    stable (std 0.004 vs 0.012 log loss). CNN is worst and unstable (best epoch is always 2, then overfits).
  * Skill input matters for win prediction (0.558 vs 0.623) but not for regret at the shooter's own skill
    (0.111 vs 0.112), consistent with the shot evaluation.
  * The Set Transformer's turnover inconsistency (0.70 at seed 0) varies widely across seeds (0.46 ± 0.22),
    so the rise in the held-out win curve is partly a seed-0 artifact; it is not a stable property.

### [2026-11-06] shot_eval_main_v2 (reproduction + full teacher vectors + cross-skill)
- Config: src/evaluate.py, same arguments as shot_eval_main (300 test states, seed=0, teacher M=16, R=64,
  same five runs in the same order); new outputs teacher_vectors.npz, per_state_cross.csv, summary_cross.csv
- Goal: the first evaluation saved only the teacher estimate of picked candidates, so the full per-skill
  teacher vectors needed for the skill-dependence analysis were not on disk. Rerun with the same seeds to
  regenerate exactly those estimates, and add cross-skill execution on a separate random stream.
- Result: 7,191 of 7,200 rows of per_state.csv are identical to shot_eval_main; 9 rows in 2 states differ
  (one model pick changed, which shifts the random stream inside that state). summary.csv is identical to
  3 decimals (teacher 0.319 / 0.575 / 0.769, Set Transformer 0.289 / 0.525 / 0.705). 22.8 min wall, 96 workers.
- Outcome: success
- Notes: the teacher vectors here are therefore the "existing" per-skill estimates up to those 2 states.

### [2026-11-06] skill dependence of the teacher
- Config: src/skill_dependence.py on results/shot_eval_main_v2 -> results/skill_dependence/summary.json,
  figures/fig_skill_dependence.{png,pdf}
- Goal: how often does the best shot depend on skill, and how much does a player lose by playing another
  skill's best shot?
- Result (300 test states, teacher argmax over M=16 estimates):

  | pair | best candidate differs | different ball or pocket |
  |---|---|---|
  | beginner vs intermediate | 0.613 | 0.193 |
  | beginner vs expert | 0.653 | 0.237 |
  | intermediate vs expert | 0.503 | 0.173 |
  | noise floor (same skill, stored label vs eval teacher, both M=16) | 0.493 | 0.107 |

  Success lost by a player of skill X playing skill Y's best shot instead of their own, both measured with
  fresh 64-rollout estimates (mean ± s.e. over 300 states; own-best success on the diagonal):

  | executing \ best shot of | beginner | intermediate | expert |
  |---|---|---|---|
  | beginner | (0.319) | 0.005 ± 0.005 | 0.024 ± 0.007 |
  | intermediate | 0.028 ± 0.006 | (0.575) | 0.017 ± 0.005 |
  | expert | 0.069 ± 0.010 | 0.013 ± 0.004 | (0.769) |

  For reference, the same quantity from the M=16 teacher vectors themselves is 0.04 to 0.11, inflated by
  the winner's curse (the own argmax is selected on the same noisy estimates).
- Outcome: success (measurement), weak effect
- Notes: most "best shot changes" are label noise: two M=16 estimates at the SAME skill disagree on the
  argmax 49% of the time, close to the 50 to 65% seen across skills. Changes of ball or pocket are about 2x the
  noise floor (0.17 to 0.24 vs 0.11), so a real but modest fraction of states is skill-dependent. The cost of
  ignoring skill is small: a beginner playing the expert's best shot loses 2.4 points (7.6% relative), and the
  largest loss is the reverse direction (expert playing the beginner's choice, 6.9 points). This bounds what
  any skill-conditioned shot head can gain at the shooter's own skill and explains why the no-skill model
  ties the skill-conditioned one. Important framing for the poster: within our candidate set (ghost-ball
  pots, 3 speeds, no spin) skill mostly changes how likely a shot is, not which shot is best.

### [2026-11-06] relabel at all three skills (full run)
- Config: src/data/relabel.py, data/main/states.npz, M=16, stored candidates, 96 workers, seed=20261106
  -> data/main/relabel3/labels3.npz (not committed), meta in data/main/relabel3/meta.json (copied to results/relabel3_meta.json)
- Goal: labels at beginner, intermediate and expert for every state (train/val/test).
- Result: all 21,521 states relabeled, 5.75 h wall (estimate was 4.3 h; slower because the CPUs were shared
  with the seed runs at the start and with other jobs), 532 CPU-hours.
  Label noise over all states (stored label vs. new label at the shooter's own skill, both M=16):
  MAE 0.047, correlation 0.959, argmax agreement 0.544, same ball and pocket 0.881, regret of the stored argmax
  under the new label 0.059. Best candidate differs between skills on all states: beginner vs intermediate
  0.600 (ball or pocket 0.175), beginner vs expert 0.615 (0.221), intermediate vs expert 0.469 (0.158).
- Outcome: success
- Notes: matches the 300-state analysis: with M=16 the argmax is half noise, and real skill dependence of the
  ball or pocket choice is about 5 to 11 points above the 12% noise floor. M=16 is clearly too small to rank
  candidates reliably; an M ablation (32 or 64 on a subset) is the obvious next measurement.

### [2026-11-06] shot heads (a) skill input, (b) no skill input, (c) three-output head
- Config: scripts/step3_after_relabel.sh; configs/train_set_transformer.yaml, seeds 0, 1, 2.
  (a) shot_a_skill_s*: stored single-skill labels, skill input (same as set_transformer, retrained with
  all-skill test metrics); (b) shot_b_noskill_s*: stored labels, --use_skill 0; (c) shot_c_3head_s*:
  --labels3 --shot_outputs 3 (shot head without skill input, one output per skill, trained on all-skill labels;
  the win head still gets both skills). Summary: results/shot_abc_summary.csv
- Goal: does training on labels at every skill make the recommendation skill-calibrated?
- Result, test split vs. the NEW all-skill labels (mean over 3 seeds; regret against the noisy M=16 argmax):

  | head | regret beg / int / exp | MSE beg / int / exp | win log loss |
  |---|---|---|---|
  | (a) skill input | 0.108 / 0.114 / 0.102 | 0.013 / 0.024 / 0.037 | 0.559 ± 0.010 |
  | (b) no skill | 0.109 / 0.116 / 0.105 | 0.037 / 0.026 / 0.074 | 0.622 ± 0.005 |
  | (c) three outputs | 0.112 / 0.115 / 0.104 | 0.014 / 0.024 / 0.037 | 0.554 ± 0.007 |

  Fresh 64-rollout shot evaluation, results/shot_eval_abc (300 test states, same states as before, mean over
  seeds, seed std in parentheses):

  | method | beginner | intermediate | expert |
  |---|---|---|---|
  | teacher | 0.319 | 0.572 | 0.769 |
  | (a) skill input | 0.286 (0.003) | 0.521 (0.005) | 0.702 (0.011) |
  | (b) no skill | 0.285 (0.002) | 0.520 (0.003) | 0.690 (0.002) |
  | (c) three outputs | 0.284 (0.001) | 0.520 (0.003) | 0.703 (0.005) |
  | heuristic | 0.267 | 0.503 | 0.695 |
  | random | 0.116 | 0.215 | 0.360 |

  Paired differences per state (mean ± s.e.): (c) - (a): -0.002 / -0.001 / +0.001; (a) - (b): +0.001 / +0.001 /
  +0.012 ± 0.006; (c) - (b): -0.001 / +0.000 / +0.014 ± 0.007.
  Cross-skill (pick made for skill A, executed at skill B; per_state_cross.csv). Executed by a beginner:
  (a) 0.286 / 0.287 / 0.284 for picks made for beginner / intermediate / expert, (c) 0.284 / 0.284 / 0.285;
  teacher 0.319 / 0.310 / 0.295. Executed by an expert: (a) 0.688 / 0.688 / 0.702, (c) 0.685 / 0.682 / 0.703;
  teacher 0.699 / 0.756 / 0.769.
- Outcome: partial (negative for the main hypothesis)
- Notes:
  * Training on all-skill labels does not improve recommendations: (c) ties (a) at every skill. The skill input
    makes the predicted success rates calibrated per skill (MSE 0.013 vs 0.037 for beginners, 0.037 vs 0.074 for
    experts) but barely changes the ranking.
  * The only skill effect on the chosen shot is for experts (+1.2 to +1.4 points over the skill-agnostic head,
    about 2 s.e.). For beginners the models pick shots of the same quality whatever skill they are told, while
    the teacher's beginner-specific pick is worth 2.4 points more than its expert pick. The models recover none
    of that small gap.
  * Why: the skill-dependent part of the label is small relative to M=16 noise (previous entry), and every head
    stays about 3 to 7 points below the teacher at all skills. The bottleneck is ranking quality in general, not skill
    conditioning. Next options: larger M (less noisy targets), a ranking loss (listwise softmax on the label),
    or a candidate set where skill matters more (spin, safeties, bank shots).

### [2026-11-14] final round: M=32 labels and listwise ranking loss
- Config: scripts/final_round_train_eval.sh; configs/train_set_transformer.yaml + --labels3, seeds 0, 1, 2.
  shot_a_m32_s*: head (a) on "free M=32" labels (mean of the stored label and the relabel3 label at the
  shooter's skill, two independent M=16 estimates). shot_rank_m16_s* / shot_rank_m32_s*: head (a) plus a
  listwise softmax ranking loss (ListNet top-one: cross-entropy between softmax(label / 0.05) and
  softmax(logits) over valid candidates, weight 1), on M=16 or M=32 labels. The calibration term is kept;
  note it is our existing soft-label BCE, not an MSE. Evaluation: results/shot_eval_final (src/evaluate.py,
  same 300 test states, seed 0, R=64, the existing shot_a_skill_s* runs included for paired comparison, no
  cross-skill). Seed summary: results/shot_final_summary.csv
- Goal: are the weak recommendations caused by label noise (M) or by the pointwise loss?
- Result, fresh 64-rollout success (mean over seeds, seed std):

  | method | beginner | intermediate | expert |
  |---|---|---|---|
  | teacher | 0.320 | 0.574 | 0.770 |
  | (a) M=16 (reference) | 0.286 (0.003) | 0.520 (0.005) | 0.701 (0.012) |
  | (a) M=32 | 0.287 (0.006) | 0.522 (0.002) | 0.695 (0.009) |
  | (a) + rank, M=16 | 0.297 (0.002) | 0.525 (0.003) | 0.676 (0.001) |
  | (a) + rank, M=32 | 0.295 (0.003) | 0.525 (0.003) | 0.679 (0.003) |
  | heuristic | 0.267 | 0.506 | 0.693 |
  | random | 0.115 | 0.212 | 0.338 |

  Paired difference vs (a) M=16, per state (mean ± s.e.): M=32 +0.001 / +0.002 / -0.006 ± 0.003;
  rank M=16 +0.011 ± 0.004 / +0.005 ± 0.005 / -0.024 ± 0.009; rank M=32 +0.009 ± 0.004 / +0.005 ± 0.005 / -0.022 ± 0.009.
  Test win log loss (3 seeds): (a) 0.559 ± 0.010, M=32 0.553 ± 0.002, rank M=16 0.552 ± 0.007, rank M=32 0.550 ± 0.002.
  Label-based regret (vs. each run's own training label, so M=16 and M=32 numbers are not comparable):
  0.110 / 0.096 / 0.105 / 0.090; top-1 agreement 0.31 / 0.28 / 0.47 / 0.47.
- Outcome: partial
- Notes:
  * Doubling M to 32 changes nothing measurable in real success: label noise at M=16 is not the bottleneck
    for the pointwise head (or M=32 is still far too noisy to matter).
  * The ranking loss is the first change that moves the beginner number: +1.1 points (about 3 s.e.), 35% of
    the gap to the teacher closed from 0.034 to 0.023. It costs 2.4 points for experts (about 2.7 s.e.), where
    it falls below the heuristic. A plausible reason: the target temperature 0.05 is fixed while success rates
    for experts are compressed near the top (many near-ties), so the softmax target is flat and noisy for
    experts and sharp for beginners. A per-skill temperature was not tried (experiments frozen).
  * Top-1 agreement with the label jumps from 0.31 to 0.47 with the ranking loss, which mostly reflects the
    loss optimizing that quantity; real success moves much less.
