# Poster content (draft)

Every number below comes from a logged run (see EXPERIMENTS.md; the source file is given in brackets).
Figures are in `figures/poster_*.{png,pdf}` and are made by `python -m src.poster_figures`.
No em-dashes in poster text.

Title: **Skill-Aware Shot Advice and Win Prediction for Pool Beginners**
Authors: Yuri Ishitoya (yi10), Ari Prezerowitz (ap320), Arjun Naik (an182)

---

## Big figures (4)

### Fig 1. Concept: the best shot depends on who is shooting
File: `figures/poster_concept`

Caption:
> Same test table, best shot found by simulator search (64 noisy executions per candidate) at two skill levels.
> The beginner's best shot (ball 7, short) succeeds 58% of the time for a beginner; the expert's best shot gives a
> beginner only 14%. The expert's best shot (ball 2, long) succeeds 94% for an expert.

Source: results/pipeline_figs/state_teacher.npz, test state 933. This table was chosen as an example where the best
shot differs; it is not typical (see Fig 3).

### Fig 2. Recommended-shot success by skill
File: `figures/poster_shot_success`

Caption:
> Success rate of the recommended shot, re-executed 64 times with fresh noise, on 300 test tables
> (networks: mean ± std over 3 seeds). All networks beat the heuristic for beginners and intermediates and tie it for
> experts. Skill input, or a head trained on labels at all three skills, barely changes the choice.

| | Beginner | Intermediate | Expert |
|---|---|---|---|
| Simulator search (teacher), 23 s per table | 0.319 | 0.575 | 0.769 |
| Set Transformer, skill input | 0.286 ± 0.003 | 0.520 ± 0.006 | 0.703 ± 0.011 |
| Set Transformer, no skill input | 0.285 ± 0.002 | 0.521 ± 0.002 | 0.691 ± 0.001 |
| Set Transformer, 3-output head | 0.284 ± 0.001 | 0.520 ± 0.003 | 0.704 ± 0.005 |
| Geometric heuristic | 0.267 | 0.504 | 0.696 |

Source: results/shot_eval_single (results/poster_shot_success.csv).

Key message: the network reaches about 90% of the search's success at about 2 ms instead of 23 s per table, but
is at most 2 points better than the heuristic (0.7 points for experts).

### Fig 3. How much does skill change the best shot?
File: `figures/poster_skill_dependence`

Caption:
> Left: tables where the best shot uses a different ball or pocket for two skill levels, against the noise floor
> (two independent estimates at the SAME skill disagree 11% of the time). Right: success lost when a player takes
> the other skill level's best shot instead of their own (mean ± s.e. over 300 test tables).
> Best shots picked on 32 executions and scored on 32 independent ones, so there is no winner's curse.

- Different ball or pocket: beginner vs intermediate 15%, beginner vs expert 18%, intermediate vs expert 14%;
  noise floor 11%.
- Beginner plays the expert's best shot: loses 2.4 ± 0.7 points. Expert plays the beginner's best shot: loses 4.5 ± 0.9 points.

Source: results/wide_candidates/summary.json (current candidate set), results/poster_skill_dependence.json.

Key message: skill changes the best ball or pocket on only about 3 to 7% of tables beyond noise, and a beginner
loses only 2.4 points by ignoring skill. This explains why skill-aware heads cannot beat skill-agnostic ones here.

### Fig 4. Win prediction: networks vs. logistic regression
File: `figures/poster_win`

Caption:
> Test win log loss (2,255 states; networks: mean ± std over 3 seeds). A logistic regression on the two skills and
> the ball counts beats every network. The skill matchup alone (0.560) explains most of the networks' performance.

| Model | Win log loss |
|---|---|
| LR on skills + ball counts | **0.541** |
| DeepSets | 0.551 ± 0.004 |
| Set Transformer | 0.558 ± 0.012 |
| LR on skills only | 0.560 |
| MLP | 0.570 ± 0.001 |
| CNN on image | 0.611 ± 0.043 |
| LR on ball counts only | 0.621 |

Source: results/win_baselines/test_metrics.json, results/heuristic/test_metrics.json, results/seeds_summary_per_seed.csv.

---

## One-liners and small tables

### Model comparison (small table)
"Set models are within seed noise; CNN is worst and unstable."

| Model | Shot regret (vs. M=16 labels) | Win log loss | GPU ms per table |
|---|---|---|---|
| MLP | 0.116 ± 0.001 | 0.570 ± 0.001 | 0.49 |
| DeepSets | 0.110 ± 0.002 | 0.551 ± 0.004 | 0.69 |
| Set Transformer | 0.111 ± 0.003 | 0.558 ± 0.012 | 2.18 |
| CNN on image | 0.119 ± 0.001 | 0.611 ± 0.043 | 2.58 |

Regret and log loss: mean ± std over 3 seeds (results/seeds_summary.csv). ms per table: seed 0, batch size 1
(results/<run>/test_metrics.json). Simulator search: 23 s per table per skill.

### What else we tried (one line each; paired differences vs. the skill-input head, mean ± s.e., 300 test tables)
- **Less noisy labels (M=32 instead of 16):** no measurable change (+0.1 / +0.2 / -0.6 ± 0.3 points).
- **Listwise ranking loss:** beginners +1.1 ± 0.4 points, experts -2.4 ± 0.9 points (falls below the heuristic).
- **Adding spin to the shot set (search only):** experts +1.4 ± 0.4 points, beginners +0.0 ± 0.6; the best ball or
  pocket changes with skill no more often than before.

Sources: results/shot_eval_final, results/wide_candidates/summary.json.

### Failure cards (4)
Keep the four cards from the current poster.
For reference, logged failures in EXPERIMENTS.md: skill-order bug in evaluation (2026-11-05), stored racks not
reproducible from seeds (2026-11-05), simulator hang fixed with an event cap (2026-11-05), runtime of the wide-candidate
check underestimated (3.2 h vs. 0.6 to 0.9 h, 2026-11-14).

---

## Report only (not on the poster)
- Training-set size ablation: `figures/final_size_ablation`
- Label noise (stored vs. relabel): `figures/final_label_noise`
- Win probability over one held-out rack: `figures/final_win_curve`
- Random-candidate baseline (0.116 / 0.214 / 0.344), per-seed details, ECE and Brier: EXPERIMENTS.md
