# PoolCoach: data generation and training details

This document describes exactly how the dataset is generated and how the models are trained.
Every setting below is taken from the code and configs in this repository; every number is from a logged run
(see EXPERIMENTS.md). Code paths are given so each step can be checked.

Contents:
1. Data generation
2. A concrete example (one stored table state)
3. Training
4. Evaluation protocol

---

## 1. Data generation

Code: `src/sim/shots.py`, `src/sim/game.py`, `src/data/generate.py`, `src/data/relabel.py`.
Config: `configs/data_main.yaml`.

### 1.1 Simulator and rules
- Physics: pooltool-billiards 0.3.3 (Kiefl, JOSS 2024), default 7 ft table (0.9906 m x 1.9812 m), ball radius 28.575 mm.
- Rules: pooltool's built-in 8-ball ruleset (open table, group assignment, fouls, ball in hand, losing on an early
  or illegal 8-ball).
- Our simplifications:
  - Every shot is a called ghost-ball pot attempt. There are no safeties and no bank or kick shots.
  - Break: fixed 7 m/s shot aimed at the apex ball, with the breaker's skill noise. Break states are not saved.
  - Ball in hand: 30 random legal cue positions are tried, and the one whose easiest shot has the best geometric
    score is kept.
  - Racks longer than 200 shots are discarded. A simulation that exceeds 2,000 events counts as a failed shot, and a
    rack whose real shot hits that cap is discarded.

### 1.2 Skill model
A player's skill is execution precision only. The executed shot is the intended shot plus Gaussian noise:

| Skill | Aim angle phi, std (deg) | Speed V0, relative std | Spin offsets a, b, std (ball radii) |
|---|---|---|---|
| Beginner | 1.5 | 0.20 | 0.12 |
| Intermediate | 0.4 | 0.07 | 0.05 |
| Expert | 0.12 | 0.03 | 0.02 |

- Speed is clipped to 0.2 to 8 m/s. The spin offset is clipped to a radius of 0.6 (no miscue model).
- The expert aim noise is close to the 0.125 deg used in the computational pool tournaments (Archibald et al. 2010).
- The beginner aim noise was first 1.0 deg. In a pilot, the best candidate's success averaged 0.63, too high for a
  first-time player, so it was raised to 1.5 deg.

### 1.3 Candidate shots
For each state, `generate_candidates` builds the shots a player may choose from:
1. For every legal object ball x pocket pair, compute the ghost-ball aim point (pooltool `calc_shadow_ball_center`),
   the cut angle, the cue-to-ghost and ball-to-pocket distances, and a "viable" flag
   (no ball blocking the path, no pocket jaw in the way). Pairs with a cut angle above 85 deg are dropped.
2. Rank the pairs by a geometric difficulty score
   `h = |cut| / 90 + 0.5 * (d_cue_ghost + d_ball_pocket) / L + 10 * (not viable)` (lower is easier),
   and keep the 8 easiest pairs.
3. Combine each pair with 3 speeds: 1.2 m/s (soft), 2.2 m/s (medium), 3.5 m/s (firm), center ball (a = b = 0).

So a state has at most 8 x 3 = 24 candidates (15.95 on average; 7,982 of 21,521 states have all 24).

Each candidate is described by 17 numbers (`cand_feats`):

| # | Feature | # | Feature |
|---|---|---|---|
| 0 | abs(cut angle) / 90 | 9 | cos(aim angle) |
| 1 | cue to ghost-ball distance / L | 10, 11 | object ball x / W, y / L |
| 2 | object ball to pocket distance / L | 12, 13 | potting point x / W, y / L |
| 3 | viable (0/1) | 14, 15 | ghost-ball x / W, y / L |
| 4 | corner pocket (0/1) | 16 | object ball is the 8-ball (0/1) |
| 5-7 | speed one-hot (soft, medium, firm) | | |
| 8 | sin(aim angle) | | |

It also stores `cand_meta` = (ball index, pocket index, speed index) and `cand_params` = intended (phi, V0, a, b).

### 1.4 Teacher labels (simulator search)
- A candidate **succeeds** if the called ball enters the called pocket, the cue ball is not pocketed, the first ball
  hit is legal, and the 8-ball is not pocketed early (`is_success`).
- **Shot label** = success rate over M = 16 noisy executions at the shooter's skill (`estimate_success`).
  This search costs about 23 s per table per skill level on one CPU core.
- **Win label** = 1 if the player about to shoot eventually wins the rack, else 0.

### 1.5 Simulated games
- Each game draws a skill level for each of the two players uniformly at random (all 9 pairings occur).
- Before every shot, the shooter estimates all candidates with 16 rollouts and picks one by a softmax over the
  estimates with temperature 0.05, so the best shot is chosen most of the time but games still vary.
- The chosen shot is executed with the shooter's noise, pooltool applies the 8-ball rules, and play continues
  until the rack ends.
- The same rollouts are used for the player's choice and for the labels, so labels come at no extra cost.

### 1.6 Dataset
- Seed 0, 96 CPU workers. Generation was stopped at the 2-hour mark (CPU-bound) and the contiguous prefix of finished
  game ids was kept.
- **1,291 games, 21,521 states** (12 racks discarded). States per game: median 16.
- Split by game id (`game_id % 10`: 0 = test, 1 = validation, else train), so no rack appears in two splits:
  **train 17,127 / validation 2,139 / test 2,255** states.
- Shooter skill: beginner 6,733, intermediate 7,455, expert 7,333 states. 55.4% of states have win label 1.
- Mean shot label over all candidates: beginner 0.107, intermediate 0.206, expert 0.294.
- Known limitation: these racks cannot be regenerated from their seeds (pooltool also uses Python's `random`, which
  was not seeded at the time; fixed later). Every stored state, label and outcome is real.

### 1.7 Relabeling at all three skills
The generated labels exist only at the shooter's own skill. `src/data/relabel.py` re-estimated every candidate of every
state at beginner, intermediate and expert skill (M = 16 each, seed 20261106), producing `cand_label3` with shape
(21,521, 24, 3). It took 5.75 h wall time (532 CPU-hours).
The relabel also measures label noise: two independent M = 16 estimates at the same skill have mean absolute difference
0.047 and correlation 0.959, but agree on the best candidate only 54.4% of the time (same ball and pocket 88.1%).

### 1.8 Stored arrays (`data/main/states.npz`)

| Key | Shape | Meaning |
|---|---|---|
| `pos` | (N, 16, 2) | ball x, y in meters (index 0 = cue ball, index i = ball i) |
| `on_table` | (N, 16) | 1 if the ball is on the table |
| `roles` | (N, 16, 5) | one-hot role from the shooter's view: cue, 8-ball, own group, opponent group, open table |
| `hittable` | (N, 16) | balls the shooter may legally hit first |
| `shooter`, `skill_self`, `skill_opp` | (N,) | shooter id and both skill levels (0 beginner, 1 intermediate, 2 expert) |
| `shot_index` | (N,) | shot number in the rack |
| `cand_feats` | (N, 24, 17) | candidate features (Section 1.3), zero-padded |
| `cand_meta`, `cand_params` | (N, 24, 3), (N, 24, 4) | (ball, pocket, speed) indices; intended (phi, V0, a, b) |
| `cand_label` | (N, 24) | teacher success rate at the shooter's skill (M = 16) |
| `cand_mask` | (N, 24) | 1 for real candidates, 0 for padding |
| `cand_heur` | (N, 24) | geometric difficulty score h |
| `chosen`, `chosen_success` | (N,) | candidate the player took, and whether it went in |
| `win` | (N,) | win label |
| `shots_to_end` | (N,) | shots left in the rack |
| `game_id` | (N,) | rack id (used for the split) |

---

## 2. A concrete example: test state 933

Test game 60, beginner vs. beginner, player 0 to shoot, solids (balls 1 to 7) to pot. Player 0 went on to win
(`win` = 1), with 17 saved shots left in the rack. Balls 1, 6 and 9 are already pocketed.

**Ball input** (positions in meters; role from the shooter's view):

| Ball | x | y | Role | | Ball | x | y | Role |
|---|---|---|---|---|---|---|---|---|
| cue | 0.322 | 1.522 | cue | | 10 | 0.950 | 1.705 | opponent |
| 2 | 0.068 | 0.762 | own | | 11 | 0.563 | 0.443 | opponent |
| 3 | 0.888 | 1.393 | own | | 12 | 0.961 | 1.373 | opponent |
| 4 | 0.622 | 0.955 | own | | 13 | 0.391 | 1.861 | opponent |
| 5 | 0.869 | 0.226 | own | | 14 | 0.554 | 0.907 | opponent |
| 7 | 0.176 | 1.875 | own | | 15 | 0.902 | 1.617 | opponent |
| 8 | 0.380 | 1.726 | 8-ball | | | | | |

**Candidates**: 8 pairs x 3 speeds = 24 candidates. The six with the highest stored label:

| Candidate | Ball to pocket | Speed | Aim phi (deg) | Stored label (M=16, beginner) | Relabel: beginner / intermediate / expert |
|---|---|---|---|---|---|
| 4 | 7 to left top | medium 2.2 m/s | 106.3 | 0.6875 (11/16) | 0.69 / 0.88 / 0.94 |
| 3 | 7 to left top | soft 1.2 m/s | 106.3 | 0.625 | 0.69 / 0.94 / 1.00 |
| 5 | 7 to left top | firm 3.5 m/s | 106.3 | 0.375 | 0.69 / 0.88 / 0.88 |
| 1 | 2 to left bottom | medium | 250.4 | 0.25 | 0.06 / 0.38 / 0.88 |
| 0 | 2 to left bottom | soft | 250.4 | 0.1875 | 0.19 / 0.69 / 0.88 |
| 18 | 4 to right side | soft | 298.7 | 0.125 | 0.13 / 0.00 / 0.06 |

Features of candidate 4 (ball 7, left top, medium):
`[0.51, 0.172, 0.10, 1, 1, 0, 1, 0, 0.96, -0.281, 0.177, 0.947, 0.0, 0.993, 0.228, 0.933, 0]`
(cut angle about 46 deg, viable, corner pocket, medium speed, ...).

What the example shows:
- Ball 7 to the left top pocket is the best choice for a beginner. Ball 2 to the left bottom pocket is a long shot that
  a beginner makes 6 to 19% of the time but an expert makes 88%.
- Labels are noisy: candidates 3, 4 and 5 have stored labels 0.625 / 0.6875 / 0.375 but the same relabel value 0.69 at
  beginner skill. With 16 executions the standard error of a label is up to 12.5 points.
- The poster's concept figure uses this state with 64 fresh executions per candidate: the beginner's best shot succeeds
  58% for a beginner (the expert's best shot gives a beginner 14%), and the expert's best shot succeeds 94% for an expert.

---

## 3. Training

Code: `src/data/dataset.py`, `src/models/nets.py`, `src/train.py`. Configs: `configs/train_*.yaml`.
Launch: `scripts/train_all.sh` (seed 0), `scripts/train_seeds.sh` (seeds 1 and 2), one A40 GPU per run.

### 3.1 Model inputs (`to_tensors`)
- **Ball features** (16, 7) per state: [x / W, y / L, 5 role one-hot]; pocketed balls are zeroed and masked.
- **Global features** (4): [shooting at the 8-ball, open table, own balls left / 7, opponent balls left / 7].
- **Skills**: shooter skill and opponent skill, each a learned 16-dim embedding.
  For the skill-agnostic ablation (`--use_skill 0`) both embeddings are replaced by zeros.
- **Candidates** (24, 17) features + object-ball index + pocket index, with the candidate mask.
- **Image** (CNN only): a (5, 256, 128) top-down rendering, rendered on the GPU; each ball is an anti-aliased disk in the
  channel of its role.

### 3.2 Architecture
Every model has a shared encoder and two heads:
- **Encoder** gives a state embedding z (256-dim) and one token per ball.
- **Win head**: MLP [z, shooter skill, opponent skill] -> 256 -> 1 logit.
- **Shot head**: for each candidate k, MLP [z, token of k's object ball, pocket embedding (8-dim), 17 candidate
  features, shooter skill] -> 256 -> 256 -> 1 logit. The recommendation is the argmax over valid candidates.
- **Three-output variant** (`--shot_outputs 3`): the shot head gets no skill input and outputs 3 logits per candidate,
  one per skill level, trained on `cand_label3`.

| Model | Encoder | Parameters | GPU ms per table (batch 1) |
|---|---|---|---|
| MLP | flattened 16 x (7 + mask) + global + skills -> 512 -> 512 -> 256 | 697k | 0.49 |
| DeepSets | per-ball MLP 7 -> 128 -> 128 -> 128, masked sum + max pooling, MLP -> 256 | 424k | 0.69 |
| Set Transformer (main) | ball tokens + one global token (global features + skills), 4 pre-norm Transformer layers, d = 128, 4 heads, FFN 256, padding mask for pocketed balls; global-token output -> z | 851k | 2.18 |
| CNN | ResNet-18 (5 input channels, no pretraining) on the rendered image, + global + skills -> 256 | 11.6M | 2.58 |

All activations are GELU.

### 3.3 Losses
- Win: binary cross-entropy against the win label.
- Shot: binary cross-entropy with the soft label (the success rate in [0, 1]) for every valid candidate, averaged over
  valid candidates. The target is the stored label at the shooter's skill (M = 16), or `cand_label3` for the
  three-output head.
- Total loss = win BCE + 1.0 x shot BCE.
- Final-round variants:
  - **M = 32 labels** (`--shot_label m32`): target = mean of the stored label and the relabel label at the shooter's
    skill (two independent M = 16 estimates).
  - **Listwise ranking loss** (`--rank_weight 1 --rank_tau 0.05`): adds the ListNet top-one loss, the cross-entropy
    between softmax(label / 0.05) and softmax(shot logits) over each state's valid candidates. The BCE term is kept
    for calibration.

### 3.4 Optimization

| Setting | MLP | DeepSets | Set Transformer | CNN |
|---|---|---|---|---|
| Optimizer | AdamW, weight decay 1e-4 | same | same | same |
| Peak learning rate | 1e-3 | 1e-3 | 5e-4 | 5e-4 |
| Schedule | OneCycle, 5% warm-up | same | same | same |
| Batch size | 512 | 512 | 512 | 256 |
| Epochs | 40 | 40 | 40 | 25 |
| Gradient clipping | norm 1.0 | same | same | same |

- The whole split is loaded onto the GPU; each epoch uses a random permutation of the training states.
- **Model selection**: after every epoch, validation score = win log loss + 10 x shot MSE (lower is better); the
  checkpoint with the best score is kept and evaluated once on the test split. Best epochs at seed 0: MLP 4,
  DeepSets 32, Set Transformer 18, CNN 2 (the CNN overfits after epoch 2 in every seed).
- Training is fast: under 1 minute for the set models and about 4 minutes for the CNN per run (17,127 training states).
- **Seeds**: every reported model is trained with seeds 0, 1 and 2; results are mean +- std over seeds.
- **Training-set size ablation** (`--max_train_states 1000 / 3000 / 10000`): whole games are sampled until the
  state budget is reached, so no rack is split.

### 3.5 Runs reported on the poster

| Run | What changes | Command flags |
|---|---|---|
| Set Transformer, skill input (a) | main model | `--config configs/train_set_transformer.yaml` |
| Set Transformer, no skill input (b) | skill embeddings zeroed | `--use_skill 0` |
| Set Transformer, 3-output head (c) | one output per skill, all-skill labels | `--labels3 data/main/relabel3/labels3.npz --shot_outputs 3` |
| MLP, DeepSets, CNN | encoder | `--config configs/train_{mlp,deepsets,cnn}.yaml` |
| M = 32 labels / ranking loss | target or loss | `--shot_label m32`, `--rank_weight 1` |

### 3.6 Baselines (no deep learning)
- **Geometric heuristic** for shots: the candidate with the lowest difficulty score h at medium speed, the same for
  every skill level.
- **Logistic regression** for win prediction (scikit-learn, fit on train): constant prior; skills only (one-hot of
  both skills, 6 features); ball counts only (own balls left, opponent balls left, open table, shooting at the 8-ball);
  and skills + ball counts (10 features). Code: `src/models/heuristic.py`, `src/models/win_baselines.py`.

---

## 4. Evaluation protocol (shot recommendations)

Code: `src/evaluate.py`.
- 300 test states, sampled once with seed 0 and used for every method.
- Each method's recommended shot is re-executed **64 times with fresh noise** at the evaluated skill level, using random
  streams independent of the training labels. The score is the mean success rate.
- The teacher (simulator search) re-estimates all candidates with M = 16 and takes the argmax; the random baseline picks
  a valid candidate uniformly.
- Paired comparisons use per-state differences between methods (mean +- standard error over the 300 states).
- Final numbers (beginner / intermediate / expert, `results/shot_eval_single`):
  teacher 0.319 / 0.575 / 0.769; Set Transformer with skill input 0.286 / 0.520 / 0.703;
  geometric heuristic 0.267 / 0.504 / 0.696; random 0.116 / 0.214 / 0.344.
