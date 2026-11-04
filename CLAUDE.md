# CLAUDE.md: PoolCoach (ELEC/COMP 576 Final Project)

## 1. Project summary

We are building a deep learning model for 8-ball pool that, given the current table state and the shooter's skill level:

1. **Predicts win probability**: the probability that the current shooter eventually wins the rack, updated after every shot.
2. **Recommends a shot**: the shot (aim angle, speed, spin) with the highest success rate *for a player of that skill level*, not the shot a perfect player would choose.

Motivation: beginners (e.g., international students trying pool for the first time) lose quickly without understanding why. The tool should tell them "how am I doing" and "what is a realistic good shot for me."

Gap vs. prior work (keep this framing consistent in all writing):
- PickPocket / CueCard: expert-level simulator search, slow, no win estimate for humans.
- PoolLiveAid / Sousa et al. 2016 (AR): only project the trajectory of the shot already being aimed.
- pix2pockets (SCIA 2025): image-based shot suggestion with RL, but one "optimal" shot regardless of skill, no win probability.
- NFL win-probability models (Lock & Nettleton 2014): exist in other sports, not in pool.
- **Our contribution:** one learned model that jointly outputs live win probability and skill-calibrated shot recommendations, amortizing expensive simulator search into a fast network.

## 2. Deadlines

| Date | Deliverable |
|---|---|
| Dec 2, 11:59 pm | Team availability form |
| **Dec 5 (internal)** | Poster content and all figures frozen |
| Dec 8, 5 pm | Poster submitted (Canvas + Gathertown) |
| Dec 9 | Poster presentation (5 min talk + 5 min Q&A) |
| Dec 15, 11:59 pm | Final report (IEEE, 4 to 6 pages excl. refs, at least 3 figures, GitHub link) |

Weekly milestones:
- **By Nov 11:** data generation pipeline works end to end; smoke-test dataset exists.
- **By Nov 22:** all baselines and the main model trained once; results table (preliminary).
- **By Dec 2:** ablations, error analysis, final figures.
- **Dec 9 to 15:** report writing, optional sim-to-real demo.

## 3. Grading priorities (drive every decision by these)

- Background/Motivation (30 poster / 20 report) and Experimentation (30) dominate the grade.
- **Every experiment must be logged, including failures.** Failed or weak runs are worth points if documented.
- **Code must be on GitHub with the link in the report**, or Experimentation and Results (50 pts) get zero.
- Required figures: concept figure, pipeline/model figure, at least one main results figure.
- State-of-the-art accuracy is NOT required. Clear comparisons and honest analysis are.

## 4. Environment

- Compute: **one node with NVIDIA A40 GPUs.** Simulation is CPU-bound; parallelize data generation across CPU cores with multiprocessing.
- Python **3.11** (pooltool requires >=3.9, <3.13).
- Simulator: `pooltool-billiards` (pip). **Pin the exact version in `requirements.txt`** once installed.
- Deep learning: PyTorch. Config: YAML files under `configs/`. Logging: CSV + Weights & Biases if available, otherwise TensorBoard.
- Always set and record random seeds.

## 5. Repository layout

```
pool-coach/
  CLAUDE.md
  README.md                 # setup, how to reproduce every figure
  EXPERIMENTS.md            # running log of ALL experiments (see Section 9)
  requirements.txt
  configs/                  # one YAML per experiment
  src/
    sim/                    # pooltool wrappers, noisy players, game loop
    data/                   # dataset generation, labeling, rendering, Dataset classes
    models/                 # model definitions
    train.py
    evaluate.py
  scripts/                  # shell scripts to launch generation/training/eval
  data/                     # generated data (gitignored; keep a small sample committed)
  results/                  # metrics CSVs, checkpoints (gitignored)
  figures/                  # final PNG/PDF figures for poster and report
  notebooks/                # exploration only; nothing the results depend on
```

## 6. Technical specification

### 6.1 Simulated players (skill model)
- A player chooses a shot, then the executed shot = intended shot + Gaussian noise on aim angle (phi), speed (V0), and spin offsets (a, b).
- Define at least 3 skill levels via noise std: `beginner`, `intermediate`, `expert`. Store the numeric std values in config and report them in the paper.
- Shot selection policy for simulated players: evaluate candidate shots (see 6.3) and pick the best under their own noise, with some extra randomness so that games are diverse.

### 6.2 Game generation
- Play full 8-ball racks between two simulated players (same or different skill levels).
- Save every pre-shot state: ball positions, ball groups (solids/stripes/open), whose turn, both players' skill levels, shot index.
- Label each state with the eventual winner of the rack (from the current shooter's perspective).
- Enforce basic 8-ball rules (fouls, ball-in-hand, 8-ball win/loss). If pooltool provides a ruleset for 8-ball, use it; otherwise implement a simplified, documented ruleset.
- Target: a smoke-test set (~1k states) first, then scale to 10k / 50k / 100k+ states for the data-size ablation.

### 6.3 Shot-recommendation labels
- Generate a candidate set per state: for each legal object ball and pocket, compute the ghost-ball aim angle; combine with a few speed levels (and optionally a few spin options). Add simple bank shots only if time allows.
- For each candidate, run M noisy rollouts at the target skill level and estimate success probability (object ball pocketed, no foul).
- Label = success probability for every candidate (store all, not only the argmax). This "teacher" is the simulator search baseline.
- Keep M moderate (start with 16 to 32) and measure label noise; document the trade-off.

### 6.4 Inputs
- **Representation A (set):** list of ball features (x, y, group one-hot, is_cue, is_8), padded with a mask.
- **Representation B (image):** rendered top-down table image (e.g., 128x256), balls drawn as colored circles. Write our own simple renderer if pooltool rendering is slow or hard to use headless.
- Skill level is given as an input (embedding or scalar).

### 6.5 Models
Each model has two heads sharing an encoder:
- Win head: sigmoid, binary cross-entropy against eventual winner.
- Shot head: scores each candidate shot (candidate features + state encoding), trained to regress the teacher success probabilities. Recommend the argmax.

Models to compare:
1. **Geometric heuristic** (no learning): pick the straightest, shortest shot. Win probability from a simple logistic regression on ball counts.
2. **MLP** on flattened, padded set representation.
3. **Set Transformer / DeepSets** on the set representation (main model).
4. **CNN (e.g., ResNet-18)** on the rendered image.
5. **Simulator search teacher**: upper bound for shot quality, report its wall-clock time.

### 6.6 Evaluation metrics
- Win probability: log loss, Brier score, calibration (ECE + reliability diagram), accuracy at the final shots of a rack.
- Shot recommendation: actual success rate when the recommended shot is executed with noise at each skill level (via fresh simulator rollouts), compared to the teacher and the heuristic.
- Speed: inference time per state vs. simulator search time.

### 6.7 Required ablations
- Training set size (e.g., 10k / 50k / 100k+).
- Skill conditioning on vs. off (does a skill-agnostic model recommend shots that beginners miss?). This is the key result supporting our novelty claim.
- Set vs. image representation.

### 6.8 Error analysis
- Collect failure cases: bank/kick shots, clustered balls, long shots, late-game states.
- Save example tables with predicted vs. true outcomes as figures.

### 6.9 Optional (only after everything above is done)
- Sim-to-real: photograph a real table, estimate ball positions (manual clicks are acceptable, or reuse pix2pockets' detector), run the model, and show qualitative recommendations.
- Zero-shot VLM baseline on rendered images, as an extra comparison.

## 7. Figures to produce (save to `figures/`, both PNG and PDF)
1. Concept figure: beginner at the table -> model -> win-probability meter + recommended shot overlay.
2. Pipeline figure: noisy players -> simulated games -> labels -> model with two heads.
3. Main results: shot success rate by skill level for each method (grouped bar chart).
4. Win-probability curve across one example game, plus a reliability diagram.
5. Data-size ablation plot.
6. Qualitative examples, including failure cases.

## 8. How Claude should work in this repo
- **Check the pooltool API in its docs or source before using it.** Do not guess function names or argument conventions; write a tiny test script first.
- Start every new component with a small smoke test (a few games, a few training steps) before launching long jobs.
- **Never fabricate or estimate results.** Every number in figures, tables, or text must come from a logged run in `results/`.
- Ask before starting any job expected to run longer than ~2 hours, and report the estimated runtime first.
- Commit to git after each working milestone with a descriptive message. Never commit `data/` or checkpoints.
- Keep code simple and readable; graders will open the GitHub repo. Add docstrings to public functions.
- When a design decision is made (rules simplification, noise values, candidate set), record it in `EXPERIMENTS.md` under "Decisions".
- When writing English text for the poster or report, do not use em-dashes.

## 9. EXPERIMENTS.md format
Append one entry per run, including failed or abandoned runs:

```
### [YYYY-MM-DD] <short name>
- Config: configs/<file>.yaml, seed=<n>
- Goal: <what we wanted to learn>
- Result: <key metrics>
- Outcome: success / partial / failed
- Notes: <why it worked or failed, what we changed next>
```

## 10. References (use these exact citations)
1. E. Kiefl, "Pooltool: A Python package for realistic billiards simulation," JOSS, 9(101), 7301, 2024.
2. M. Smith, "PickPocket: A computer billiards shark," Artificial Intelligence, 171, 2007.
3. C. Archibald, A. Altman, M. Greenspan, Y. Shoham, "Computational pool: A new challenge for game theory pragmatics," AI Magazine, 31(4), 33-41, 2010.
4. R. Alves, L. Sousa, J. M. F. Rodrigues, "PoolLiveAid: Augmented reality pool table to assist inexperienced players," WSCG, 184-193, 2013.
5. L. Sousa, R. Alves, J. M. F. Rodrigues, "Augmented reality system to assist inexperienced pool players," Computational Visual Media, 2(2), 183-193, 2016.
6. D. Lock, D. Nettleton, "Using random forests to estimate win probability before each play of an NFL game," JQAS, 10(2), 197-205, 2014.
7. K. Fragkiadaki, P. Agrawal, S. Levine, J. Malik, "Learning visual predictive models of physics for playing billiards," ICLR, 2016.
8. J. M. Schiøtt, V. S. Petersen, D. P. Papadopoulos, "pix2pockets: Shot suggestions in 8-ball pool from a single image in the wild," SCIA, 2025.