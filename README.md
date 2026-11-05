# PoolCoach: skill-aware win probability and shot recommendation for 8-ball pool

ELEC/COMP 576 final project. A single learned model takes the table state and the shooter's
skill level and outputs (1) the live probability that the shooter wins the rack, and
(2) a success-rate score for every candidate shot, so it can recommend the shot that is
most likely to work *for a player of that skill*.

## Setup
```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```
Note: pooltool 0.3.3 uses `np.math`, which numpy 2 removed; `src/sim/__init__.py` restores the alias.

## Pipeline
| Step | Command | Output |
|---|---|---|
| Smoke data (~400 states) | `python -m src.data.generate --config configs/data_smoke.yaml` | `data/smoke/` |
| Main data | `python -m src.data.generate --config configs/data_main.yaml` | `data/main/states.npz` |
| Label noise (M vs 64) | `python -m src.data.label_noise --data data/main/states.npz` | `results/label_noise/` |
| Train all models + ablations | `bash scripts/train_all.sh data/main/states.npz` | `results/<run>/` |
| Shot evaluation (fresh rollouts) | `python -m src.evaluate --data data/main/states.npz --runs results/set_transformer ... --out results/shot_eval_main` | `results/shot_eval_main/` |
| Figures | `python -m src.figures --data data/main/states.npz --shot_eval results/shot_eval_main --win_runs set_transformer cnn mlp deepsets heuristic --ablation_prefix size_set_transformer --ablation_extra set_transformer` | `figures/` |

## Code map
- `src/sim/shots.py`: skill noise, ghost-ball candidate generation, execution, success check, Monte-Carlo teacher.
- `src/sim/game.py`: full 8-ball racks with pooltool's ruleset; state encoding.
- `src/data/generate.py`: parallel, resumable data generation.
- `src/data/dataset.py`, `src/data/render.py`: tensors, game-level splits, GPU table renderer.
- `src/models/nets.py`: MLP, DeepSets, Set Transformer (main), ResNet-18 CNN, all with a win head and a shot head.
- `src/models/heuristic.py`: geometric heuristic + logistic regression on ball counts.
- `src/train.py`, `src/evaluate.py`, `src/figures.py`.

All design decisions and every run (including failures) are logged in `EXPERIMENTS.md`.
