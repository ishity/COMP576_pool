#!/bin/bash
# 1) reproduce shot_eval_main (same args + full teacher vectors + cross-skill), 2) relabel all states at 3 skills.
source .venv/bin/activate
python -m src.evaluate --data data/main/states.npz --n_states 300 --seed 0 \
  --runs results/set_transformer results/set_transformer_noskill results/cnn results/mlp results/deepsets \
  --out results/shot_eval_main_v2 > results/logs/shot_eval_main_v2.log 2>&1
python -m src.data.relabel --data data/main/states.npz --out data/main/relabel3 > results/logs/relabel3.log 2>&1
python -m src.data.relabel --data data/main/states.npz --out data/main/relabel3 --merge >> results/logs/relabel3.log 2>&1
