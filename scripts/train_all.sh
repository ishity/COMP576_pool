#!/bin/bash
# Train main models + ablations on the main dataset, one GPU per job (8x A40).
# Usage: bash scripts/train_all.sh data/main/states.npz
set -e
DATA=${1:-data/main/states.npz}
source .venv/bin/activate
mkdir -p results/logs
run() { gpu=$1; shift; tag=$1; shift; CUDA_VISIBLE_DEVICES=$gpu python -m src.train --data $DATA --tag $tag "$@" > results/logs/$tag.log 2>&1 & }
run 0 set_transformer         --config configs/train_set_transformer.yaml
run 1 set_transformer_noskill --config configs/train_set_transformer.yaml --use_skill 0
run 2 cnn                     --config configs/train_cnn.yaml
run 3 mlp                     --config configs/train_mlp.yaml
run 4 deepsets                --config configs/train_deepsets.yaml
run 5 size_set_transformer_3k  --config configs/train_set_transformer.yaml --max_train_states 3000
run 6 size_set_transformer_10k --config configs/train_set_transformer.yaml --max_train_states 10000
run 7 size_set_transformer_1k  --config configs/train_set_transformer.yaml --max_train_states 1000
wait
python -m src.models.heuristic --data $DATA --tag heuristic > results/logs/heuristic.log 2>&1
