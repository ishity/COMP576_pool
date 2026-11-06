#!/bin/bash
# Re-run the main models and the size ablation with extra seeds (seed 0 = runs from train_all.sh).
# Usage: bash scripts/train_seeds.sh data/main/states.npz "1 2"
DATA=${1:-data/main/states.npz}
SEEDS=${2:-"1 2"}
source .venv/bin/activate
mkdir -p results/logs
gpu=0
run() { tag=$1; shift; CUDA_VISIBLE_DEVICES=$gpu python -m src.train --data $DATA --tag $tag "$@" > results/logs/$tag.log 2>&1 & gpu=$(( (gpu + 1) % 8 )); }
for s in $SEEDS; do
  run set_transformer_s$s          --config configs/train_set_transformer.yaml --seed $s
  run set_transformer_noskill_s$s  --config configs/train_set_transformer.yaml --use_skill 0 --seed $s
  run cnn_s$s                      --config configs/train_cnn.yaml --seed $s
  run mlp_s$s                      --config configs/train_mlp.yaml --seed $s
  run deepsets_s$s                 --config configs/train_deepsets.yaml --seed $s
  run size_set_transformer_1k_s$s  --config configs/train_set_transformer.yaml --max_train_states 1000 --seed $s
  run size_set_transformer_3k_s$s  --config configs/train_set_transformer.yaml --max_train_states 3000 --seed $s
  run size_set_transformer_10k_s$s --config configs/train_set_transformer.yaml --max_train_states 10000 --seed $s
done
wait
