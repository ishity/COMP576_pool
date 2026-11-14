#!/bin/bash
# Final round: (a) on M=32 labels, (a) + listwise ranking loss on M=16 and on M=32 labels, seeds 0-2,
# then the 64-rollout shot evaluation (same 300 states) including the existing (a) runs for paired comparison.
DATA=data/main/states.npz
L3=data/main/relabel3/labels3.npz
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
source .venv/bin/activate
mkdir -p results/logs
j=0
run() { tag=$1; shift; CUDA_VISIBLE_DEVICES=${GPUS[$((j % ${#GPUS[@]}))]} python -m src.train --data $DATA --labels3 $L3 \
          --config configs/train_set_transformer.yaml --tag $tag "$@" > results/logs/$tag.log 2>&1 & j=$((j + 1)); }
for s in 0 1 2; do
  run shot_a_m32_s$s      --seed $s --shot_label m32
  run shot_rank_m16_s$s   --seed $s --rank_weight 1 --rank_tau 0.05
  run shot_rank_m32_s$s   --seed $s --rank_weight 1 --rank_tau 0.05 --shot_label m32
done
wait
RUNS=""
for g in shot_a_skill shot_a_m32 shot_rank_m16 shot_rank_m32; do for s in 0 1 2; do RUNS="$RUNS results/${g}_s$s"; done; done
python -m src.evaluate --data $DATA --n_states 300 --seed 0 --runs $RUNS --no_cross \
  --out results/shot_eval_final > results/logs/shot_eval_final.log 2>&1
