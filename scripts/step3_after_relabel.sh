#!/bin/bash
# Wait for the all-skill labels, then train (a) single-skill labels + skill input, (b) single-skill labels,
# no skill input, (c) all-skill labels + three-output head, seeds 0-2, then the 64-rollout shot evaluation
# with cross-skill execution. Usage: GPUS="0 1 4 5 6 7" bash scripts/step3_after_relabel.sh
DATA=data/main/states.npz
L3=data/main/relabel3/labels3.npz
GPUS=(${GPUS:-0 1 4 5 6 7})
source .venv/bin/activate
mkdir -p results/logs
until [ -f $L3 ]; do sleep 60; done
j=0
run() { tag=$1; shift; CUDA_VISIBLE_DEVICES=${GPUS[$((j % ${#GPUS[@]}))]} python -m src.train --data $DATA --labels3 $L3 \
          --config configs/train_set_transformer.yaml --tag $tag "$@" > results/logs/$tag.log 2>&1 & j=$((j + 1)); }
for s in 0 1 2; do
  run shot_a_skill_s$s   --seed $s
  run shot_b_noskill_s$s --seed $s --use_skill 0
  run shot_c_3head_s$s   --seed $s --shot_outputs 3
done
wait
python -m src.evaluate --data $DATA --n_states 300 --seed 0 \
  --runs results/shot_a_skill_s0 results/shot_b_noskill_s0 results/shot_c_3head_s0 \
         results/shot_a_skill_s1 results/shot_b_noskill_s1 results/shot_c_3head_s1 \
         results/shot_a_skill_s2 results/shot_b_noskill_s2 results/shot_c_3head_s2 \
  --out results/shot_eval_abc > results/logs/shot_eval_abc.log 2>&1
