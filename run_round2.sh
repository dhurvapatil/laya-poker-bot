#!/bin/bash
# Round 2: rollout labels (postflop + preflop), then warm-start fine-tune from the PokerBench checkpoint.
set -e
cd "$(dirname "$0")"
echo "start $(date)"
uv run python gen_data.py data/roll_test --sessions 60 --seed 2 --procs 22 > data/roll_test.log 2>&1
uv run python gen_preflop.py data/pre_test --spots 500 --seed 2 > data/pre_test.log 2>&1
uv run python gen_preflop.py data/pre_train --spots 8000 --seed 1 > data/pre_train.log 2>&1
uv run python gen_data.py data/roll_train --sessions 2000 --seed 1 --procs 22 --minutes 100 > data/roll_train.log 2>&1
echo "data done $(date)"; wc -l data/roll_*.flat.jsonl data/pre_*.flat.jsonl
uv run python gen_data.py x --stats data/roll_train.flat.jsonl > data/roll_train.stats 2>&1 || true
# ponytail: 20k PokerBench rows kept so postflop solver knowledge isn't forgotten; tune the mix after eval.
uv run python finetune.py --base checkpoints/laya-poker-flat --out checkpoints/laya-poker-roll \
  --train data/roll_train.flat.jsonl,data/pre_train.flat.jsonl,data/train.flat.jsonl:20000 \
  --val data/roll_test.flat.jsonl,data/pre_test.flat.jsonl,data/test.flat.jsonl --val-n 1000 \
  --epochs 1.5 --lr 1e-5 --head-lr 5e-5 > checkpoints/roll.log 2>&1
echo "train done $(date)"
