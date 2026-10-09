#!/bin/bash
# Resume round 2 after a swap-thrashing kill: roll_test/pre_test already done.
# Runs pre_train + roll_train at --procs 8 (was 22, caused OOM-swap thrashing), then trains.
set -e
cd "$(dirname "$0")"
echo "resume start $(date)"

mem_check() {
  avail=$(free -m | awk '/Mem:/{print $7}')
  echo "  [mem check] available ${avail}MB $(date)"
  if [ "$avail" -lt 1000 ]; then
    echo "  [mem check] CRITICAL: <1GB available, aborting pipeline" 
    exit 1
  fi
}

uv run python gen_preflop.py data/pre_train --spots 8000 --seed 1 --procs 8 > data/pre_train.log 2>&1
mem_check
echo "pre_train done $(date): $(wc -l < data/pre_train.flat.jsonl) rows"

uv run python gen_data.py data/roll_train --sessions 2000 --seed 1 --procs 8 --minutes 180 > data/roll_train.log 2>&1
mem_check
echo "roll_train done $(date): $(wc -l < data/roll_train.flat.jsonl) rows"

uv run python gen_data.py x --stats data/roll_train.flat.jsonl > data/roll_train.stats 2>&1 || true
mem_check

# ponytail: 20k PokerBench rows kept so postflop solver knowledge isn't forgotten; tune mix ratio later if needed.
uv run python finetune.py --base checkpoints/laya-poker-flat --out checkpoints/laya-poker-roll \
  --train data/roll_train.flat.jsonl,data/pre_train.flat.jsonl,data/train.flat.jsonl:20000 \
  --val data/roll_test.flat.jsonl,data/pre_test.flat.jsonl,data/test.flat.jsonl --val-n 1000 \
  --epochs 1.5 --lr 1e-5 --head-lr 5e-5 > checkpoints/roll.log 2>&1
echo "train done $(date)"
mem_check

echo "PIPELINE_COMPLETE $(date)"
