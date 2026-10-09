#!/bin/bash
# Round 3: same as round 2 but (a) gen_preflop.py's imap deadlock bug fixed, (b) postflop rollout labels
# use pot_odds:range=1,value=1,bluff=1 as hero's continuation policy instead of plain pot_odds -- that
# bot beats the old one +77 +/- 13 bb/100 (2026-09-30), so its follow-up play should give better value-bet
# labels. Sequential stages, --procs 6 (not 22): 22 caused OOM + a 7h imap deadlock overnight.
set -e
cd "$(dirname "$0")"
echo "round3 start $(date)"

mem_guard() {
  avail=$(free -m | awk '/Mem:/{print $7}')
  echo "  [mem] available ${avail}MB $(date)"
  [ "$avail" -lt 1200 ] && { echo "  [mem] CRITICAL <1.2GB, aborting"; exit 1; }
}

rm -f data/pre_train.flat.jsonl data/roll_train.flat.jsonl   # stale/partial from the killed run
uv run python gen_preflop.py data/pre_train --spots 8000 --seed 1 --procs 6 > data/pre_train.log 2>&1
mem_guard
n=$(wc -l < data/pre_train.flat.jsonl)
echo "pre_train done $(date): $n rows"
[ "$n" -lt 6000 ] && { echo "ABORT: pre_train only $n/8000 rows, something's still wrong"; exit 1; }

uv run python gen_data.py data/roll_train --sessions 2000 --seed 1 --procs 6 --minutes 180 \
  --cont "pot_odds:range=1,value=1,bluff=1" > data/roll_train.log 2>&1
mem_guard
n=$(wc -l < data/roll_train.flat.jsonl)
echo "roll_train done $(date): $n rows"
[ "$n" -lt 20000 ] && { echo "ABORT: roll_train only $n rows, too few to train on"; exit 1; }

uv run python gen_data.py x --stats data/roll_train.flat.jsonl > data/roll_train.stats 2>&1 || true

# also regenerate roll_test/pre_test with the same --cont, so val matches train distribution
uv run python gen_data.py data/roll_test --sessions 60 --seed 2 --procs 6 \
  --cont "pot_odds:range=1,value=1,bluff=1" > data/roll_test.log 2>&1
mem_guard
uv run python gen_preflop.py data/pre_test --spots 500 --seed 2 --procs 6 > data/pre_test.log 2>&1
mem_guard

# ponytail: 20k PokerBench rows kept so postflop solver knowledge isn't forgotten; tune mix ratio later if needed.
uv run python finetune.py --base checkpoints/laya-poker-flat --out checkpoints/laya-poker-roll \
  --train data/roll_train.flat.jsonl,data/pre_train.flat.jsonl,data/train.flat.jsonl:20000 \
  --val data/roll_test.flat.jsonl,data/pre_test.flat.jsonl,data/test.flat.jsonl --val-n 1000 \
  --epochs 1.5 --lr 1e-5 --head-lr 5e-5 > checkpoints/roll.log 2>&1
mem_guard
echo "train done $(date)"

echo "ROUND3_PIPELINE_COMPLETE $(date)"
