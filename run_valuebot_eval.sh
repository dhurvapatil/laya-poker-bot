#!/bin/bash
# Evaluate pot_odds value=1,bluff=1 vs pot_odds:range=1 and calling_station, seed 1 then seed 2 confirm.
set -e
cd "$(dirname "$0")"
mkdir -p sim_logs
echo "start $(date)"

run() {
  name=$1; a=$2; b=$3; deals=$4; seed=$5
  n=6   # 6-way split, cheap CPU-only agents, leaves room for the GPU pipeline's CPU needs
  per=$((deals / n))
  pids=()
  for i in $(seq 0 $((n-1))); do
    uv run python sim.py --a "$a" --b "$b" --deals $per --start $((i*per)) --seed $seed \
      > sim_logs/${name}_${i}.txt 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  python3 - "$name" "$n" <<'PYEOF'
import sys, re, math
name, n = sys.argv[1], int(sys.argv[2])
means, cis, deals = [], [], 0
for i in range(n):
    t = open(f"sim_logs/{name}_{i}.txt").read()
    m = re.search(r"win rate: ([+-]?[\d.]+) bb/100 \(95% CI \+/- ([\d.]+)\)", t)
    d = re.search(r"(\d+) deals x", t)
    means.append(float(m.group(1))); cis.append(float(m.group(2))); deals += int(d.group(1))
mean = sum(means) / n
ci = math.sqrt(sum(c**2 for c in cis)) / n
print(f"{name}: {deals} deals ({2*deals} hands) -> {mean:+.1f} +/- {ci:.1f} bb/100")
PYEOF
}

run vb_vs_range "pot_odds:range=1,value=1,bluff=1" "pot_odds:range=1" 8000 1
run vb_vs_cs "pot_odds:range=1,value=1,bluff=1" "calling_station" 4000 1
run vb_vs_po "pot_odds:range=1,value=1,bluff=1" "pot_odds" 4000 1
run vb_vs_range_seed2 "pot_odds:range=1,value=1,bluff=1" "pot_odds:range=1" 8000 2

echo "VALUEBOT_EVAL_COMPLETE $(date)"
