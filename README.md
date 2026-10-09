# poker_bot: test Laya as a heads-up NLHE decision model

Heads-up no-limit hold'em, 100bb stacks. Poker math (equity, pot odds, SPR, hand class) is computed
in code and fed to Laya as a JSON state plus one `choice` question over the legal actions.
Laya's per-action probabilities are sampled (or argmax with `--greedy`).

```
uv sync        # if downloads fail: UV_CONCURRENT_DOWNLOADS=1 UV_HTTP_TIMEOUT=600 uv sync
python test_poker.py                                       # no torch needed
uv run python sim.py --a laya --b pot_odds --deals 500     # baselines: pot_odds | calling_station | random
uv run python sim.py --a laya --b calling_station --deals 20 -v --log decisions.jsonl
uv run python sim.py --a laya --laya-model /path/to/finetuned --b pot_odds   # test a fine-tuned checkpoint
```

- **Duplicate format:** each deal is played twice with seats swapped, so card luck mostly cancels.
  Win rate is A's bb/100 with a 95% CI. If the CI crosses 0, the result isn't significant.
- **All-in EV:** a called all-in before the river pays each player their equity share of the pot instead of
  dealing the board out. Same expected result, less noise (since 2026-09-28; older results used the real runout).
- `--log` writes one JSON line per Laya decision (state, options, probabilities, choice, latency).
- Equity is vs a *random* hand, not a range (`# ponytail` in `poker.py`), unless the agent has `range=1`.
- `laya:preflop=1` plays the pot-odds rule preflop. Use it with `checkpoints/laya-poker-flat`, which was trained on postflop spots only.
- The progress line shows A's running win rate, so a stopped run still tells you something.

## Results: zero-shot `convaiinnovations/laya`, RTX 4060 laptop, seed 1, 500 deals (1000 hands)

| Match | Laya win rate (bb/100) |
|---|---|
| vs pot_odds (sampled) | -1469 +/- 227 |
| vs calling_station (sampled) | -2052 +/- 199 |
| vs calling_station (`--greedy`) | +49 +/- 233 (not significant) |

Latency p50 about 23 ms, p95 about 26 ms. State is about 180 tokens, under the ~250 budget.
Zero-shot Laya loses to a trivial math baseline, as the model card predicts. With `--greedy` it played
"pot" on every postflop decision, whatever the state, so it is mostly following a label prior and not
reading the numbers. Use these numbers as the baseline to beat with a fine-tuned checkpoint.

## Benchmarks

All sims: `sim.py`, seed 1, duplicate deals, win rate is A's bb/100 with a 95% CI. Raw output is in
`results.txt`, `sim_roundrobin.txt`, `sim_finetuned.txt`, `valuebot_eval.log` and `sim_logs/*.txt`.
Results from before the all-in engine fix (2026-09-28) are marked stale and should not be compared with later ones.

Agent flags: `range=1` equity vs a range (not a random hand), `guard=1` blocks clearly bad moves,
`rich=1` richer state, `tree=1` tree-structured questions, `preflop=1` pot-odds rule preflop,
`alpha` guard strength, `model=` fine-tuned checkpoint, `value=1,bluff=1` value-bet and bluff rules for `pot_odds`.
FT = `checkpoints/laya-poker-flat`.

### Fine-tune: `laya-poker-flat`

PokerBench postflop solver labels, 177k flat rows (`convert_pokerbench.py`, `finetune.py`).
Stopped at step 6750 of 11085 (1.96 h on an RTX 4060 laptop). Training log: `checkpoints/flat.log`.

| Checkpoint | Solver-match on val |
|---|---|
| zero-shot | 26% |
| 500 steps | 57.6% |
| best | 80.6% |

The weights (808 MB) are not in this repo.

### Quiz: regret vs the best action (200 situations, `quiz_results.txt`)

| Agent | Regret bb (95% CI) | Near-best | Blunder |
|---|---|---|---|
| random (floor) | 9.13 +/- 1.83 | 32.4% | 42.9% |
| calling_station | 7.16 +/- 2.17 | 20.5% | 46.5% |
| laya | 7.45 +/- 2.09 | 34.2% | 38.4% |
| laya:rich=1 | 7.61 +/- 2.07 | 32.4% | 39.2% |
| laya:tree=1 | 8.05 +/- 2.39 | 28.9% | 38.3% |
| laya:tree=1,rich=1 | 8.16 +/- 2.24 | 27.7% | 40.1% |
| laya:tree=1,rich=1,range=1 | 8.28 +/- 2.21 | 28.5% | 41.0% |
| laya:tree=1,rich=1,range=1,guard=1 | 6.18 +/- 2.24 | 38.0% | 33.4% |
| laya:range=1,guard=1 | 6.09 +/- 2.21 | 44.0% | 31.9% |
| pot_odds | 4.86 +/- 1.75 | 41.5% | 33.0% |
| pot_odds:range=1 | 5.69 +/- 2.18 | 45.0% | 30.5% |

Zero-shot Laya does not read the numbers: its fold/play answer has a -0.06 correlation with equity minus pot odds.
Tree and rich prompts are no better than flat. The guard is what helps.

### Sims vs `pot_odds` (2000 hands each)

| A | Win rate (bb/100) |
|---|---|
| laya:range=1,guard=1 (zero-shot) | +91.8 +/- 68.6 |
| FT, rich=1 (no guard) | -52.2 +/- 69.4 (not significant) |
| FT, rich=1,guard=1 (no range) | -152 +/- 77 |
| FT, rich=1,range=1,guard=1 | +62 +/- 55 |
| FT, rich=1,range=1,guard=1,preflop=1 | +116 +/- 50 |
| pot_odds:range=1 (no model) | +244 +/- 65 |
| pot_odds:range=1,value=1,bluff=1 (no model, 7992 hands) | +255.4 +/- 33.6 |

The FT checkpoint was trained on postflop only and folded 452 small blinds to a 0.5 bb complete; `preflop=1` fixes that.

### Sims vs `pot_odds:range=1`

| A | Hands | Win rate (bb/100) |
|---|---|---|
| FT, rich=1,range=1,guard=1,preflop=1 | 2000 | -24 +/- 22 |
| same, alpha=0.5 | 2000 | -10 +/- 18 |
| same, `--greedy` | 4000 | -38.8 +/- 18.6 |
| FT, rich=1,range=1,preflop=1 (no guard), `--greedy` | 4000 | -15.2 +/- 20.5 |
| pot_odds:range=1,value=1,bluff=1 | 15996 | +77.1 +/- 12.9 |

### Gauntlet (2000 hands each)

| Opponent | FT best (preflop=1, alpha=0.5) | pot_odds:range=1 |
|---|---|---|
| calling_station | +293 +/- 50 | +504 +/- 53 |
| random | +646 +/- 108 | +574 +/- 103 |
| laya:range=1,guard=1 | +62 +/- 32 | +79 +/- 45 |

`pot_odds:range=1,value=1,bluff=1` vs calling_station (7992 hands): +2289.1 +/- 75.1.

### Head to head

FT best (`rich=1,range=1,guard=1,preflop=1`) vs zero-shot best (`laya:range=1,guard=1`), 4000 hands: **+55.2 +/- 31.6**.

### Stale (before the all-in engine fix, `sim_roundrobin.txt`)

| Match | Win rate (bb/100) |
|---|---|
| laya:tree=1,rich=1,range=1,guard=1 vs pot_odds | +155.9 +/- 63.3 |
| laya:range=1,guard=1 vs pot_odds | +243.2 +/- 69.4 |
| tree/rich/range/guard vs range/guard | -30.7 +/- 64.8 |

### Takeaways

- Range-equity math with no model (`pot_odds:range=1`) beats every Laya config; adding value-bet and bluff rules makes it stronger still.
- The fine-tune beats zero-shot Laya, but only ties the math bot. It is too passive: it checks about 60% of postflop decisions.
- Greedy decoding does not help, and the guard is at best neutral against the range bot.

## Status

Round 3 (`run_round3.sh`) is not finished: rollout labels from the value/bluff bot plus preflop labels, then a
warm-start fine-tune from `laya-poker-flat`. Plan is in `FINETUNE_PLAN.md`.
