# Plan: Laya poker test simulator

> **Status: v1 built and verified** (poker.py, agents.py, sim.py, test_poker.py, README.md). Results are in README.md.

A handoff plan: a fresh session should be able to build this without the earlier conversation.
Work through the steps in order. After each step, run the check listed for it.

## Goal

A heads-up no-limit hold'em (HU NLHE) simulator for testing **Laya** (`convaiinnovations/laya`,
a 421M ModernBERT "System 1" decision model) as the poker decision-maker:

1. Poker math (pot odds, equity, SPR, hand class) is computed in code.
2. That state goes to Laya as a JSON `state` plus one typed `choice` question over the legal actions.
3. Laya returns a probability per action. The agent **samples** from that distribution; `--greedy` takes the argmax instead.
4. Laya plays matches against baseline bots. We report its win rate with a confidence interval, its decision latency, and what it did.

Expectation: the **zero-shot base checkpoint is near random** (the model card says so). The simulator is
the harness for measuring that now, and for testing fine-tuned checkpoints later (`--laya-model /path`).
Do not tune anything to make zero-shot Laya look good.

## Constraints (the user's rules)

- Keep it minimal: fewest files, stdlib first, no new dependencies beyond `laya`. No frameworks, no pytest
  requirement, no config files, no class hierarchies with one implementation.
- Mark deliberate shortcuts with a `# ponytail:` comment that names the limit and the upgrade path.
- Leave one runnable check: `test_poker.py`, assert-based, runnable with plain `python test_poker.py`.
- The project is **not** a git repo. Don't `git init` unless asked.

## Environment (verified)

- Dir: `/home/dhruva/workspace/poker_bot`. It has `pyproject.toml` (deps: `laya>=0.3.20`), `uv.lock` and `.venv`.
- Python 3.14.7. PyPI has torch 2.14.0 cp314 wheels, and the default Linux wheel is CUDA-enabled.
- GPU: RTX 4060 Laptop, 8 GB, driver 610 (CUDA 13).
- `uv sync` failed **twice** with a TLS error ("cannot decrypt peer's message") while downloading the large
  torch/CUDA wheels, so the network is flaky on big downloads. **Step 0:** retry with fewer parallel downloads:
  `UV_CONCURRENT_DOWNLOADS=1 UV_HTTP_TIMEOUT=600 uv sync` (repeat it; uv caches finished wheels). If it keeps
  failing, stop and ask the user to run it themselves. Then confirm with
  `uv run python -c "import torch,laya;print(torch.__version__, torch.cuda.is_available(), laya.__version__)"`.
  It should print `True` and `0.3.20`.
- The model downloads about 800 MB into the HF cache on the first `laya.load`.

## Laya API (verified by reading the laya 0.3.20 source; don't guess beyond this)

```python
import laya                                  # cheap; torch is imported lazily
agent = laya.load("convaiinnovations/laya", device="cuda")   # or a local fine-tuned dir
res = agent.predict(state_dict, questions)   # alias of system_one(); one state, one forward pass
a = res["answers"]["action"]
a["choice"]              # argmax label (str)
a["probabilities"]       # {label: p}, rounded to 4dp, temperature applied
a["answer_confidence"]   # max(p): the calibrated confidence
a["confidence"]          # normalized entropy: NOT calibrated, don't gate on it
a["action"]["act_probability"]  # broken upstream (always ~1.0); ignore it
agent.predict_batch([s1, s2, ...], questions)  # throughput path; not needed for v1
```

- A `state` dict is serialized with `json.dumps(state, ensure_ascii=False)` (default separators).
- Questions look like `{"action": {"type": "choice", "instructions": "...", "criteria": {label: description}}}`.
  Each option is rendered as `"label: description"`, and dict order = option order.
- Token budget for the English checkpoint: `max_len` 512 and `head_max_len` 192. The head (instruction + options)
  gets up to 192 tokens, each option description is truncated at 48 tokens, and the state gets what's left
  (about 300 tokens). **Keep the state compact** (well under ~250 tokens), or it is silently cut off at the end.
- Temperatures are bucketed by option count (`choice:2`, `choice:3-5`, `choice:6-10`). Keep it to ≤5 options.

## Files to create

```
poker.py       cards, 7-card evaluator, Monte Carlo equity, HU engine, features()
agents.py      RandomAgent, CallingStation, PotOddsAgent (math baseline), LayaAgent
sim.py         CLI: duplicate match runner, stats, JSONL decision log
test_poker.py  assert-based checks (no torch import)
README.md      ~15 lines: setup, and 3 example commands
```

### poker.py

**Cards:** use ints 0–51, with `rank = c // 4` (0 = '2' … 12 = 'A') and `suit = c % 4`. Add `card_str(c)` giving
e.g. `"As"` (ranks `"23456789TJQKA"`, suits `"cdhs"`), and `parse(s)` doing the reverse. `parse` is used by the tests.

**Evaluator:** `rank7(cards) -> tuple`. Higher tuple = better hand. It must work for 5–7 cards.
- Categories: 8 straight flush, 7 quads, 6 full house, 5 flush, 4 straight, 3 trips, 2 two pair, 1 pair, 0 high card.
- Return `(category, *tiebreak_ranks)`, e.g. `(1, pair_rank, k1, k2, k3)`.
- Flush: take the suit with ≥5 cards and check for a straight flush within that suit before the plain flush.
  The flush tiebreak is the top 5 ranks of that suit.
- Straight: use the unique ranks. The wheel A-2-3-4-5 has high card 3 (the '5'). Helper `straight_high(ranks_set)`.
- Full house from two trips: the higher trips is the set, the lower gives the pair. Also handle trips plus two pairs.
- Two pair from three pairs: take the top two pairs, and the kicker is the best remaining card, which can
  come from the third pair.
- Add `CATEGORY = ["high card", "pair", "two pair", "trips", "straight", "flush", "full house", "quads", "straight flush"]`.

**Equity:** `equity(hole, board, n=800) -> float`, the win probability plus half the tie probability, against **one uniformly random
opponent hand**, rolling out the rest of the board.
- Decorate with `@functools.lru_cache(maxsize=100_000)`. Args are tuples, and the caller passes `tuple(sorted(...))` for hole and `tuple(board)`.
- Use `rng = random.Random(hash((hole, board)))`. Int-tuple hashes are deterministic, so equity is reproducible.
- `# ponytail: equity vs a random hand, not a range; range-weighted equity is the upgrade (see Future work).`

**Engine** (heads-up only, integer chips): SB=50, BB=100, stacks reset to 10_000 (100bb) every hand.
- Seat 0 is the **button = small blind**. It acts first preflop, and the BB (seat 1) acts first on flop, turn and river.
- `play_hand(agents, deck, log=None) -> (delta_seat0, delta_seat1)`. The deltas are net chips and must sum to 0.
  - Deal: seat0 gets `deck[0:2]`, seat1 gets `deck[2:4]`, the board is `deck[4:9]` (flop 4:7, turn 7, river 8).
  - Per street, track `street_bet[2]`, `stack[2]`, `pot` (chips from earlier streets), `last_raise` (for min-raise;
    starts at BB each street, and preflop too), and `history` (a short list of strings like `"BTN raise 3.0"`, `"BB call"`).
  - A betting round ends when both players have acted since the last raise and the street bets are equal, or when
    someone folds, or when a player is all-in and the bets are matched. Preflop: the BB gets its option if the SB just calls.
  - If either player is all-in (and bets are matched), run out the board and show down.
  - Showdown uses `rank7(hole + board)`; the higher hand wins, and a tie splits the pot (put the odd chip on seat 1).
- `legal_options(...) -> dict[label, (kind, raise_to)]`. Every agent picks a **label** from this dict:
  - Facing a bet (`to_call > 0`): `fold`, `call`, plus raises.
  - Not facing one: `check`, plus bets.
  - Raise/bet sizes: `half_pot` and `pot`, where `raise_to = opp_street_bet + frac * (pot_total + to_call)`
    (`pot_total` includes the current street bets), plus `allin`.
  - Clamp `raise_to` to `[opp_street_bet + max(last_raise, BB), cap]`, where
    `cap = min(my_street_bet + my_stack, opp_street_bet + opp_stack)` (never bet more than the opponent can call).
    If `cap < min_raise_to`, the only raise is `allin` = `cap`. Drop sizes that clamp to the same amount
    (keep the smaller label), and drop all raises if `cap <= opp_street_bet` or the opponent is all-in.
  - Label every size that equals `cap` as `allin`.
  - Preflop sanity check: SB facing the BB has `half_pot` → raise to 200 and `pot` → raise to 300.
- The observation passed to `agent.act(obs, options) -> label` is a plain dict: `hole`, `board` (visible cards only), `street`,
  `position` ("BTN"/"BB"), `pot_total`, `to_call`, `my_stack`, `opp_stack`, `history`.
- `features(obs) -> dict`: the "computed externally" math shared by the PotOdds and Laya agents:
  `equity`, `pot_odds = to_call / (pot_total + to_call)` (0 if nothing to call), `spr = min(stacks)/pot_total`,
  `hand = CATEGORY[rank7(...)[0]]` (only when ≥3 board cards; preflop, describe it as e.g. `"AKs"` / `"72o"` / `"pair of 9s"`).

### agents.py

Every agent has `name` and `act(obs, options) -> label`. RNGs are seeded from the CLI `--seed`.

- `RandomAgent`: uniform over the labels.
- `CallingStation`: `call` if present, else `check`.
- `PotOddsAgent`: **the baseline Laya must beat.** It uses the same `features()`:
  - Facing a bet: if equity > 0.75 → the largest raise, else if equity ≥ pot_odds + 0.05 → call, else fold.
  - Not facing a bet: if equity > 0.65 → `pot` (or the nearest available), else if equity > 0.5 → `half_pot`, else check.
  - Keep it this dumb, and don't tune it.
- `LayaAgent(model, device, greedy, rng, log_fn)`:
  - Lazy `import laya` inside `__init__`, so baseline matches and tests never import torch. `device` defaults to
    `"cuda" if torch.cuda.is_available() else "cpu"`.
  - `state` = compact dict with all numbers in bb, rounded to 1 decimal (probabilities to 2 decimals):
    `street`, `position`, `hole` ("Ah Kd"), `board`, `hand`, `pot_bb`, `to_call_bb`, `stack_bb`, `opp_stack_bb`,
    `spr`, `pot_odds`, `equity_vs_random`, `equity_minus_pot_odds` (as a string with a verbal hint:
    `"+0.16 (profitable call)"` / `"-0.10 (unprofitable call)"` / `"n/a (no bet)"`), and `history` (last ~6 entries, joined with "; ").
    Assert or measure that `len(tok(json.dumps(state)))` stays under ~250 tokens. `agent.tok` is the tokenizer.
  - `questions = {"action": {"type": "choice", "instructions": "You are hero in heads-up no-limit hold'em. Which action maximizes hero's expected winnings?", "criteria": {...}}}`.
    The criteria are the legal labels with short descriptions: `"fold": "give up the hand"`, `"call": "call 2.0 bb"`,
    `"raise_pot": "raise to 9.0 bb"`, `"allin": "all-in for 97.5 bb"`, etc. Same keys as `options`.
  - Sample the label from `probabilities`, or take the argmax if `greedy`. Guard: renormalize the probabilities, and if a label is missing
    fall back to `a["choice"]`.
  - Time each `predict` call with `time.perf_counter()`. Record `latency_ms` and `answer_confidence`, and call
    `log_fn({...state, "options": criteria, "probs": ..., "chosen": ..., "latency_ms": ...})`.

### sim.py

The CLI (argparse):

```
uv run python sim.py --a laya --b pot_odds --deals 200 --seed 1 [--greedy] [--laya-model ID_OR_PATH]
                     [--device cuda|cpu] [--log decisions.jsonl] [-v]
```

- `--a/--b` accept `laya | pot_odds | calling_station | random`.
- **Duplicate format:** deal `i` uses `deck = random.Random(seed*1_000_003 + i).sample(range(52), 52)`.
  - Game 1: A in seat 0, B in seat 1.
  - Game 2: same deck, **B in seat 0, A in seat 1**.
  - `pair_net_A = A's delta in game 1 + A's delta in game 2`. This cancels most card luck.
- Report the following. The statistics use the stdlib `statistics` module (`mean`, `stdev`).
  - A's win rate in bb/100 = `mean(pair_net)/2/BB*100`, ± 95% CI = `1.96*stdev(pair_net)/sqrt(n)/2/BB*100`.
  - Hands played, and the showdown vs non-showdown split (optional; skip it if it adds complexity).
  - For each Laya agent in the match: decisions, latency p50/p95 (`statistics.quantiles`), mean `answer_confidence`,
    and action frequencies by street.
- `-v` prints every hand: the cards, each action, and for Laya the top probabilities, e.g. `laya: call (call .41, fold .33, raise_pot .26)`.
- `--log` writes one JSON line per Laya decision. It is useful later as fine-tuning data or for error analysis.
- Print a one-line reminder with the results: `win rate CI crosses 0 → not significant, run more deals`.

### test_poker.py

These are plain asserts in `test_*` functions, with `if __name__ == "__main__":` calling them all and printing `ok`.
It must not import laya or torch.

1. Evaluator ordering:
   - royal flush > quads > full house > flush > straight > trips > two pair > pair > high card, using fixed 7-card hands built with `parse`.
   - Wheel `A2345` < `23456`.
   - Pair kickers compare correctly.
   - Three pairs pick the best kicker.
   - Two trips make a full house.
   - A flush beats a straight that uses different cards.
2. Equity sanity: `AA` preflop ≈ 0.85 (±0.03), `72o` ≈ 0.35 (±0.04), and a made nut hand on the river = 1.0.
3. Engine:
   - Over 300 random deals with `RandomAgent` vs `RandomAgent`, every hand's deltas sum to 0, and |delta| ≤ 10_000.
   - Two `CallingStation`s always reach showdown.
   - In the preflop sizes check, the SB `half_pot` is 200 and `pot` is 300.
4. Duplicate symmetry: with the same agent type on both sides and `--greedy`-style determinism (use `CallingStation`),
   the pair net is exactly 0 for every deal.

## Build order and checks

0. Run `uv sync` and the import check above.
1. Write `poker.py` and the evaluator and equity tests. Check: `uv run python test_poker.py`.
2. Add the engine and the engine tests. Check: `uv run python test_poker.py`.
3. Write `agents.py` (baselines only) and `sim.py`. Check:
   `uv run python sim.py --a pot_odds --b calling_station --deals 300` → PotOdds should win clearly,
   and `--a random --b random` → a CI containing 0.
4. Add `LayaAgent`. Check:
   `uv run python sim.py --a laya --b calling_station --deals 20 -v --log decisions.jsonl`.
   Confirm the model loads on CUDA, `latency_ms` is roughly 20–60 ms, and the probabilities vary between spots.
   Check the token-length assertion.
5. Run the real test and record the numbers in the README:
   `--a laya --b pot_odds --deals 500` and `--a laya --b calling_station --deals 500`.
   Report the results honestly. Zero-shot will likely lose to pot_odds.
6. Write `README.md`: setup (`uv sync`), the commands, what the metrics mean, and the caveat that equity is vs a random hand.

## Pitfalls

- Preflop action order is reversed in HU (the button acts first preflop and last postflop). This is the most common engine bug.
- Min-raise: `last_raise` is the size of the last raise *increment*, not the raise-to amount.
- A short all-in doesn't need re-open logic, because bets are capped at the effective stack.
- Don't let equity depend on the global `random`. Seeding per `(hole, board)` keeps runs reproducible and cacheable.
- The laya output probabilities are rounded to 4dp, so renormalize before `rng.choices`.
- Don't use `a["confidence"]` or `act_probability` for anything; they're uncalibrated or broken upstream.
- If CUDA is unavailable, CPU inference may be very slow (one independent review measured seconds or more per call). Print a warning.

## Out of scope for v1 (upgrade path, don't build now)

- Range-weighted equity (narrow the villain range by their actions).
- A human-vs-Laya interactive mode.
- Solver-label generation and fine-tuning (see the brief: the Kaggle notebook `laya_finetune_typed_decisions_2xT4_kaggle.ipynb`,
  JSONL rows with `state`/`questions`/`gold{label, probabilities}`).
- An MLP baseline trained on the same features.
- `predict_batch` across parallel tables for throughput.
- Multiway pots, and stack depths other than 100bb.
