"""Heads-up PREFLOP training rows for finetune.py, labelled by rollouts (FINETUNE_PLAN.md Phase 1, preflop only).

PokerBench is postflop only, so the first fine-tune never saw a preflop spot (it folded the SB to a 0.5 bb complete).
Spots: every hero preflop decision, hero = epsilon-random pot-odds (quiz.Snapper's policy), villain = quiz.RefVillain.
Label: for each legal action, replay the hand with villain's hand drawn from quiz.consistent_hands and a random board,
force the action, then both seats play RefVillain to the end. The same deals are used for every action (common random
numbers), so the differences are low-noise. target = softmax(EV_bb / TAU). Rows match convert_pokerbench.py (rich state).
# ponytail: rule-bot continuation, so labels = "best action if both play simply afterwards"; round 2 = Laya continuation.

  uv run python gen_preflop.py data/pre_train --spots 8000 --seed 1
  uv run python gen_preflop.py data/pre_test --spots 500 --seed 2
  uv run python gen_preflop.py x --check          # AA in the SB must not fold, 72o must not shove
"""
import argparse
import copy
import json
import math
import multiprocessing
import random

import agents as ag
import quiz
from poker import BB, parse, play_hand

TAU = 0.5   # bb


class Script:
    """Plays the shared label list `seq` first (both seats pop from the same list), then `policy`."""

    def __init__(self, seq, policy):
        self.seq, self.policy = seq, policy

    def act(self, obs, options):
        if self.seq:
            label = self.seq.pop(0)
            assert label in options, (label, list(options))
            return label
        return self.policy.act(obs, options)


class PreSnap(quiz.Snapper):
    """Snapshots every preflop decision, then plays like Snapper."""

    def act(self, obs, options):
        if obs["street"] == "preflop":
            self.snaps.append({"obs": copy.deepcopy(obs), "options": dict(options)})
        return self.rng.choice(list(options)) if self.rng.random() < self.eps else self.base.act(obs, options)


def spots(n, seed):
    rng = random.Random(seed)
    out, i = [], 0
    while len(out) < n:
        deck = random.Random(seed * 1_000_003 + i).sample(range(52), 52)
        hero = i % 2
        agents = [None, None]
        agents[hero], agents[1 - hero] = PreSnap(rng, out), quiz.RefVillain()
        play_hand(agents, deck)
        i += 1
    return out[:n]


def evs(obs, options, n, seed, hands=None, weights=None, hero=None, villain=None):
    """EV in bb (hero's net for the whole hand) of each legal action, by `n` rollouts.
    Villain's hand is drawn from `hands` (default quiz.consistent_hands) by `weights`; the rest of the board is random.
    After the forced action hero plays `hero` and villain `villain` (both default quiz.RefVillain)."""
    rng = random.Random(seed)
    s, hole, board = obs["seat"], obs["hole"], obs["board"]
    hero, villain = hero or quiz.RefVillain(), villain or quiz.RefVillain()
    decks = []
    for h in rng.choices(quiz.consistent_hands(obs) if hands is None else hands, weights, k=n):
        deck = [0] * 9
        deck[2 * s:2 * s + 2], deck[2 - 2 * s:4 - 2 * s] = hole, list(h)
        deck[4:9] = board + rng.sample([c for c in range(52) if c not in hole and c not in h and c not in board],
                                       5 - len(board))
        decks.append(deck)
    prefix = [a["label"] for a in obs["acts"]]
    out = {}
    for label in options:
        tot = 0
        for deck in decks:
            seq = prefix + [label]
            agents = [None, None]
            agents[s], agents[1 - s] = Script(seq, hero), Script(seq, villain)
            tot += play_hand(agents, deck)[s]
        out[label] = tot / n / BB
    return out


def to_row(obs, options, ev, state):
    """A finetune.py row (flat question): target = softmax(EV_bb / TAU) in option order."""
    keys, m = list(options), max(ev.values())
    w = [math.exp((ev[k] - m) / TAU) for k in keys]
    return {"state": state, "qid": "action", "question": {
        "type": "choice", "instructions": ag.FLAT_INSTRUCTIONS, "criteria": ag.laya_criteria(obs, options)},
        "target": [x / sum(w) for x in w], "label": max(ev, key=ev.get), "street": obs["street"],
        "evs": {k: round(v, 3) for k, v in ev.items()}}


def row(args):
    snap, n, seed = args
    obs, options = snap["obs"], snap["options"]
    return to_row(obs, options, evs(obs, options, n, seed), ag.laya_state(obs, ag.features(obs), rich=True))


def check(n):
    from poker import legal_options, STACK, SB
    for hole, bad in (("As Ah", "fold"), ("7c 2d", "allin")):
        obs = {"hole": parse(hole), "board": [], "street": "preflop", "position": "BTN", "pot_total": SB + BB,
               "to_call": BB - SB, "my_stack": STACK - SB, "opp_stack": STACK - BB, "history": [], "seat": 0,
               "my_bet": SB, "opp_bet": BB, "acts": []}
        ev = evs(obs, legal_options(0, [SB, BB], [STACK - SB, STACK - BB], 0, BB), n, 0)
        print(hole, {k: round(v, 2) for k, v in ev.items()})
        assert max(ev, key=ev.get) != bad and ev[bad] < max(ev.values()), (hole, bad)
    print("check ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output prefix; writes <out>.flat.jsonl")
    ap.add_argument("--spots", type=int, default=8000)
    ap.add_argument("--rollouts", type=int, default=128, help="deals per spot, shared by all actions")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--procs", type=int, default=22)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        return check(args.rollouts)
    snaps = spots(args.spots, args.seed)
    with open(args.out + ".flat.jsonl", "w") as f, multiprocessing.Pool(args.procs) as pool:
        jobs = ((s, args.rollouts, args.seed * 1_000_003 + i) for i, s in enumerate(snaps))
        # ponytail: imap_unordered, not imap -- imap blocks forever on one OOM-killed worker's chunk even
        # though the pool replaces it (2026-09-30: a 7h stall, 59/8000 rows). Row order doesn't matter here.
        for k, r in enumerate(pool.imap_unordered(row, jobs, chunksize=4)):
            f.write(json.dumps(r) + "\n")
            if (k + 1) % 500 == 0:
                print("  %d/%d" % (k + 1, len(snaps)), flush=True)
    print("%d preflop rows -> %s.flat.jsonl" % (len(snaps), args.out))


if __name__ == "__main__":
    main()
