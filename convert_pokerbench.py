"""PokerBench postflop CSV (solver labels) -> JSONL training rows in the exact state/question format LayaAgent uses.

Every PokerBench postflop spot is a two-player pot. IP maps to our BTN (seat 0), OOP to our BB (seat 1). 100bb stacks.
The one solver bet size maps to the nearest of our sizes. Suits are randomly relabelled per row (poker is
suit-symmetric), because the whole set has only 22 distinct flops.

  uv run python convert_pokerbench.py data/postflop_500k_train_set_game_scenario_information.csv data/train --n 200000
Writes <out>.flat.jsonl (one action question) and <out>.tree.jsonl (fold/play, call/raise, size).
"""
import argparse
import csv
import json
import multiprocessing
import random

import agents as ag
from poker import BB, PASSIVE, STACK, legal_options, parse

ORDER = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]   # postflop acting order: later = in position
SB_DEAD = BB // 2   # the small blind left in the pot when SB folded preflop
SMOOTH = 0.1   # label smoothing: the solver picks one action; keep a little mass on the rest


def cards(s, perm):
    return [c // 4 * 4 + perm[c % 4] for c in parse(" ".join(s[i:i + 2] for i in range(0, len(s), 2)))]


def spot(row, perm):
    """Replay the PokerBench line into our engine's obs, options and label. None if it doesn't fit."""
    pre = row["preflop_action"].split("/")
    names = [pre[0], pre[2]]                                    # opener, then the caller or 3-bettor
    ip = max(names, key=ORDER.index)
    seat_of = {n: 0 if n == ip else 1 for n in names}
    pos = {0: "BTN", 1: "BB"}
    amounts = [float(t[:-2]) for t in pre if t.endswith("bb")]
    put = round(amounts[-1] * BB)
    dead = (SB_DEAD if "SB" not in names else 0) + (BB if "BB" not in names else 0)
    pot, bet, stack = 2 * put + dead, [0, 0], [STACK - put, STACK - put]
    acts, hist = [], []
    for i, amt in enumerate(amounts):                           # preflop raises, then the final call
        s = seat_of[pre[2 * i]]
        acts.append({"seat": s, "street": "preflop", "kind": "raise"})
        hist.append("pre:%s raise to %.1fbb" % (pos[s], amt))
    acts.append({"seat": 1 - acts[-1]["seat"], "street": "preflop", "kind": "call"})
    hist.append("pre:%s call" % pos[acts[-1]["seat"]])

    streets = iter(("flop", "turn", "river"))
    street, last_raise = next(streets), BB
    toks = row["postflop_action"].split("/")
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "dealcards":
            pot, bet, last_raise, street = pot + sum(bet), [0, 0], BB, next(streets)
            i += 2
            continue
        side, kind, *amt = t.split("_")
        s, o = (0, 1) if side == "IP" else (1, 0)
        if kind in ("BET", "RAISE"):
            to = round(float(amt[0]) * BB)
            last_raise = to - bet[o]
            stack[s] -= to - bet[s]
            bet[s] = to
            hist.append("%s:%s %s to %.1fbb" % (street[:3], pos[s], kind.lower(), to / BB))
        elif kind == "CALL":
            add = min(bet[o] - bet[s], stack[s])
            stack[s] -= add
            bet[s] += add
            hist.append("%s:%s call" % (street[:3], pos[s]))
        elif kind == "CHECK":
            hist.append("%s:%s check" % (street[:3], pos[s]))
        else:
            return None
        acts.append({"seat": s, "street": street, "kind": "raise" if kind in ("BET", "RAISE") else kind.lower()})
        i += 1
    p = 0 if row["hero_position"] == "IP" else 1
    if street != row["evaluation_at"].lower() or stack[p] <= 0:
        return None
    board = cards(row["board_flop"], perm) + [c for k in ("board_turn", "board_river")[:["flop", "turn", "river"].index(street)]
                                               for c in cards(row[k], perm)]
    obs = {"hole": cards(row["holding"], perm), "board": board, "street": street, "position": pos[p],
           "pot_total": pot + sum(bet), "to_call": bet[1 - p] - bet[p], "my_stack": stack[p],
           "opp_stack": stack[1 - p], "history": hist, "seat": p, "my_bet": bet[p], "opp_bet": bet[1 - p],
           "acts": acts}
    if abs(obs["pot_total"] / BB - float(row["pot_size"])) > 1.5:   # our replay disagrees with the source
        return None
    options = legal_options(pot, bet, stack, p, last_raise)
    word, *size = row["correct_decision"].split()
    label = word.lower()
    if size:
        raises = {k: v[1] for k, v in options.items() if k not in PASSIVE}
        if not raises:
            return None
        to = round(float(size[0]) * BB)
        label = min(raises, key=lambda k: abs(raises[k] - to))
    return (obs, options, label) if label in options else None



def soft(label, keys):
    base = SMOOTH / len(keys)
    return [base + (1 - SMOOTH) * (k == label) for k in keys]


def convert(args):
    i, row, seed = args
    rng = random.Random(seed * 1_000_003 + i)
    perm = rng.sample(range(4), 4)
    got = spot(row, perm)
    if got is None:
        return None
    obs, options, label = got
    f = ag.features(obs)
    state = ag.laya_state(obs, f, rich=True)
    flat = {"state": state, "qid": "action", "question": {
        "type": "choice", "instructions": ag.FLAT_INSTRUCTIONS, "criteria": ag.laya_criteria(obs, options)},
        "target": soft(label, list(options)), "label": label, "street": obs["street"], "src": i}
    tree = []
    qs = ag.tree_questions(obs, options, f)
    raises = [k for k in options if k not in PASSIVE]
    ans = {"play": "fold" if label == "fold" else "play",                    # which answer the solver's move implies
           "aggr": None if label == "fold" else ("call" if label == "call" else "check" if label == "check"
                                                 else "raise" if obs["to_call"] else "bet"),
           "size": label if label in raises else None}
    for qid, q in qs.items():
        if ans[qid] is not None:                                             # no signal, e.g. call-vs-raise after a fold
            tree.append({"state": state, "qid": qid, "question": q, "target": soft(ans[qid], list(q["criteria"])),
                         "label": ans[qid], "street": obs["street"], "src": i})
    return flat, tree


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("out", help="output prefix")
    ap.add_argument("--n", type=int, default=0, help="rows to use (random sample), 0 = all")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--procs", type=int, default=20)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    if args.n:
        rows = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    skipped = n_flat = n_tree = 0
    with open(args.out + ".flat.jsonl", "w") as ff, open(args.out + ".tree.jsonl", "w") as ft, \
            multiprocessing.Pool(args.procs) as pool:
        for k, res in enumerate(pool.imap(convert, ((i, r, args.seed) for i, r in enumerate(rows)), chunksize=256)):
            if res is None:
                skipped += 1
                continue
            flat, tree = res
            ff.write(json.dumps(flat) + "\n")
            n_flat += 1
            for t in tree:
                ft.write(json.dumps(t) + "\n")
                n_tree += 1
            if (k + 1) % 20000 == 0:
                print("  %d/%d" % (k + 1, len(rows)), flush=True)
    print("%d rows -> %d flat, %d tree question rows, %d skipped (%.1f%%)"
          % (len(rows), n_flat, n_tree, skipped, 100 * skipped / len(rows)))


if __name__ == "__main__":
    main()
