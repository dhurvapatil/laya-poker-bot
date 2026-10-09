"""Offline decision quiz: fixed poker situations with reference EVs, scored by expected regret.

Much faster and far less noisy than playing matches. Situations come from real hands against a known
reference opponent; each legal action gets a reference EV; an agent's action *probabilities* (no sampling
noise) are scored by expected regret against the best action. See README.md for what the EV assumes.

  python quiz.py --n 150 --agent pot_odds --agent pot_odds:range=1 --agent laya --agent laya:guard=1,range=1
"""
import argparse
import copy
import json
import os
import random
import statistics

import agents as ag
from poker import BB, play_hand, rank7, strengths

STREET_CARDS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
SNAP_PROB = {"preflop": 0.08, "flop": 0.35, "turn": 0.4, "river": 0.5}


class RefVillain:
    """The reference opponent: the pot-odds rule with hand-rank percentile standing in for equity.
    Its policy is known exactly, so which hands it could hold given its past actions can be worked out."""

    @staticmethod
    def decide(eq, to_call, pot_total, options):
        return ag.pot_odds_decision(eq, to_call / (pot_total + to_call) if to_call else 0.0, options)

    def act(self, obs, options):
        eq = strengths(tuple(sorted(obs["board"])))[tuple(sorted(obs["hole"]))]
        return self.decide(eq, obs["to_call"], obs["pot_total"], options)


class Snapper:
    """Hero while collecting situations: epsilon-random pot-odds play (for varied spots) that snapshots
    some decisions, at most one per hand."""

    def __init__(self, rng, snaps, eps=0.3):
        self.rng, self.snaps, self.eps, self.taken = rng, snaps, eps, False
        self.base = ag.PotOddsAgent()

    def end_hand(self, acts, seat):
        self.taken = False

    def act(self, obs, options):
        if not self.taken and len(options) >= 3 and self.rng.random() < SNAP_PROB[obs["street"]]:
            self.snaps.append({"obs": copy.deepcopy(obs), "options": dict(options)})
            self.taken = True
        if self.rng.random() < self.eps:
            return self.rng.choice(list(options))
        return self.base.act(obs, options)


def consistent_hands(obs):
    """Villain hands that RefVillain would have played exactly as it did so far this hand."""
    mine = set(obs["hole"])
    cands = [h for h in strengths(tuple(sorted(obs["board"]))) if not mine & set(h)]
    for a in obs["acts"]:
        if a["seat"] == obs["seat"]:
            continue
        st = strengths(tuple(sorted(obs["board"][:STREET_CARDS[a["street"]]])))
        cands = [h for h in cands if RefVillain.decide(st[h], a["to_call"], a["pot_before"], a["opts"]) == a["label"]]
    return cands


def option_evs(obs, options, n=1000, seed=0):
    """Reference EV in bb of each legal action (fold = 0; sunk chips ignored).

    Model: villain's hand is drawn from the consistent hands; after hero's action the hand is checked down
    to showdown. Facing hero's raise, villain folds or calls by its rule (re-raises are treated as calls).
    # ponytail: check-down continuation, so it under-rewards aggression and position; a solver is the upgrade.
    The same samples are used for every action, so differences between actions are low-noise.
    """
    rng = random.Random(seed)
    hole, board = tuple(sorted(obs["hole"])), tuple(obs["board"])
    cands = consistent_hands(obs) or list(strengths(tuple(sorted(board))))
    st = strengths(tuple(sorted(board)))
    dead = set(hole) | set(board)
    samples = []
    for h in rng.choices(cands, k=n):
        rest = [c for c in range(52) if c not in dead and c not in h]
        full = list(board) + rng.sample(rest, 5 - len(board))
        a, b = rank7(list(hole) + full), rank7(list(h) + full)
        samples.append((h, 1.0 if a > b else 0.5 if a == b else 0.0))
    pot, win = obs["pot_total"], sum(r for _, r in samples) / n
    evs = {}
    for label, (kind, amt) in options.items():
        if kind == "fold":
            ev = 0.0
        elif kind == "check":
            ev = win * pot
        elif kind == "call":
            a = amt - obs["my_bet"]
            ev = win * (pot + a) - a
        else:
            a, v = amt - obs["my_bet"], amt - obs["opp_bet"]
            ev = sum(pot if RefVillain.decide(st[h], v, pot + a, ("fold", "call")) == "fold"
                     else r * (pot + a + v) - a for h, r in samples) / n
        evs[label] = ev / BB
    return evs


def build(n, seed):
    rng = random.Random(seed)
    sits, i = [], 0
    while len(sits) < n:
        deck = random.Random(seed * 1_000_003 + i).sample(range(52), 52)
        hero_seat = i % 2
        new = []
        agents = [None, None]
        agents[hero_seat], agents[1 - hero_seat] = Snapper(rng, new), RefVillain()
        play_hand(agents, deck)
        for s in new:
            s["vhole"] = deck[2 * (1 - hero_seat):2 * (1 - hero_seat) + 2]   # true villain hand; for tests only
            s["evs"] = option_evs(s["obs"], s["options"], seed=seed * 1000 + len(sits))
            sits.append(s)
        i += 1
    return sits[:n]


def load_or_build(path, n, seed, rebuild):
    if os.path.exists(path) and not rebuild:
        with open(path) as f:
            sits = json.load(f)
        for s in sits:
            s["options"] = {k: tuple(v) for k, v in s["options"].items()}
        return sits
    print("building %d situations (one-off, ~1 min)..." % n)
    sits = build(n, seed)
    with open(path, "w") as f:
        json.dump(sits, f)
    return sits


def score(agent, sits):
    """Per situation: (expected regret in bb, mass on near-best actions, mass on blunders > 1bb)."""
    out = []
    for s in sits:
        p = agent.dist(s["obs"], s["options"])
        best = max(s["evs"].values())
        reg = {k: best - v for k, v in s["evs"].items()}
        out.append((sum(p.get(k, 0) * r for k, r in reg.items()),
                    sum(p.get(k, 0) for k, r in reg.items() if r <= 0.1),
                    sum(p.get(k, 0) for k, r in reg.items() if r > 1.0)))
    return out


def ci(xs):
    return 1.96 * statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150, help="number of situations")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--agent", action="append", help="agent spec (repeatable); the first is the paired baseline")
    ap.add_argument("--file", help="situations file (default quiz_<seed>_<n>.json)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--laya-model", default="convaiinnovations/laya")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    sits = load_or_build(args.file or "quiz_%d_%d.json" % (args.seed, args.n), args.n, args.seed, args.rebuild)
    specs = args.agent or ["pot_odds", "pot_odds:range=1", "laya", "laya:guard=1", "laya:range=1,guard=1",
                           "laya:range=1,guard=1,alpha=0.5"]
    kw = dict(model=args.laya_model, device=args.device)
    rows = [("random (floor)", score(ag.make("random", **kw), sits)),
            ("calling_station", score(ag.make("calling_station", **kw), sits))]
    rows += [(sp, score(ag.make(sp, seed=args.seed, **kw), sits)) for sp in specs]
    base = rows[2][1]
    by_street = {}
    for s in sits:
        by_street[s["obs"]["street"]] = by_street.get(s["obs"]["street"], 0) + 1
    print("%d situations (%s)\n" % (len(sits), ", ".join("%s %d" % kv for kv in by_street.items())))
    print("%-34s %-18s %-10s %-9s %s" % ("agent", "regret bb (95% CI)", "near-best", "blunder", "vs %s (paired)" % rows[2][0]))
    for name, sc in rows:
        reg = [x[0] for x in sc]
        diff = [a[0] - b[0] for a, b in zip(sc, base)]
        print("%-34s %5.2f +/- %-9.2f %5.1f%%    %5.1f%%   %+.2f +/- %.2f" % (
            name, statistics.mean(reg), ci(reg), 100 * statistics.mean(x[1] for x in sc),
            100 * statistics.mean(x[2] for x in sc), statistics.mean(diff), ci(diff)))
    print("\nregret = EV given up vs the best action, in big blinds (lower is better); "
          "near-best = probability mass on actions within 0.1bb of best; blunder = mass on actions losing >1bb.")


if __name__ == "__main__":
    main()
