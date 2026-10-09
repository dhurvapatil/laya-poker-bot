"""Heads-up POSTFLOP training rows for finetune.py, labelled by rollouts (FINETUNE_PLAN.md Phase 1).

Sessions of --hands hands: hero = epsilon-random pot-odds (quiz.Snapper's policy) with one OppStats alive, so the
villain-stats line appears in the state as in real play. Villain per session: pot_odds (quiz.RefVillain),
calling_station, or loose (RefVillain rule on equity +SHIFT, plus a random bet BLUFF of the time).
Snapshots: at most one hero flop/turn/river decision per hand, weighted toward turn/river (SNAP_PROB).
Label: for each legal action, deal villain a hand weighted by how likely his style was to play it as he did, deal the
rest of the board, force the action, then play out with the engine: hero = --cont, villain = its session style.
The same deals are used for every action (common random numbers). target = softmax(EV_bb / TAU).
State = what `laya:rich=1,range=1` sees (agents.laya_state with the session's OppStats); flat question only.
# ponytail: rule-bot continuation, so labels = "best action if both play simply afterwards"; round 2 = Laya continuation.

  uv run python gen_data.py data/roll_test --sessions 60 --seed 2
  uv run python gen_data.py data/roll_train --sessions 2000 --seed 1 --minutes 110   # writes data/roll_train.flat.jsonl
  uv run python gen_data.py x --check                                      # state == LayaAgent's, nuts call, value bet
  uv run python gen_data.py x --stats data/roll_train.flat.jsonl           # label stats of a finished file
"""
import argparse
import collections
import copy
import json
import multiprocessing
import random
import statistics
import time

import agents as ag
import poker
import quiz
from gen_preflop import Script, evs, to_row
from poker import PASSIVE, OppStats, parse, play_hand, strengths

SNAP_PROB = {"preflop": 0.0, "flop": 0.2, "turn": 0.45, "river": 1.0}   # flop spots cost the most
SHIFT, BLUFF = 0.12, 0.15   # loose villain: plays as if 12% more equity, and bets at random 15% of the time


class Loose(quiz.RefVillain):
    """The pot-odds rule with shifted thresholds plus random bets.
    # ponytail: the random bet is a hash of the spot, so it is repeatable across rollouts (common random numbers)."""

    @staticmethod
    def decide(eq, to_call, pot_total, options):
        return quiz.RefVillain.decide(min(1.0, eq + SHIFT), to_call, pot_total, options)

    def act(self, obs, options):
        raises = [k for k in options if k not in PASSIVE]
        r = random.Random(hash((tuple(obs["hole"]), tuple(obs["board"]), len(obs["acts"])))).random()
        if raises and r < BLUFF:
            return raises[int(r / BLUFF * len(raises))]
        return super().act(obs, options)


STYLES = {"pot_odds": quiz.RefVillain, "calling_station": ag.CallingStation, "loose": Loose}


def likelihood(style, eq, a):
    """P(villain of `style` with hand-rank `eq` took act `a`)."""
    if style == "calling_station":
        return 1.0
    hit = STYLES[style].decide(eq, a["to_call"], a["pot_before"], a["opts"]) == a["label"]
    raises = [k for k in a["opts"] if k not in PASSIVE]
    if style == "pot_odds" or not raises:
        return float(hit)
    return (1 - BLUFF) * hit + BLUFF * (a["label"] in raises) / len(raises)


def villain_range(obs, style):
    """Villain hands and their weights given what he did this hand (quiz.consistent_hands for any style)."""
    mine = set(obs["hole"])
    hands = [h for h in strengths(tuple(sorted(obs["board"]))) if not mine & set(h)]
    w = [1.0] * len(hands)
    for a in obs["acts"]:
        if a["seat"] != obs["seat"]:
            st = strengths(tuple(sorted(obs["board"][:quiz.STREET_CARDS[a["street"]]])))
            w = [x * likelihood(style, st[h], a) for h, x in zip(hands, w)]
    return hands, w


class Hero(quiz.Snapper):
    """Snapper with an OppStats kept across the session; snapshots postflop decisions only."""

    def __init__(self, rng, snaps):
        super().__init__(rng, snaps)
        self.opp = OppStats()

    def end_hand(self, acts, seat):
        self.taken = False
        self.opp.end_hand(acts, seat)

    def act(self, obs, options):
        if not self.taken and self.rng.random() < SNAP_PROB[obs["street"]]:
            self.snaps.append({"obs": copy.deepcopy(obs), "options": dict(options), "opp": copy.copy(self.opp)})
            self.taken = True
        return self.rng.choice(list(options)) if self.rng.random() < self.eps else self.base.act(obs, options)


def cont_agent(spec, opp):
    """Hero's continuation policy: 'ref' = quiz.RefVillain (hand-rank pot-odds), else an agents.make spec."""
    if spec == "ref":
        return quiz.RefVillain()
    a = ag.make(spec)
    a.opp = copy.copy(opp)   # its opponent model as of the snapshot
    return a


def label(snap, style, n, seed, cont):
    obs, options, opp = snap["obs"], snap["options"], snap["opp"]
    hands, w = villain_range(obs, style)
    ev = evs(obs, options, n, seed, hands, w, cont_agent(cont, opp), STYLES[style]())
    state = ag.laya_state(obs, ag.features(obs, opp, True), True, opp, rich=True)
    r = to_row(obs, options, ev, state)
    r.update(villain=style, seed=seed)
    return r


def session(args):
    """One session: play `hands` hands vs one villain style, label every snapshot. Returns the rows."""
    sid, seed, hands, n, cont = args
    s = seed * 1_000_003 + sid
    rng, snaps = random.Random(s), []
    style = list(STYLES)[sid % len(STYLES)]
    hero = Hero(rng, snaps)
    for i in range(hands):
        agents = [None, None]
        agents[i % 2], agents[1 - i % 2] = hero, STYLES[style]()
        play_hand(agents, random.Random(s * 1000 + i).sample(range(52), 52))
    rows = []
    for j, snap in enumerate(snaps):
        rows.append(label(snap, style, n, s * 1000 + j, cont))
        poker.strengths.cache_clear()   # ponytail: ~130 KB per board; keeps 16 workers in RAM, costs a few re-ranks
    return rows


def spot(seq, deck):
    """(obs, options) of the first decision after the scripted labels `seq`, dealt from `deck`."""
    class Stop(Exception):
        pass

    class Grab:
        def act(self, obs, options):
            raise Stop(obs, options)
    try:
        play_hand([Script(seq, Grab()), Script(seq, Grab())], deck)
    except Stop as e:
        return e.args
    raise ValueError("hand ended")


def check(n, cont):
    # 1. state/question == what LayaAgent (laya:rich=1,range=1) sends to the model, with the model stubbed out
    sent = {}

    class Fake:
        def tok(self, s, add_special_tokens=False):
            return {"input_ids": []}

        def predict(self, state, questions):
            sent.update(state=state, questions=questions)
            return {"answers": {"action": {"probabilities": {}, "answer_confidence": 1.0}}}
    ag._load = lambda model, device: Fake()
    rows, snaps = [], []
    for sid in range(3):
        rng, sn = random.Random(sid), []
        hero = Hero(rng, sn)
        style = list(STYLES)[sid]
        for i in range(40):
            agents = [None, None]
            agents[i % 2], agents[1 - i % 2] = hero, STYLES[style]()
            play_hand(agents, random.Random(i).sample(range(52), 52))
        snaps += [(s, style) for s in sn[-7:]]
    assert any(s["opp"].hands >= 20 for s, _ in snaps)
    for s, style in snaps[:20]:
        r = json.loads(json.dumps(label(s, style, 8, 0, cont)))
        la = ag.make("laya:rich=1,range=1")
        la.opp = s["opp"]
        la.dist(s["obs"], s["options"])
        assert r["state"] == sent["state"] and r["question"] == sent["questions"]["action"], (r, sent)
    assert any("villain" in r["state"] for r in [json.loads(json.dumps(label(s, st, 2, 0, cont))) for s, st in snaps[-5:]])
    print("state/question match LayaAgent on 20 spots")
    # 2. nuts on the river facing a pot bet: calling beats folding
    deck = parse("Js Ts 9c 8d As Ks Qs 2d 7h")        # hero BTN royal flush, villain bets on the river
    seq = ["call", "check", "check", "check", "check", "check", "third_pot"]
    obs, options = spot(seq, deck)
    hands, w = villain_range(obs, "pot_odds")
    ev = evs(obs, options, n, 0, hands, w, cont_agent(cont, OppStats()), quiz.RefVillain())
    print("nuts river:", {k: round(v, 2) for k, v in ev.items()})
    assert ev["fold"] < ev["call"]
    # 3. a set on the river vs calling_station, checked to hero: some bet beats checking
    deck = parse("7c 7d Ah 4c 7h Kd 2s 9c 3d")         # hero BTN 77
    obs, options = spot(["call", "check"] + ["check"] * 5, deck)
    hands, w = villain_range(obs, "calling_station")
    ev = evs(obs, options, n, 0, hands, w, cont_agent(cont, OppStats()), ag.CallingStation())
    print("set vs calling_station:", {k: round(v, 2) for k, v in ev.items()})
    assert max(v for k, v in ev.items() if k not in PASSIVE) > ev["check"]
    print("check ok")


def stats(rows):
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["street"]][r["label"]] += 1
    for street in ("preflop", "flop", "turn", "river"):
        c = by.get(street)
        if c:
            k = sum(c.values())
            print("  %-7s %6d rows | best: %s" % (street, k, ", ".join("%s %.0f%%" % (a, 100 * v / k) for a, v in c.most_common())))
    aggr = [r["label"] not in PASSIVE for r in rows]
    print("  bet/raise is the best action in %.1f%% of spots" % (100 * statistics.mean(aggr)))
    for v in sorted({r.get("villain") for r in rows} - {None}):
        print("    vs %-15s %.1f%%" % (v, 100 * statistics.mean(r["label"] not in PASSIVE for r in rows if r["villain"] == v)))
    gap = [sorted(r["evs"].values())[-1] - sorted(r["evs"].values())[-2] for r in rows]
    spread = [max(r["evs"].values()) - min(r["evs"].values()) for r in rows]
    print("  mean EV gap best - 2nd best %.2f bb (median %.2f), best - worst %.2f bb; mean max target %.2f" % (
        statistics.mean(gap), statistics.median(gap), statistics.mean(spread), statistics.mean(max(r["target"]) for r in rows)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output prefix; writes <out>.flat.jsonl")
    ap.add_argument("--sessions", type=int, default=100)
    ap.add_argument("--hands", type=int, default=100, help="hands per session (one OppStats per session)")
    ap.add_argument("--rollouts", type=int, default=64, help="deals per spot, shared by all actions")
    ap.add_argument("--cont", default="ref", help="hero continuation: 'ref' (quiz.RefVillain) or an agent spec, "
                    "e.g. pot_odds:range=1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--minutes", type=float, default=0, help="stop after this wall time (0 = no limit)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stats", help="print label stats of this .jsonl and exit")
    args = ap.parse_args()
    if args.check:
        return check(args.rollouts, args.cont)
    if args.stats:
        with open(args.stats) as f:
            return stats([json.loads(line) for line in f])
    rows, t0 = [], time.time()
    with open(args.out + ".flat.jsonl", "w") as f, multiprocessing.Pool(args.procs) as pool:
        jobs = ((i, args.seed, args.hands, args.rollouts, args.cont) for i in range(args.sessions))
        for k, rs in enumerate(pool.imap_unordered(session, jobs)):
            for r in rs:
                f.write(json.dumps(r) + "\n")
            f.flush()
            rows += rs
            if (k + 1) % 20 == 0:
                print("  %d/%d sessions, %d rows, %.1f rows/s" % (k + 1, args.sessions, len(rows),
                                                             len(rows) / (time.time() - t0)), flush=True)
            if args.minutes and time.time() - t0 > 60 * args.minutes:
                print("  time limit: stopping after %d sessions" % (k + 1))
                break
    print("%d postflop rows -> %s.flat.jsonl" % (len(rows), args.out))
    stats(rows)


if __name__ == "__main__":
    main()
