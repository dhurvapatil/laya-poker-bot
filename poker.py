"""Heads-up no-limit hold'em: cards, 7-card evaluator, Monte Carlo equity, engine, features."""
import bisect
import functools
import itertools
import math
import random

RANKS, SUITS = "23456789TJQKA", "cdhs"
SB, BB, STACK = 50, 100, 10_000
SIZES = (("third_pot", 1 / 3), ("two_thirds_pot", 2 / 3), ("pot", 1.0))   # bet/raise sizes besides all-in
PASSIVE = ("fold", "check", "call")   # every other label is a bet or raise
CATEGORY = ["high card", "pair", "two pair", "trips", "straight", "flush",
            "full house", "quads", "straight flush"]


def card_str(c):
    return RANKS[c // 4] + SUITS[c % 4]


def parse(s):
    return [RANKS.index(t[0]) * 4 + SUITS.index(t[1]) for t in s.split()]


def straight_high(ranks):
    """High rank of the best straight in a set of ranks, or -1. The wheel counts as 3 (a '5' high)."""
    for hi in range(12, 3, -1):
        if all(hi - i in ranks for i in range(5)):
            return hi
    return 3 if {12, 0, 1, 2, 3} <= ranks else -1


def rank7(cards):
    """Hand strength for 5-7 cards as a tuple: higher is better."""
    counts = {}
    for c in cards:
        counts[c // 4] = counts.get(c // 4, 0) + 1
    flush = None
    for s in range(4):
        sr = sorted((c // 4 for c in cards if c % 4 == s), reverse=True)
        if len(sr) >= 5:
            sh = straight_high(set(sr))
            if sh >= 0:
                return (8, sh)
            flush = (5, *sr[:5])
            break
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    top, top_rank = groups[0][1], groups[0][0]
    if top == 4:
        return (7, top_rank, max(r for r in counts if r != top_rank))
    if top == 3 and len(groups) > 1 and groups[1][1] >= 2:
        return (6, top_rank, groups[1][0])
    if flush:
        return flush
    sh = straight_high(set(counts))
    if sh >= 0:
        return (4, sh)
    if top == 3:
        return (3, top_rank, *sorted((r for r in counts if r != top_rank), reverse=True)[:2])
    if top == 2:
        pairs = [r for r, n in groups if n == 2]
        if len(pairs) >= 2:
            p1, p2 = pairs[0], pairs[1]
            return (2, p1, p2, max(r for r in counts if r not in (p1, p2)))
        return (1, top_rank, *sorted((r for r in counts if r != top_rank), reverse=True)[:3])
    return (0, *sorted(counts, reverse=True)[:5])


@functools.lru_cache(maxsize=100_000)
def equity(hole, board, n=500):
    """P(win) + P(tie)/2 for `hole` vs ONE uniformly random hand. `hole`/`board` are sorted tuples.

    # ponytail: equity vs a random hand, not a range; range-weighted equity is the upgrade.
    """
    rng = random.Random(hash((hole, board)))
    dead = set(hole) | set(board)
    deck = [c for c in range(52) if c not in dead]
    need = 5 - len(board)
    score = 0.0
    for _ in range(n):
        draw = rng.sample(deck, 2 + need)
        full = list(board) + draw[2:]
        a, b = rank7(list(hole) + full), rank7(draw[:2] + full)
        score += 1.0 if a > b else 0.5 if a == b else 0.0
    return score / n


@functools.lru_cache(maxsize=10_000)
def runout_equity(h0, h1, board, n=5000):
    """Seat 0's share of the pot vs seat 1's known hand over every remaining runout.
    Exact from the flop on; preflop (1.7M boards) samples `n` of them."""
    rest = [c for c in range(52) if c not in h0 + h1 + board]
    need = 5 - len(board)
    if math.comb(len(rest), need) <= n:
        runs = list(itertools.combinations(rest, need))
    else:
        rng = random.Random(hash((h0, h1, board)))
        runs = [rng.sample(rest, need) for _ in range(n)]
    score = 0.0
    for r in runs:
        full = list(board) + list(r)
        a, b = rank7(list(h0) + full), rank7(list(h1) + full)
        score += 1.0 if a > b else 0.5 if a == b else 0.0
    return score / len(runs)


def legal_options(pot, bet, stack, p, last_raise):
    """{label: (kind, raise_to_or_call_total)}. `bet` is this street's per-seat bet, `pot` earlier streets."""
    o = 1 - p
    to_call = bet[o] - bet[p]
    pot_total = pot + sum(bet)
    opts = {}
    if to_call > 0:
        opts["fold"] = ("fold", 0)
        opts["call"] = ("call", bet[p] + min(to_call, stack[p]))
    else:
        opts["check"] = ("check", 0)
    cap = min(bet[p] + stack[p], bet[o] + stack[o])
    if stack[o] > 0 and cap > bet[o]:
        min_to = bet[o] + max(last_raise, BB)
        seen = set()
        for label, frac in SIZES:
            t = max(int(round(bet[o] + frac * (pot_total + to_call))), min_to)
            if t < cap and t not in seen:
                seen.add(t)
                opts[label] = ("raise", t)
        opts["allin"] = ("raise", cap)
    return opts


def play_hand(agents, deck, log=None):
    """One hand. Seat 0 is button/SB. Returns net chip deltas (seat0, seat1); they sum to 0.

    `log`, if a list, receives a readable trace of the hand. Agents with an `end_hand(acts, seat)`
    method are told what happened afterwards (used for opponent stats).
    """
    acts = []
    deltas = _play(agents, deck, log, acts)
    for seat, a in enumerate(agents):
        if hasattr(a, "end_hand"):
            a.end_hand(acts, seat)
    return deltas


def _play(agents, deck, log, acts):
    hole = [deck[0:2], deck[2:4]]
    full_board = deck[4:9]
    stack, bet, pot, hist = [STACK - SB, STACK - BB], [SB, BB], 0, []
    say = log.append if log is not None else (lambda _: None)
    say("hole: BTN %s | BB %s | board %s" % (
        " ".join(map(card_str, hole[0])), " ".join(map(card_str, hole[1])),
        " ".join(map(card_str, full_board))))

    def result(winners_share):
        d = [stack[i] + winners_share[i] - STACK for i in (0, 1)]
        say("result: BTN %+d chips, BB %+d chips" % tuple(d))
        return d[0], d[1]

    seen = 0   # board cards dealt before the betting closed
    for street, nb in (("preflop", 0), ("flop", 3), ("turn", 4), ("river", 5)):
        if street != "preflop":
            pot += sum(bet)
            bet = [0, 0]
            if 0 in stack:   # all-in and called: no more decisions, so pay by equity instead of dealing it out
                break
        board = full_board[:nb]
        last_raise = BB
        p = 0 if street == "preflop" else 1   # heads-up: button acts first preflop, last postflop
        acted = [False, False]
        while stack[p] > 0 and (stack[1 - p] > 0 or bet[1 - p] > bet[p]):   # facing an all-in still gets to act
            o = 1 - p
            opts = legal_options(pot, bet, stack, p, last_raise)
            pos = "BTN" if p == 0 else "BB"
            obs = {"hole": list(hole[p]), "board": list(board), "street": street, "position": pos,
                   "pot_total": pot + sum(bet), "to_call": bet[o] - bet[p], "my_stack": stack[p],
                   "opp_stack": stack[o], "history": list(hist), "seat": p, "my_bet": bet[p],
                   "opp_bet": bet[o], "acts": list(acts)}
            label = agents[p].act(obs, opts)
            kind, amt = opts[label]
            add = 0 if kind in ("fold", "check") else amt - bet[p]
            acts.append({"seat": p, "pos": pos, "street": street, "label": label, "kind": kind, "add": add,
                         "pot_before": obs["pot_total"], "to_call": obs["to_call"], "opts": tuple(opts)})
            note = getattr(agents[p], "note", "")
            if kind == "fold":
                hist.append("%s:%s fold" % (street[:3], pos))
                say("%s %s: fold %s" % (street, pos, note))
                total = pot + sum(bet)
                return result([total if o == 0 else 0, total if o == 1 else 0])
            if kind == "call":
                stack[p] -= add
                bet[p] = amt
                hist.append("%s:%s call" % (street[:3], pos))
            elif kind == "raise":
                last_raise = amt - bet[o]
                stack[p] -= add
                bet[p] = amt
                acted[o] = False
                hist.append("%s:%s %s to %.1fbb" % (street[:3], pos, "bet" if bet[o] == 0 else "raise", amt / BB))
            else:
                hist.append("%s:%s check" % (street[:3], pos))
            acted[p] = True
            say("%s %s: %s%s %s" % (street, pos, label, " (%.1fbb)" % (amt / BB) if kind == "raise" else "", note))
            if bet[0] == bet[1] and all(acted):
                break
            p = o
        seen = nb
    total = pot + sum(bet)
    if seen < 5:
        eq = runout_equity(tuple(hole[0]), tuple(hole[1]), tuple(full_board[:seen]))
        say("all-in before the river: BTN equity %.1f%%, paid by equity (all-in EV)" % (100 * eq))
        x = round(total * eq)
        return result([x, total - x])
    a, b = rank7(hole[0] + full_board), rank7(hole[1] + full_board)
    if a > b:
        share = [total, 0]
    elif b > a:
        share = [0, total]
    else:
        share = [total // 2, total - total // 2]   # odd chip to seat 1
    say("showdown: BTN %s vs BB %s" % (CATEGORY[a[0]], CATEGORY[b[0]]))
    return result(share)


def preflop_name(hole):
    a, b = sorted(hole, key=lambda c: -(c // 4))
    if a // 4 == b // 4:
        return "pair of %ss" % RANKS[a // 4]
    return RANKS[a // 4] + RANKS[b // 4] + ("s" if a % 4 == b % 4 else "o")


def preflop_score(hole):
    """Rough preflop strength (higher = better). # ponytail: heuristic, not a solved hand ranking."""
    hi, lo = sorted((c // 4 for c in hole), reverse=True)
    if hi == lo:
        return 20 + 2 * hi
    return 2 * hi + lo + (3 if hole[0] % 4 == hole[1] % 4 else 0) - min(hi - lo - 1, 3) * 1.5 + (2 if hi - lo == 1 else 0)


@functools.lru_cache(maxsize=4096)
def strengths(board):
    """{hand: P(beats a random hand)} from hand-rank percentile, over all hands not touching `board`
    (a sorted tuple). Made hands only, so draws are undervalued. # ponytail: no draw/runout awareness."""
    hands = list(itertools.combinations([c for c in range(52) if c not in board], 2))
    key = (lambda h: rank7(list(h) + list(board))) if len(board) >= 3 else preflop_score
    keys = [key(h) for h in hands]
    order = sorted(keys)
    n = len(hands)
    out = {}
    for h, k in zip(hands, keys):
        lo, hi = bisect.bisect_left(order, k), bisect.bisect_right(order, k)
        out[h] = (lo + 0.5 * (hi - lo - 1)) / (n - 1)
    return out


class OppStats:
    """What we have seen villain do over completed hands (with a weak prior, so early numbers are sane)."""

    def __init__(self):
        self.hands = self.acts = self.raises = self.faced = self.folds = 0

    def end_hand(self, acts, my_seat):
        self.hands += 1
        for a in acts:
            if a["seat"] == my_seat:
                continue
            self.acts += 1
            self.raises += a["kind"] == "raise"
            if a["to_call"] > 0:
                self.faced += 1
                self.folds += a["kind"] == "fold"

    @property
    def agg(self):
        return (self.raises + 3.5) / (self.acts + 10)

    @property
    def fold_to_bet(self):
        return (self.folds + 1.5) / (self.faced + 5)


def villain_k(acts, my_seat, opp=None):
    """How top-heavy villain's range is, from what they did this hand. Bigger bets and calls mean a
    stronger range; preflop actions count half (opens are wide). Passive opponents' aggression means more
    than a maniac's, so the total is scaled by their measured aggression. # ponytail: hand-tuned constants."""
    k = 0.0
    for a in acts:
        if a["seat"] == my_seat:
            continue
        scale = 0.5 if a["street"] == "preflop" else 1.0
        if a["kind"] == "raise":
            k += scale * (0.6 + 0.9 * min(a["add"] / max(a["pot_before"], 1), 1.5))
        elif a["kind"] == "call":
            k += scale * 0.3
    return k * (min(1.5, max(0.5, 1.6 - 1.5 * opp.agg)) if opp else 1.0)


def range_word(k):
    return "any hand" if k < 0.5 else "wide" if k < 1.5 else "medium-strong" if k < 2.5 else "very strong"


@functools.lru_cache(maxsize=100_000)
def range_equity(hole, board, k10, n=400):
    """Equity vs a villain range weighted toward strong hands: weight = (0.05 + percentile) ** k, k = k10/10.
    k = 0 is a uniformly random hand. `hole`/`board` are sorted tuples."""
    st = strengths(board)
    hands = [h for h in st if hole[0] not in h and hole[1] not in h]
    weights = [(0.05 + st[h]) ** (k10 / 10) for h in hands]
    rng = random.Random(hash((hole, board, k10)))
    dead = set(hole) | set(board)
    score = 0.0
    for h in rng.choices(hands, weights, k=n):
        rest = [c for c in range(52) if c not in dead and c not in h]
        full = list(board) + rng.sample(rest, 5 - len(board))
        a, b = rank7(list(hole) + full), rank7(list(h) + full)
        score += 1.0 if a > b else 0.5 if a == b else 0.0
    return score / n


def hand_detail(hole, board):
    """Hero's hand relative to the board: 'top pair, weak kicker', 'overpair', 'set', 'no pair, 2 overcards'..."""
    if len(board) < 3:
        return preflop_name(hole)
    r = rank7(hole + board)
    if len(board) == 5 and r == rank7(board):
        return "board plays (%s)" % CATEGORY[r[0]]
    hr = sorted((c // 4 for c in hole), reverse=True)
    br = sorted({c // 4 for c in board}, reverse=True)
    if r[0] == 3:
        return "set" if hr[0] == hr[1] else "trips"
    if r[0] == 2:
        n = sum(p in hr for p in r[1:3])
        return "two pair" if n == 2 else "one pair plus a board pair" if n else "board two pair"
    if r[0] == 1 and r[1] in hr:
        p = r[1]
        if hr[0] == hr[1]:
            return "overpair" if p > br[0] else "underpair" if p < br[-1] else "pocket pair below the top card"
        i = br.index(p)
        if i == 0:
            return "top pair, %s kicker" % ("good" if (hr[1] if hr[0] == p else hr[0]) >= 10 else "weak")
        return "bottom pair" if i == len(br) - 1 else "middle pair"
    if r[0] <= 1:   # high card, or the only pair is on the board
        over = sum(h > br[0] for h in hr)
        return "no pair" + (", %d overcard%s" % (over, "s" * (over > 1)) if over else "")
    return CATEGORY[r[0]]


def draws(hole, board):
    """Draws on the flop/turn that use a hole card: flush draw, open-ended straight draw, gutshot."""
    if not 3 <= len(board) <= 4:
        return []
    cards, out = hole + board, []
    if any(sum(c % 4 == s for c in cards) == 4 and any(c % 4 == s for c in hole) for s in range(4)):
        out.append("flush draw")
    ranks, br = {c // 4 for c in cards}, {c // 4 for c in board}
    if straight_high(ranks) < 0:
        outs = sum(straight_high(ranks | {r}) >= 0 and straight_high(br | {r}) < 0 for r in range(13))
        if outs:
            out.append("open-ended straight draw" if outs >= 2 else "gutshot")
    return out


def texture(board):
    """How dangerous the board is: 'paired, flush possible, straight possible', or 'dry'."""
    if len(board) < 3:
        return "none"
    br = [c // 4 for c in board]
    suit = max(sum(c % 4 == s for c in board) for s in range(4))
    t = ["paired"] if len(set(br)) < len(br) else []
    if suit >= 3:
        t.append("flush possible")
    elif suit == 2 and len(board) < 5:
        t.append("flush draw possible")
    u = set(br) | ({-1} if 12 in br else set())   # ace also plays low
    if any(len(u & set(range(lo, lo + 5))) >= 3 for lo in range(-1, 9)):
        t.append("straight possible")
    return ", ".join(t) or "dry"


def line(obs):
    """Pot type and who was the last aggressor this hand, from `obs["acts"]`."""
    acts, me = obs.get("acts", []), obs.get("seat", 0)
    pre = sum(a["kind"] == "raise" for a in acts if a["street"] == "preflop")
    last = [a["seat"] for a in acts if a["kind"] == "raise"]
    return (["limped", "raised", "3-bet", "4-bet+"][min(pre, 3)],
            ("hero" if last[-1] == me else "villain") if last else "nobody")


def villain_words(opp):
    """# ponytail: hand-picked thresholds; fine for these bots, re-check against human-like opponents."""
    a = "aggressive" if opp.agg > 0.45 else "passive" if opp.agg < 0.25 else "balanced"
    f = "folds a lot" if opp.fold_to_bet > 0.5 else "rarely folds" if opp.fold_to_bet < 0.25 else "folds sometimes"
    return "%s, %s" % (a, f)


def features(obs, opp=None, use_range=False):
    """The externally computed poker math shared by the math baseline and the Laya agent.
    With `use_range`, equity is against a range narrowed by villain's actions instead of a random hand."""
    hole, board = obs["hole"], obs["board"]
    to_call, pot_total = obs["to_call"], obs["pot_total"]
    h, b = tuple(sorted(hole)), tuple(sorted(board))
    k = villain_k(obs.get("acts", []), obs.get("seat", 0), opp) if use_range else 0.0
    return {
        "equity": range_equity(h, b, round(k * 10)) if use_range else equity(h, b),
        "range_k": k,
        "pot_odds": to_call / (pot_total + to_call) if to_call else 0.0,
        "spr": min(obs["my_stack"], obs["opp_stack"]) / pot_total,
        "hand": CATEGORY[rank7(hole + board)[0]] if len(board) >= 3 else preflop_name(hole),
    }
