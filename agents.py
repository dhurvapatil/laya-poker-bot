"""Poker agents. Each has `act(obs, options) -> label` and `dist(obs, options) -> {label: prob}`.

Build one from a spec string with `make`: "pot_odds", "laya:alpha=0.5,guard=1,range=1", ...
"""
import functools
import json
import random
import statistics
import sys
import time

from poker import (BB, PASSIVE, OppStats, card_str, draws, features, hand_detail, line, range_word, rank7, strengths,
                   texture, villain_words)


def norm(d):
    s = sum(d.values())
    return {k: v / s for k, v in d.items()} if s > 0 else {}


def onehot(label, options):
    return {k: float(k == label) for k in options}


def pot_odds_decision(eq, pot_odds, options):
    """The math rule: raise when far ahead, call when equity beats the price (+5%), else fold/check.
    `options` is anything iterable of labels."""
    raises = [k for k in options if k not in PASSIVE]   # smallest first, all-in last
    if "call" in options:
        if eq > 0.75 and raises:
            return raises[-1]
        return "call" if eq >= pot_odds + 0.05 else "fold"
    if eq > 0.65 and raises:
        return "pot" if "pot" in options else raises[-1]
    if eq > 0.5 and raises:
        return raises[0]
    return "check"


def guard(options, f):
    """Labels a sane player might pick, given equity `f["equity"]`. Never empty.

    Never fold when far ahead of the price, never call when far behind, never raise with a weak hand,
    and only shove when strong or shallow. # ponytail: hand-tuned thresholds; also means no bluffs.
    """
    eq, po = f["equity"], f["pot_odds"]
    ok = set(options)
    facing = "fold" in options
    if facing and eq >= po + 0.15:
        ok.discard("fold")
    if facing and eq <= po - 0.10:
        ok.discard("call")
    if eq < 0.40:
        ok -= {k for k in options if k not in PASSIVE}
    if eq < 0.75 and f["spr"] > 3:
        ok.discard("allin")
    return ok or set(options)


class RandomAgent:
    def __init__(self, rng):
        self.rng = rng

    def dist(self, obs, options):
        return {k: 1 / len(options) for k in options}

    def act(self, obs, options):
        return self.rng.choice(list(options))


class CallingStation:
    def dist(self, obs, options):
        return onehot("call" if "call" in options else "check", options)

    def act(self, obs, options):
        return "call" if "call" in options else "check"


@functools.lru_cache(maxsize=100_000)
def top_range_equity(hole, board, k10, cut20, n=400):
    """`range_equity` vs what is left of villain's range after he folds his weakest cut20/20 of it (by weight).
    cut20 = 0 gives exactly range_equity. # ponytail: copy of poker.range_equity's sampler; merge if poker.py opens up."""
    st = strengths(board)
    hands = [h for h in st if hole[0] not in h and hole[1] not in h]
    weights = [(0.05 + st[h]) ** (k10 / 10) for h in hands]
    if cut20:
        lim, acc = sum(weights) * cut20 / 20, 0.0
        for i in sorted(range(len(hands)), key=lambda i: st[hands[i]]):
            acc += weights[i]
            if acc > lim:
                break
            weights[i] = 0.0
    rng = random.Random(hash((hole, board, k10)))
    dead = set(hole) | set(board)
    score = 0.0
    for h in rng.choices(hands, weights, k=n):
        rest = [c for c in range(52) if c not in dead and c not in h]
        full = list(board) + rng.sample(rest, 5 - len(board))
        a, b = rank7(list(hole) + full), rank7(list(h) + full)
        score += 1.0 if a > b else 0.5 if a == b else 0.0
    return score / n


FOLD_PRIOR = {"third_pot": 0.25, "two_thirds_pot": 0.4, "pot": 0.5, "allin": 0.6}   # s/(1+s): folds vs a sane defender


class PotOddsAgent:
    """The math baseline Laya has to beat. `use_range` swaps random-hand equity for range equity.

    `value`: postflop, pick among check/call and every bet size by one-street EV, using how often villain has folded
    to each size (weakest hands fold first). `bluff`: also consider bets with under 50% equity.
    # ponytail: one-street EV (ignores later streets and villain re-raises); a real tree search is the upgrade."""

    def __init__(self, use_range=False, value=False, bluff=False):
        self.use_range, self.opp = use_range, OppStats()
        self.value, self.bluff, self.folds = value, bluff, {}   # (street, label, facing) -> [folds, times]

    def end_hand(self, acts, seat):
        self.opp.end_hand(acts, seat)
        for a, b in zip(acts, acts[1:]):
            if a["seat"] == seat and a["kind"] == "raise" and a["street"] != "preflop" and b["seat"] != seat:
                s = self.folds.setdefault((a["street"], a["label"], a["to_call"] > 0), [0, 0])
                s[0] += b["kind"] == "fold"
                s[1] += 1

    def fold_rate(self, street, label, facing):
        f, n = self.folds.get((street, label, facing), (0, 0))
        return (f + 4 * FOLD_PRIOR.get(label, 0.5)) / (n + 4)

    def act(self, obs, options):
        f = features(obs, self.opp, self.use_range)
        base = pot_odds_decision(f["equity"], f["pot_odds"], options)
        if not self.value or obs["street"] == "preflop":
            return base
        eq, pot, to_call = f["equity"], obs["pot_total"], obs["to_call"]
        best = base if base in PASSIVE else "call" if "call" in options else "check"
        best_ev = 0.0 if best == "fold" else eq * (pot + to_call) - to_call
        if eq < 0.5 and not self.bluff:
            return best
        hole, board, k10 = tuple(sorted(obs["hole"])), tuple(sorted(obs["board"])), round(f["range_k"] * 10)
        for label, (kind, to) in options.items():
            if kind != "raise":
                continue
            fold = self.fold_rate(obs["street"], label, to_call > 0)
            add, called = to - obs["my_bet"], to - obs["opp_bet"]
            eqc = top_range_equity(hole, board, k10, round(fold * 20))
            ev = fold * pot + (1 - fold) * (eqc * (pot + add + called) - add)
            if ev > best_ev:
                best, best_ev = label, ev
        return best

    def dist(self, obs, options):
        return onehot(self.act(obs, options), options)


@functools.cache
def _load(model, device):
    import laya
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("warning: CPU inference is very slow; use a GPU", file=sys.stderr)
    return laya.load(model, device=device)


def laya_state(obs, f, use_range=False, opp=None, rich=False):
    """The compact JSON state Laya reads (keep it under ~250 tokens).
    `rich` adds hand/board descriptions in words and drops raw numbers the model reads poorly."""
    def bb(x):
        return round(x / BB, 1)
    d = f["equity"] - f["pot_odds"]
    edge = ("%+.2f (%s call)" % (d, "profitable" if d > 0 else "unprofitable")) if obs["to_call"] else "n/a (no bet)"
    if rich:
        hole, board = obs["hole"], obs["board"]
        beats = strengths(tuple(sorted(board)))[tuple(sorted(hole))]
        pot_type, aggressor = line(obs)
        state = {
            "street": obs["street"], "position": obs["position"],
            "hole": " ".join(map(card_str, hole)), "board": " ".join(map(card_str, board)) or "none",
            "hand": hand_detail(hole, board),
            "hand_rank": "beats %d%% of %s" % (round(100 * beats), "hands on this board" if board else "starting hands"),
            "draws": ", ".join(draws(hole, board)) or "none", "board_texture": texture(board),
            "pot": "%s pot, %.1f bb" % (pot_type, obs["pot_total"] / BB), "last_aggressor": aggressor,
        }
        if obs["to_call"]:
            state["facing"] = "bet of %.1f bb (%d%% of pot)" % (
                obs["to_call"] / BB, round(100 * obs["to_call"] / max(obs["pot_total"] - obs["to_call"], 1)))
        state.update({
            "effective_stack_bb": bb(min(obs["my_stack"], obs["opp_stack"])), "spr": round(f["spr"], 1),
            ("equity_vs_range" if use_range else "equity_vs_random"): round(f["equity"], 2),
        })
        if obs["to_call"]:
            state["calling"] = "needs %d%% equity: %s" % (round(100 * f["pot_odds"]), "profitable" if d > 0 else "unprofitable")
        if use_range:
            state["villain_range"] = range_word(f["range_k"])
        if opp and opp.hands >= 20:
            state["villain"] = villain_words(opp)
        state["history"] = "; ".join(obs["history"][-4:])
        return state
    state = {
        "street": obs["street"], "position": obs["position"],
        "hole": " ".join(map(card_str, obs["hole"])),
        "board": " ".join(map(card_str, obs["board"])) or "none",
        "hand": f["hand"], "pot_bb": bb(obs["pot_total"]), "to_call_bb": bb(obs["to_call"]),
        "stack_bb": bb(obs["my_stack"]), "opp_stack_bb": bb(obs["opp_stack"]),
        "spr": round(f["spr"], 1), "pot_odds": round(f["pot_odds"], 2),
        ("equity_vs_range" if use_range else "equity_vs_random"): round(f["equity"], 2),
        "equity_minus_pot_odds": edge,
    }
    if use_range:
        state["villain_range"] = range_word(f["range_k"])
    if opp and opp.hands >= 20:
        state["villain_stats"] = "aggression %.2f, folds to bets %.2f" % (opp.agg, opp.fold_to_bet)
    state["history"] = "; ".join(obs["history"][-6:])
    return state


def laya_criteria(obs, options):
    """One short description per legal action; same labels and order as `options`."""
    def bb(x):
        return "%.1f bb" % (x / BB)
    facing = obs["to_call"] > 0
    desc = {}
    for label, (kind, amt) in options.items():
        if label == "fold":
            desc[label] = "give up the hand"
        elif label == "check":
            desc[label] = "pass without betting"
        elif label == "call":
            desc[label] = "call " + bb(obs["to_call"])
        else:
            verb = "raise" if facing else "bet"
            desc[label] = ("all-in %s to %s" % (verb, bb(amt)) if label == "allin"
                           else "%s to %s (%s)" % (verb, bb(amt), label.replace("_", " ")))
    return desc


FLAT_INSTRUCTIONS = "You are hero in heads-up no-limit hold'em. Which action maximizes hero's expected winnings?"


SIZE_HINT = {"third_pot": "small: thin value, cheap bluff, keeps weaker hands in",
             "two_thirds_pot": "medium: standard value bet or semi-bluff",
             "pot": "large: strong hand or big draw, charges draws on wet boards",
             "allin": "all-in: very strong hand, or stacks are short"}


def tree_questions(obs, options, f):
    """The decision split into small questions: fold or play, call or raise (check or bet), then how big.
    Only the questions with more than one possible answer are asked."""
    raises = [k for k in options if k not in PASSIVE]
    qs = {}
    if obs["to_call"]:
        qs["play"] = {"type": "choice", "instructions": "Villain bet. Should hero give up the hand or keep playing?",
                      "criteria": {"fold": "give up; the hand is not worth the price",
                                   "play": "keep playing: needs %d%% equity, hero has %d%%"
                                           % (round(100 * f["pot_odds"]), round(100 * f["equity"]))}}
        if raises:
            qs["aggr"] = {"type": "choice", "instructions": "Hero keeps playing. Call or raise?",
                          "criteria": {"call": "call %.1f bb: see the next card, keep the pot small" % (obs["to_call"] / BB),
                                       "raise": "raise: build the pot with a strong hand, or push villain off a better one"}}
    elif raises:
        qs["aggr"] = {"type": "choice", "instructions": "No bet to hero. Check or bet?",
                      "criteria": {"check": "check: pot control, a free card, or give up",
                                   "bet": "bet: for value with a strong hand, or as a bluff"}}
    if len(raises) > 1:
        verb = "raise" if obs["to_call"] else "bet"
        qs["size"] = {"type": "choice", "instructions": "If hero %ss, how big?" % verb,
                      "criteria": {k: "%s to %.1f bb. %s" % (verb, options[k][1] / BB, SIZE_HINT.get(k, ""))
                                   for k in raises}}
    return qs


def tree_dist(options, answers):
    """Multiply the tree answers into one probability per legal label."""
    raises = [k for k in options if k not in PASSIVE]
    probs = {q: a["probabilities"] for q, a in answers.items()}
    play = probs["play"]["play"] if "play" in probs else 1.0
    aggr = probs["aggr"].get("raise", probs["aggr"].get("bet", 0.0)) if "aggr" in probs else 0.0
    size = probs.get("size") or {k: 1.0 for k in raises}
    p = {"fold": 1 - play} if "fold" in options else {}
    p["call" if "call" in options else "check"] = play * (1 - aggr)
    p.update({k: play * aggr * size.get(k, 0.0) for k in raises})
    return norm(p)


class LayaAgent:
    """Laya reads the state and returns action probabilities.

    alpha: trust in Laya, 1 = Laya alone, 0 = the pot-odds rule alone (geometric blend in between).
    guard: drop clearly bad actions before sampling.  use_range: equity vs a range, not a random hand.
    tree: ask fold/play, call/raise, size as separate small questions.  rich: describe hand and board in words.
    preflop: play the pot-odds rule preflop (the PokerBench fine-tune never saw preflop spots).
    """
    MAX_STATE_TOKENS = 250   # state past ~300 tokens is silently truncated by Laya (512 - 192 head budget)

    def __init__(self, model, device, greedy, rng, log_fn=None, alpha=1.0, guard=False, use_range=False,
                 tree=False, rich=False, preflop=False):
        self.laya = _load(model, device)
        self.greedy, self.rng, self.log_fn = greedy, rng, log_fn
        self.alpha, self.guard, self.use_range = alpha, guard, use_range
        self.tree, self.rich, self.preflop = tree, rich, preflop
        self.opp = OppStats()
        self.records = []      # per decision: street, chosen, latency_ms, answer_confidence
        self.max_tokens = 0
        self.note = ""
        self._last = {}

    def end_hand(self, acts, seat):
        self.opp.end_hand(acts, seat)

    def dist(self, obs, options):
        f = features(obs, self.opp, self.use_range)
        if self.preflop and obs["street"] == "preflop":
            p = onehot(pot_odds_decision(f["equity"], f["pot_odds"], options), options)
            self._last = {"state": {"preflop_rule": obs["street"]}, "criteria": list(options), "raw": p, "final": p,
                          "latency_ms": 0.0, "answer_confidence": 1.0, "tree": None}
            return p
        state = laya_state(obs, f, self.use_range, self.opp, self.rich)
        criteria = laya_criteria(obs, options)
        if self.tree:
            questions = tree_questions(obs, options, f)
        else:
            questions = {"action": {"type": "choice", "instructions": FLAT_INSTRUCTIONS, "criteria": criteria}}
        n_tok = len(self.laya.tok(json.dumps(state, ensure_ascii=False), add_special_tokens=False)["input_ids"])
        if n_tok > self.MAX_STATE_TOKENS and n_tok > self.max_tokens:
            print("warning: state is %d tokens; Laya may truncate it" % n_tok, file=sys.stderr)
        self.max_tokens = max(self.max_tokens, n_tok)
        t0 = time.perf_counter()
        answers = self.laya.predict(state, questions)["answers"] if questions else {}   # all questions, one batch
        latency = (time.perf_counter() - t0) * 1000
        if self.tree:
            raw = tree_dist(options, answers)
            conf = statistics.mean(a["answer_confidence"] for a in answers.values()) if answers else 1.0
        else:
            a = answers["action"]
            raw = norm({k: a["probabilities"].get(k, 0.0) for k in options})
            conf = a["answer_confidence"]
        raw = raw or {k: 1 / len(options) for k in options}
        p = raw
        math = pot_odds_decision(f["equity"], f["pot_odds"], options)
        if self.alpha < 1:
            eps = 0.1   # smooth the one-hot math choice so Laya can still break ties
            p = norm({k: raw[k] ** self.alpha * ((1 - eps) * (k == math) + eps / len(options)) ** (1 - self.alpha)
                      for k in options})
        if self.guard:
            ok = guard(options, f)
            p = norm({k: p[k] for k in ok}) or onehot(math if math in ok else next(iter(ok)), options)
            p = {k: v for k, v in p.items() if v > 0}
        self._last = {"state": state, "criteria": criteria, "raw": raw, "final": p, "latency_ms": latency,
                      "answer_confidence": conf,
                      "tree": {q: a["probabilities"] for q, a in answers.items()} if self.tree else None}
        return {k: p.get(k, 0.0) for k in options}

    def act(self, obs, options):
        p = self.dist(obs, options)
        chosen = max(p, key=p.get) if self.greedy else self.rng.choices(list(p), list(p.values()))[0]
        last = self._last
        self.note = "[laya %s]" % ", ".join("%s %.2f" % kv for kv in sorted(p.items(), key=lambda kv: -kv[1])[:3])
        self.records.append({"street": obs["street"], "chosen": chosen, "latency_ms": last["latency_ms"],
                             "answer_confidence": last["answer_confidence"]})
        if self.log_fn:
            self.log_fn({"state": last["state"], "options": last["criteria"], "probs": last["raw"], "tree": last["tree"],
                         "final_probs": last["final"], "chosen": chosen, "latency_ms": round(last["latency_ms"], 1)})
        return chosen


def make(spec, seed=1, model="convaiinnovations/laya", device=None, greedy=False, log_fn=None):
    """'kind' or 'kind:key=val,key=val' -> agent.
    kinds: laya (alpha, guard, range, tree, rich, preflop, model), pot_odds (range, value, bluff), calling_station, random."""
    kind, _, kv = spec.partition(":")
    opts = {}
    for part in filter(None, kv.split(",")):
        k, _, v = part.partition("=")
        opts[k] = v or "1"

    def flag(name):
        return opts.pop(name, "0") not in ("0", "false")
    use_range = flag("range")
    model = opts.pop("model", model)   # per-agent checkpoint, e.g. laya:rich=1,model=checkpoints/laya-poker-flat
    rng = random.Random(seed)
    if kind == "laya":
        agent = LayaAgent(model, device, greedy, rng, log_fn, alpha=float(opts.pop("alpha", 1.0)),
                          guard=flag("guard"), use_range=use_range, tree=flag("tree"), rich=flag("rich"),
                          preflop=flag("preflop"))
    elif kind == "pot_odds":
        agent = PotOddsAgent(use_range, value=flag("value"), bluff=flag("bluff"))
    elif kind == "calling_station":
        agent = CallingStation()
    elif kind == "random":
        agent = RandomAgent(rng)
    else:
        raise ValueError("unknown agent kind %r in %r" % (kind, spec))
    if kind in ("calling_station", "random") and use_range:
        raise ValueError("%r takes no range option" % kind)
    if opts:
        raise ValueError("unknown options %s in %r" % (sorted(opts), spec))
    agent.spec = spec
    return agent
