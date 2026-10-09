"""Run: python test_poker.py  (no torch/laya needed)"""
import random

import agents as ag
import quiz
from agents import CallingStation, PotOddsAgent, RandomAgent
from poker import (BB, SB, STACK, OppStats, draws, equity, features, hand_detail, legal_options, parse, play_hand,
                   range_equity, rank7, texture, villain_k)


def r(s):
    return rank7(parse(s))


def test_evaluator():
    order = [
        r("As Ks Qs Js Ts 2d 3c"),   # royal flush
        r("9c 9d 9h 9s Ac 2d 3c"),   # quads
        r("Kc Kd Kh 2s 2c 7d 8h"),   # full house
        r("Ac 9c 7c 5c 3c Kd Qh"),   # flush
        r("5c 6d 7h 8s 9c Ad 2h"),   # straight
        r("Qc Qd Qh 2s 5c 7d 9h"),   # trips
        r("Jc Jd 4h 4s Ac 7d 9h"),   # two pair
        r("Tc Td 2h 5s Ac 7d 9h"),   # pair
        r("Ac Jd 9h 6s 4c 3d 2h"),   # high card
    ]
    assert order == sorted(order, reverse=True) and len(set(order)) == 9
    assert r("Ac 2d 3h 4s 5c 9d Kh") < r("2c 3d 4h 5s 6c 9d Kh")             # wheel < six-high
    assert r("Ac Ad 9h 7s 4c 3d 2h") > r("Ac Ad 9h 6s 4c 3d 2h")             # pair kicker
    assert r("Kc Kd 7h 7s 5c 5d 2h") == (2, 11, 5, 3)                       # 3 pairs: kicker is the third pair's rank
    assert r("Kc Kd Kh 7s 7c 7d 2h")[:3] == (6, 11, 5)                       # two trips = full house
    assert r("Ac 9c 7c 5c 3c 8d 6h") > r("5c 6d 7h 8s 9c Ad 2h")             # flush > straight
    assert r("As Ks Qs Js Ts 9s 2d")[0] == 8                                 # straight flush beats flush


def test_equity():
    assert abs(equity(tuple(sorted(parse("As Ah"))), ()) - 0.85) < 0.03
    assert abs(equity(tuple(sorted(parse("7c 2d"))), ()) - 0.35) < 0.04
    assert equity(tuple(sorted(parse("As Ks"))), tuple(sorted(parse("Qs Js Ts 2d 3c")))) == 1.0


def test_options():
    o = legal_options(0, [SB, BB], [STACK - SB, STACK - BB], 0, BB)
    assert o["third_pot"] == ("raise", 200) and o["two_thirds_pot"] == ("raise", 233) and o["pot"] == ("raise", 300), o
    assert o["allin"] == ("raise", STACK) and o["call"] == ("call", BB) and "check" not in o
    short = legal_options(0, [0, 0], [100, 100], 1, BB)   # 1bb stacks: only all-in is a legal size
    assert "third_pot" not in short and short["allin"] == ("raise", 100), short


def test_engine():
    rng = random.Random(0)
    for i in range(300):
        deck = random.Random(i).sample(range(52), 52)
        a, b = play_hand([RandomAgent(rng), RandomAgent(rng)], deck)
        assert a + b == 0 and abs(a) <= STACK, (i, a, b)
    log = []
    for i in range(20):
        deck = random.Random(i).sample(range(52), 52)
        log.clear()
        a, b = play_hand([CallingStation(), CallingStation()], deck, log)
        assert any(l.startswith("showdown") for l in log) and abs(a) in (0, BB), (a, log)


def test_duplicate_symmetry():
    for i in range(50):
        deck = random.Random(i).sample(range(52), 52)
        d1, d2 = play_hand([CallingStation(), CallingStation()], deck), play_hand([CallingStation(), CallingStation()], deck)
        assert d1[0] + d2[1] == 0   # A in seat 0 then seat 1 with identical bots nets exactly 0


def test_pot_odds_beats_random():
    rng, net = random.Random(1), 0
    for i in range(40):
        deck = random.Random(i).sample(range(52), 52)
        net += play_hand([PotOddsAgent(), RandomAgent(rng)], deck)[0] + play_hand([RandomAgent(rng), PotOddsAgent()], deck)[1]
    assert net > 0, net


def test_range_equity():
    hole, board = tuple(sorted(parse("7c 7d"))), tuple(sorted(parse("Kh 9c 2d")))
    assert abs(range_equity(hole, board, 0) - equity(hole, board)) < 0.07   # k=0 is a random hand
    assert range_equity(hole, board, 30) < range_equity(hole, board, 0) - 0.08   # a strong range beats a mid pair more
    nuts = tuple(sorted(parse("As Ks"))), tuple(sorted(parse("Qs Js Ts 2d 3c")))
    assert range_equity(*nuts, 30) == 1.0


def test_acts_and_opp_stats():
    class Rec(CallingStation):
        seen = None
        def end_hand(self, acts, seat):
            Rec.seen = (acts, seat)
    a, b = Rec(), CallingStation()
    play_hand([a, b], random.Random(3).sample(range(52), 52))
    acts, seat = Rec.seen
    assert seat == 0 and acts[0]["seat"] == 0 and acts[0]["street"] == "preflop" and acts[0]["kind"] == "call"
    assert acts[0]["add"] == BB - SB and acts[0]["to_call"] == BB - SB and "call" in acts[0]["opts"]
    o = OppStats()
    o.end_hand([{"seat": 1, "kind": "raise", "to_call": 0}, {"seat": 1, "kind": "fold", "to_call": 50},
                {"seat": 0, "kind": "call", "to_call": 0}], 0)
    assert (o.hands, o.acts, o.raises, o.faced, o.folds) == (1, 2, 1, 1, 1)
    assert 0.3 < OppStats().agg < 0.4   # prior only
    big = [{"seat": 1, "kind": "raise", "add": 700, "pot_before": 700, "street": "flop", "to_call": 0}]
    small = [{"seat": 1, "kind": "raise", "add": 200, "pot_before": 700, "street": "flop", "to_call": 0}]
    assert villain_k(big, 0) > villain_k(small, 0) > villain_k([], 0) == 0
    maniac = OppStats()
    maniac.acts = maniac.raises = 100
    assert villain_k(big, 0, maniac) < villain_k(big, 0)   # a maniac's bet says less than a passive player's


def test_guard():
    opts = {"fold": 0, "call": 0, "half_pot": 0, "pot": 0, "allin": 0}
    g = ag.guard(opts, {"equity": 0.9, "pot_odds": 0.25, "spr": 10})
    assert "fold" not in g and "call" in g and "allin" in g
    g = ag.guard(opts, {"equity": 0.9, "pot_odds": 0.25, "spr": 2})
    assert "allin" in g
    g = ag.guard(opts, {"equity": 0.6, "pot_odds": 0.25, "spr": 10})
    assert "allin" not in g and "pot" in g
    g = ag.guard(opts, {"equity": 0.2, "pot_odds": 0.5, "spr": 10})
    assert g == {"fold"}
    assert ag.guard({"check": 0, "pot": 0}, {"equity": 0.2, "pot_odds": 0, "spr": 5}) == {"check"}


def test_make_specs():
    assert isinstance(ag.make("pot_odds:range=1"), PotOddsAgent) and ag.make("pot_odds:range=1").use_range
    for bad in ("nope", "pot_odds:bogus=1", "random:range=1"):
        try:
            ag.make(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_laya_inputs():
    obs = {"hole": parse("Ah Kd"), "board": parse("Kh 7c 2d"), "street": "flop", "position": "BB", "pot_total": 730,
           "to_call": 180, "my_stack": 9450, "opp_stack": 9270, "my_bet": 0, "opp_bet": 180, "seat": 1,
           "history": ["flo:BTN bet to 1.8bb"], "acts": []}
    opts = legal_options(550, [180, 0], [9270, 9450], 1, BB)
    crit = ag.laya_criteria(obs, opts)
    assert list(crit) == list(opts) and crit["call"] == "call 1.8 bb" and "fold" in crit
    st = ag.laya_state(obs, features(obs), False)
    assert st["to_call_bb"] == 1.8 and "equity_vs_random" in st and "villain_range" not in st
    st = ag.laya_state(obs, features(obs, use_range=True), True)
    assert "equity_vs_range" in st and st["villain_range"] == "any hand"


def test_descriptions():
    def hd(h, b):
        return hand_detail(parse(h), parse(b))
    assert hd("Ah Qd", "Ks Qc 2d") == "middle pair" and hd("Kh 4d", "Ks Qc 2d") == "top pair, weak kicker"
    assert hd("Kh Ad", "Ks Qc 2d") == "top pair, good kicker" and hd("Ah Ad", "Ks Qc 2d") == "overpair"
    assert hd("2h 2c", "Ks Qc 2d") == "set" and hd("Ah Jd", "7s 5c 2d") == "no pair, 2 overcards"
    assert hd("3h 4c", "As Ks Qs Js Ts") == "board plays (straight flush)" and hd("Ah Ad", "") == "pair of As"
    assert draws(parse("Ah 5h"), parse("Kh 9h 2c")) == ["flush draw"]
    assert draws(parse("8c 7d"), parse("6s 5h Kd")) == ["open-ended straight draw"]
    assert draws(parse("8c 7d"), parse("5s 4h Kd")) == ["gutshot"] and draws(parse("8c 7d"), parse("6s 5h Kd 2c 3c")) == []
    assert texture(parse("Kh 9h 2c")) == "flush draw possible" and texture(parse("Kc 8d 2h")) == "dry"
    assert texture(parse("Kh Kd 5h 4h 3c")) == "paired, flush possible, straight possible"


def test_tree_dist():
    ans = lambda **p: {"probabilities": p}
    opts = {"fold": 0, "call": 0, "third_pot": 0, "pot": 0, "allin": 0}
    p = ag.tree_dist(opts, {"play": ans(fold=0.2, play=0.8), "aggr": {"probabilities": {"call": 0.5, "raise": 0.5}},
                            "size": ans(third_pot=0.5, pot=0.25, allin=0.25)})
    assert abs(p["fold"] - 0.2) < 1e-9 and abs(p["call"] - 0.4) < 1e-9 and abs(p["third_pot"] - 0.2) < 1e-9
    assert abs(sum(p.values()) - 1) < 1e-9
    p = ag.tree_dist({"check": 0, "allin": 0}, {"aggr": ans(check=0.7, bet=0.3)})   # one size: no size question
    assert abs(p["allin"] - 0.3) < 1e-9
    obs = {"to_call": 0, "pot_total": 300}
    assert list(ag.tree_questions(obs, {"check": 0, "allin": ("raise", 9900)}, {"equity": 0.5, "pot_odds": 0})) == ["aggr"]


def test_quiz_oracle():
    sits = quiz.build(12, seed=5)
    for s in sits:
        v = tuple(sorted(s["vhole"]))
        assert v in quiz.consistent_hands(s["obs"]), s["obs"]["acts"]   # the true hand is never ruled out
        assert s["evs"].get("fold", 0.0) == 0.0
    river = {"hole": parse("Th 2c"), "board": parse("Ah Kh Qh Jh 3c"), "street": "river", "seat": 1, "my_bet": 0,
             "opp_bet": 700, "pot_total": 1400, "to_call": 700,
             "acts": [{"seat": 0, "street": "river", "label": "pot", "kind": "raise", "add": 700, "pot_before": 700,
                       "to_call": 0, "opts": ("check", "half_pot", "pot", "allin")}]}
    opts = {"fold": ("fold", 0), "call": ("call", 700)}
    assert quiz.option_evs(river, opts)["call"] > 0                       # royal flush: calling is +EV
    river["hole"] = parse("7c 2d")
    assert quiz.option_evs(river, opts)["call"] < 0                       # air facing a pot bet: calling loses


def test_facing_allin_acts():
    class Shove:
        def act(self, obs, opts):
            return "allin" if "allin" in opts else "call" if "call" in opts else "check"

    class Fold:
        asked = 0

        def act(self, obs, opts):
            Fold.asked += obs["opp_stack"] == 0
            return "fold" if "fold" in opts else "check"
    deck = random.Random(3).sample(range(52), 52)
    assert play_hand([Shove(), Fold()], deck) == (BB, -BB) and Fold.asked == 1   # BB gets to fold to the shove
    d = play_hand([Shove(), Shove()], deck)
    assert d[0] == -d[1] and abs(d[0]) < STACK                                  # called shove: zero-sum, paid by equity


def test_allin_ev():
    from poker import runout_equity
    aa, kk = tuple(parse("Ah As")), tuple(parse("Kh Ks"))
    assert 0.79 < runout_equity(aa, kk, ()) < 0.85                               # AA vs KK preflop ~82%
    board = tuple(parse("2c 7d 9h"))
    assert runout_equity(aa, kk, board) + runout_equity(kk, aa, board) == 1      # exact on the flop
    assert runout_equity(aa, kk, tuple(parse("2c 7d 9h Kd"))) == 2 / 44          # KK has a set: AA needs one of 2 aces



if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("ok")
