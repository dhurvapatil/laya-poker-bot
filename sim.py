"""Duplicate-format heads-up NLHE matches. See README.md."""
import argparse
import collections
import json
import random
import statistics
import sys

from agents import LayaAgent, make
from poker import BB, play_hand


def laya_report(side, agent):
    recs = agent.records
    if not recs:
        return
    lat = [r["latency_ms"] for r in recs]
    p95 = statistics.quantiles(lat, n=20)[18] if len(lat) > 1 else lat[0]
    print("\nLaya (%s): %d decisions | latency p50 %.1f ms, p95 %.1f ms | mean answer_confidence %.2f | max state tokens %d"
          % (side, len(recs), statistics.median(lat), p95,
             statistics.mean(r["answer_confidence"] for r in recs), agent.max_tokens))
    by_street = collections.defaultdict(collections.Counter)
    for r in recs:
        by_street[r["street"]][r["chosen"]] += 1
    for street in ("preflop", "flop", "turn", "river"):
        c = by_street.get(street)
        if c:
            n = sum(c.values())
            print("  %-8s %s" % (street, ", ".join("%s %.0f%%" % (k, 100 * v / n) for k, v in c.most_common())))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", default="laya", help="agent spec: kind[:key=val,...]; kinds laya|pot_odds|calling_station|random; "
                    "e.g. laya:alpha=0.5,guard=1,range=1")
    ap.add_argument("--b", default="pot_odds")
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", type=int, default=0, help="first deal index; split one run across processes")
    ap.add_argument("--greedy", action="store_true", help="Laya plays its argmax instead of sampling")
    ap.add_argument("--laya-model", default="convaiinnovations/laya", help="HF id or local fine-tuned dir")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--log", help="write one JSON line per Laya decision")
    ap.add_argument("-v", action="store_true", help="print every hand")
    args = ap.parse_args()

    logf = open(args.log, "w") if args.log else None
    log_fn = (lambda d: logf.write(json.dumps(d) + "\n")) if logf else None
    kw = dict(model=args.laya_model, device=args.device, greedy=args.greedy, log_fn=log_fn)
    A = make(args.a, seed=args.seed * 7 + 1, **kw)
    B = make(args.b, seed=args.seed * 7 + 2, **kw)

    nets = []
    for i in range(args.start, args.start + args.deals):
        deck = random.Random(args.seed * 1_000_003 + i).sample(range(52), 52)
        trace = [] if args.v else None
        d1 = play_hand([A, B], deck, trace)    # A on the button
        if trace is not None:
            trace.append("--- seats swapped ---")
        d2 = play_hand([B, A], deck, trace)    # same cards, B on the button
        nets.append(d1[0] + d2[1])
        if trace is not None:
            print("=== deal %d: A net %+d chips ===\n%s" % (i, nets[-1], "\n".join(trace)))
        elif len(nets) % 50 == 0:
            print("  ... %d/%d deals, A %+.1f bb/100 so far" % (len(nets), args.deals, statistics.mean(nets) / 2 / BB * 100),
                  file=sys.stderr, flush=True)

    n = len(nets)
    mean = statistics.mean(nets) / 2 / BB * 100
    ci = 1.96 * statistics.stdev(nets) / n ** 0.5 / 2 / BB * 100 if n > 1 else float("nan")
    print("\nA=%s vs B=%s | %d deals x 2 seats = %d hands | seed %d%s"
          % (args.a, args.b, n, 2 * n, args.seed, " | greedy" if args.greedy else ""))
    print("A win rate: %+.1f bb/100 (95%% CI +/- %.1f)" % (mean, ci))
    if abs(mean) < ci:
        print("CI crosses 0 -> not significant, run more deals")
    for side, agent in (("A", A), ("B", B)):
        if isinstance(agent, LayaAgent):
            laya_report(side, agent)
    if logf:
        logf.close()


if __name__ == "__main__":
    main()
