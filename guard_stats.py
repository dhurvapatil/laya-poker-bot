"""How often the guard changed Laya's move, from a sim.py --log file.
probs = Laya before the guard (alpha=1, so what it would have sampled); final_probs = after the guard."""
import collections
import json
import sys

WHY = {"fold": "fold with a hand well ahead of the price", "call": "call when far behind the price",
       "allin": "all-in with a medium hand and deep stacks"}

for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path)]
    n = len(rows)
    touched = changed = top_blocked = 0.0
    by_kind, by_street = collections.Counter(), collections.Counter()
    for r in rows:
        removed = {k: v for k, v in r["probs"].items() if k not in r["final_probs"]}
        mass = sum(removed.values())
        touched += mass >= 0.1   # ignore slivers (all-in at 0.5% is removed almost every time)
        changed += mass                                   # chance the unguarded bot plays a blocked action
        top_blocked += max(r["probs"], key=r["probs"].get) in removed
        by_street[r["state"]["street"]] += mass
        for k, v in removed.items():
            by_kind[WHY.get(k, "bet/raise with a weak hand")] += v
    print("%s: %d decisions" % (path.rsplit("/", 1)[-1], n))
    print("  guard removed >=10%% of Laya's probability: %d (%.0f%%)" % (touched, 100 * touched / n))
    print("  moves the guard changed (expected): %.0f (%.0f%%)" % (changed, 100 * changed / n))
    print("  Laya's top choice was blocked: %d (%.0f%%)" % (top_blocked, 100 * top_blocked / n))
    for k, v in by_kind.most_common():
        print("    %-45s %4.0f" % (k, v))
    print("  by street: " + ", ".join("%s %.0f" % kv for kv in by_street.most_common()))
