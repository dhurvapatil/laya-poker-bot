"""Fine-tune Laya's choice head (and the top encoder layers) on JSONL rows from convert_pokerbench.py.

Each row: {"state", "qid", "question", "target": [p per option], "label"}. Rows are encoded with Laya's own
input builder, so training input == play input. Only the choice logits are trained (soft cross-entropy).
The bottom encoder layers stay frozen so it fits an 8 GB GPU. Saves a folder that laya.load / --laya-model reads.

  uv run python finetune.py --train data/train.flat.jsonl --val data/test.flat.jsonl --out checkpoints/laya-poker-flat
  uv run python finetune.py --train data/test.flat.jsonl --val data/test.flat.jsonl --out /tmp/x --overfit 64   # wiring check
  # mix files: comma-separated, optional :N keeps N random rows of that file; --base warm-starts from a local checkpoint
  uv run python finetune.py --train data/roll_train.flat.jsonl,data/train.flat.jsonl:40000 --val data/roll_test.flat.jsonl,data/test.flat.jsonl \
      --base checkpoints/laya-poker-flat --out checkpoints/laya-poker-roll
"""
import argparse
import json
import math
import os
import random
import shutil
import time

import laya
import torch
from huggingface_hub import snapshot_download
from laya.common import collate_items, temp_bucket
from safetensors.torch import save_file


def load_rows(paths, n=0, seed=1, per_file=False):
    """paths: "a.jsonl,b.jsonl:N" (:N = N random rows of that file). n caps the total, or each file if per_file."""
    rows = []
    for spec in paths.split(","):
        path, _, k = spec.partition(":")
        with open(path) as f:
            rs = [dict(json.loads(line), _file=os.path.basename(path)) for line in f]
        random.Random(seed).shuffle(rs)
        rows += rs[:int(k or 0) or (n if per_file else 0) or None]
    random.Random(seed).shuffle(rows)
    return rows if per_file or not n else rows[:n]


def encode(agent, rows):
    items = []
    for r in rows:
        it = agent._encode_state(r["state"], [r["qid"]], {r["qid"]: agent._to_internal(r["question"])})[0]
        it["target"] = r["target"]
        items.append([it])
    return collate_items(items, agent.tok.pad_token_id)


def forward(agent, b):
    dev = agent.device
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = agent.model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                                b["marker_mask"].to(dev), b["qtype"].to(dev))
    return logits.float().masked_fill(~b["marker_mask"].to(dev), -1e4)


@torch.no_grad()
def evaluate(agent, rows, bs=64):
    """Mean soft cross-entropy, accuracy vs the solver label (argmax, as --greedy plays), raw logits kept for temps."""
    agent.model.eval()
    loss = hits = 0.0
    kept, by = [], {}
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        b = encode(agent, chunk)
        logits = forward(agent, b)
        t = b["target"].to(logits.device)
        loss += -(t * torch.log_softmax(logits, -1)).sum(-1).sum().item()
        ok = (logits.argmax(-1) == t.argmax(-1)).tolist()
        hits += sum(ok)
        for r, o in zip(chunk, ok):
            for k in {r.get("street") or r["state"]["street"], r["_file"]}:
                by.setdefault(k, []).append(o)
        kept += [(lg[:len(r["target"])].cpu(), r["target"].index(max(r["target"]))) for lg, r in zip(logits, chunk)]
    agent.model.train()
    return loss / len(rows), hits / len(rows), kept, " | ".join(
        "%s %.1f%% (%d)" % (k, 100 * sum(v) / len(v), len(v)) for k, v in sorted(by.items()))


def fit_temperature(kept):
    """One temperature per option-count bucket, by grid search on val NLL (laya clamps to [0.5, 5])."""
    buckets = {}
    for lg, y in kept:
        buckets.setdefault(temp_bucket(0, len(lg)), []).append((lg, y))
    out = {}
    for name, xs in buckets.items():
        grid = [0.5 + 0.05 * i for i in range(91)]
        out[name] = min(grid, key=lambda T: sum(-torch.log_softmax(lg / T, -1)[y].item() for lg, y in xs))
    return out


def save(agent, base_dir, out, temps, info):
    os.makedirs(out, exist_ok=True)
    for name in ("tokenizer", "encoder"):
        shutil.copytree(os.path.join(base_dir, name), os.path.join(out, name), dirs_exist_ok=True)
    save_file({k: v.detach().to(torch.float16).contiguous().cpu() for k, v in agent.model.state_dict().items()},
              os.path.join(out, "model.safetensors"))
    with open(os.path.join(base_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    if temps:
        cfg["temperature_by_options"] = {**cfg.get("temperature_by_options", {}), **temps}
        cfg["temperature"][0] = temps.get("choice:3-5", cfg["temperature"][0])
    cfg["training"] = {**cfg.get("training", {}), "poker_finetune": info}
    with open(os.path.join(out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="jsonl, or comma-separated files each with optional :N rows")
    ap.add_argument("--val", required=True, help="same format; --val-n applies per file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="convaiinnovations/laya", help="HF id or local checkpoint dir (warm start)")
    ap.add_argument("--n", type=int, default=0, help="train rows to use, 0 = all")
    ap.add_argument("--val-n", type=int, default=2000, help="val rows per file")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5, help="encoder layers")
    ap.add_argument("--head-lr", type=float, default=1e-4, help="decision head and scorer")
    ap.add_argument("--train-layers", type=int, default=8, help="top encoder layers to train (of 28)")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--overfit", type=int, default=0, help="train on this many rows only and check loss -> ~0")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    base_dir = args.base if os.path.isdir(args.base) else snapshot_download(args.base)
    agent = laya.load(base_dir, device="cuda")
    model = agent.model
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    layers = model.encoder.layers
    enc = [p for m in list(layers[len(layers) - args.train_layers:]) + [model.encoder.final_norm] for p in m.parameters()]
    head = [p for m in (model.head, model.scorer, model.type_emb) for p in m.parameters()]
    for p in enc + head:
        p.requires_grad = True
    print("trainable %.0fM of %.0fM params" % (sum(p.numel() for p in enc + head) / 1e6,
                                             sum(p.numel() for p in model.parameters()) / 1e6))

    train = load_rows(args.train, args.overfit or args.n, args.seed)
    val = train if args.overfit else load_rows(args.val, args.val_n, args.seed, per_file=True)
    print("train %d rows, val %d rows" % (len(train), len(val)))
    steps = max(1, int(len(train) * args.epochs / args.bs)) if not args.overfit else 300
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr}, {"params": head, "lr": args.head_lr}], weight_decay=0.01)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    loss0, acc0, _, by0 = evaluate(agent, val)
    print("step 0 (zero-shot): val loss %.3f, solver-match accuracy %.1f%% [%s]" % (loss0, 100 * acc0, by0), flush=True)
    best, t0, run = (loss0, acc0, 0), time.time(), []
    order = []
    for step in range(1, steps + 1):
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)
        batch = [train[order.pop()] for _ in range(min(args.bs, len(order)))]
        b = encode(agent, batch)
        logits = forward(agent, b)
        loss = -(b["target"].to(logits.device) * torch.log_softmax(logits, -1)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc + head, 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % 50 == 0:
            el = time.time() - t0
            print("step %d/%d loss %.3f | %.2f steps/s | eta %.0f min | gpu %.1f GB" % (
                step, steps, sum(run[-50:]) / len(run[-50:]), step / el, (steps - step) / (step / el) / 60,
                torch.cuda.max_memory_allocated() / 1e9), flush=True)
        if step % args.eval_every == 0 or step == steps:
            vl, va, kept, by = evaluate(agent, val)
            print("  val loss %.3f, solver-match accuracy %.1f%% (zero-shot %.1f%%) [%s]" % (vl, 100 * va, 100 * acc0, by),
                  flush=True)
            if vl < best[0] and not args.overfit:
                best = (vl, va, step)
                temps = fit_temperature(kept)
                save(agent, base_dir, args.out, temps, {
                    "base": args.base, "train": args.train, "rows": len(train), "val_by": by, "step": step, "val_loss": round(vl, 4),
                    "val_accuracy": round(va, 4), "zero_shot_accuracy": round(acc0, 4), "temperatures": temps,
                    "train_layers": args.train_layers, "lr": args.lr, "head_lr": args.head_lr, "bs": args.bs})
                print("  saved best -> %s (temperatures %s)" % (args.out, temps), flush=True)
    if args.overfit:
        assert sum(run[-20:]) / 20 < 0.5, "overfit check failed: loss %.3f did not drop" % (sum(run[-20:]) / 20)
        vl, va, kept, _ = evaluate(agent, val)   # save -> reload round trip: the reloaded dir must score the same
        save(agent, base_dir, args.out, fit_temperature(kept), {"overfit": args.overfit})
        del agent, model, opt, sched, enc, head
        torch.cuda.empty_cache()
        rl, ra, _, _ = evaluate(laya.load(args.out, device="cuda"), val)
        print("reloaded %s: val loss %.3f vs %.3f before save, accuracy %.1f%% vs %.1f%%" % (args.out, rl, vl, 100 * ra, 100 * va))
        assert abs(rl - vl) < 0.05 and abs(ra - va) < 0.05, "save/reload mismatch"
        print("overfit check ok")
    else:
        print("done. best val loss %.3f, accuracy %.1f%% at step %d (zero-shot %.1f%%)"
              % (best[0], 100 * best[1], best[2], 100 * acc0))


if __name__ == "__main__":
    main()
