# Plan: fine-tune Laya to play heads-up poker

> **Status (2026-09-28): first run in progress via a shortcut.** Instead of Phase 1's own rollout labels, round 1 uses
> PokerBench postflop solver labels (`convert_pokerbench.py`, 177k rows) and trains with `finetune.py`
> (`checkpoints/flat.log`, then `checkpoints/tree.log`). Postflop only: preflop decisions are untrained.
> Phase 1 rollout labels are still the path to heads-up-specific play.
>
> **Original status: not started.** A handoff plan: a fresh session should be able to build this without the earlier
> conversation. Work through the phases in order and run each phase's check. The same rules as PLAN.md apply:
> minimal code, stdlib + what's installed, `# ponytail:` on shortcuts, one assert-based check.

## Why fine-tune (measured 2026-09-28, seed 1)

- Zero-shot Laya doesn't read the numbers. On the quiz, the answer to "fold or play" is ~50% whatever the
  spot, with a correlation of -0.06 to (equity - pot odds), even though the question states both numbers.
- The split decision (`tree`) and extra inputs (`rich`) don't help zero-shot. Quiz regret rose 0.70 ± 0.48 bb.
- `laya:range=1,guard=1` beats pot_odds by +243 ± 69 bb/100 over 2,000 hands, but the **guard** does that
  work: it changes about 40% of Laya's moves (`guard_stats.py`). The goal is a model that doesn't need the guard.

## The idea in one paragraph

Generate many poker situations. For each one, compute **how much every legal action is worth**, in bb, by
simulating the rest of the hand. Turn those values into a target probability per action. Fine-tune Laya's
existing choice head so its probabilities match the targets. That is supervised learning of "the action that
wins the most chips", a one-step policy improvement over the policy used to simulate. The same action values
also give labels for the three split questions (fold/play, call/raise, size), so **both** question formats
are trained from one dataset and compared fairly.

## What the laya package gives us (verified in laya 0.3.20 source)

- There's no training script, but the pieces are public. `agent = laya.load(...)`, then:
  - `agent.model` is a `torch.nn.Module` (`laya/common.py: DecisionModel`). Its
    `forward(input_ids, attention_mask, marker_pos, marker_mask, qtype)` returns `(logits, act_logits)`.
    `logits[i, j]` is the score of option j. **Train only `logits`; ignore `act_logits`.**
  - `agent._to_internal(qdef)` and `agent._encode_state(state, ids, internal)` build the exact
    token rows `predict` uses. `laya.common.collate_items(batch, pad_id)` pads them and accepts a per-row
    `"target"` list (one soft probability per option). Use these so training input == play input.
- A checkpoint is a folder: `rl_agent_config.json`, `model.safetensors`, `tokenizer/`, `encoder/`.
  Save = copy the base folder from the HF cache and overwrite `model.safetensors`
  (`safetensors.torch.save_file(model.state_dict())`). `--laya-model <dir>` already loads it in sim.py and quiz.py.
- **Temperature gotcha:** at inference, logits are divided by a per-bucket temperature from the config
  (`temperature_by_options`: 1.76 for 3-5 options, 1.0 for 6-10; `temperature[0]` 1.64 for choice).
  After training on raw logits, refit it (Phase 3). Otherwise sampled play is too flat or too sharp.
- Size: ModernBERT-large encoder, 28 layers, d=1024 (~395M), plus a 2-layer head (~25M). Full fine-tuning with
  AdamW needs about 16 bytes/param ≈ 6.7 GB before activations, which won't fit an 8 GB RTX 4060. **Freeze the
  embeddings and the bottom 20 encoder layers. Train the top 8 layers, the head and the scorer** (~125M
  trainable), with bf16 autocast and a small batch plus gradient accumulation.

## Phase 0: measure before building (~30 min)

- Time one training step (batch 8, freeze as above) and peak GPU memory (`torch.cuda.max_memory_allocated`).
  If memory is over ~7 GB, freeze more layers or call `agent.model.encoder.gradient_checkpointing_enable()`.
- Time labelling one situation (Phase 1) with 64 rollouts per action. `strengths(board)` costs ~9 ms per new
  board, and that dominates the rollouts.
- Pick the dataset size from these two numbers. Aim: labels in **< 1 night on 24 cores**, training in **< 3 h**.
  Starting point: **30k situations**.

## Phase 1: generate labelled situations, `gen_data.py` (new file)

**Where situations come from.** Reuse `quiz.py`: `Snapper` (hero = epsilon-random pot-odds, for varied spots),
`consistent_hands`, `RefVillain`. Changes:

1. **Play sessions, not single hands.** Keep one `OppStats` alive across ~100 hands per session. That way the
   `villain` stats line appears in the state as it does in real play (it only shows after 20 hands). If it
   doesn't, the model never trains on inputs it will see.
2. **Villain pool:** in each session villain is one of pot_odds (RefVillain), calling_station, or a loose one
   (pot-odds rule with thresholds shifted and a 15% random bet). Training against RefVillain alone would learn
   to exploit one bot. `# ponytail: 3 hand-made styles; add self-play Laya as a villain in round 2.`
3. Snapshot street mix like the quiz, plus extra turn/river spots, which the quiz under-samples (21 river of 200).
4. **Split by seed:** train / val / test come from disjoint deal seeds. Keep `quiz_1_200.json` untouched as a
   final test.

**Action values (the labels).** For each snapshot and each legal action, estimate EV in bb:

- v0, cheap, exists today: `quiz.option_evs` (check-down after hero's action). It's biased against betting
  and raising. Use it only to get the pipeline working end to end.
- **v1, the real labels: rollouts.** Sample villain's hand from `consistent_hands`, sample the rest of the
  board, force hero's action, then **play the hand out with the engine**. Hero continues with the pot-odds
  rule (hand-rank as equity, like RefVillain), and villain continues with its session style, so re-raises
  and later-street bets count. Do this by building a deck (`deck[0:2]` seat 0, `[2:4]` seat 1, `[4:9]` board)
  and running `play_hand` with scripted agents that replay `obs["acts"]` up to the snapshot, then take the
  forced action, then follow the policies. Use the same samples (common random numbers) for every action,
  so the differences between actions are low-noise. Start with 64 samples.
- `# ponytail: the continuation is a rule bot, so labels = "best action if everyone plays simply afterwards".`
  Upgrade path: round 2 uses the fine-tuned Laya as the continuation policy (policy iteration).

**Targets.**
- Flat question: `target = softmax(EV_bb / tau)` over legal actions, with `tau = 0.5` bb. Soft targets keep
  near-ties as mixed play and absorb Monte Carlo noise. One-hot argmax is the fallback if play gets too random.
- Tree questions, from the same EVs: `play` = softmax over (fold = 0, play = max of the rest);
  `aggr` = (call or check EV, best raise EV); `size` = softmax over the raise sizes.
  Only emit the questions `tree_questions` would ask.

**Output.** JSONL, one row per (situation, question). The fields are `state` (from `agents.laya_state`, rich +
range, exactly as in play), `question` (the dict passed to `predict`), `target` (list in option order),
`evs`, `street`, `seed`. Build `state` and `question` with the **same functions the agent uses**
(`laya_state`, `laya_criteria`, `tree_questions`). Never copy them.

**Check:** assert that for 20 random rows the `state`/`question` equal what `LayaAgent.dist` builds for that obs,
and that on a river spot with the nuts, `EV(fold) < EV(call)`.

## Phase 2: fine-tune, `finetune.py` (new file)

- Load the base with `laya.load(base, device="cuda")`. Freeze as in "What the laya package gives us". AdamW,
  lr 2e-5 for the encoder layers and 1e-4 for the head/scorer, 5% warmup, cosine decay, weight decay 0.01,
  bf16 autocast, grad clip 1.0.
- Batch: encode rows with `_encode_state`, attach `"target"`, then `collate_items`. Loss = soft cross-entropy
  `-(target * log_softmax(logits.masked_fill(~marker_mask, -1e4))).sum(-1).mean()`.
- Train **two models** from the same data: `flat` (one action question) and `tree` (three questions).
  1-2 epochs. Evaluate val loss and val regret every ~500 steps and keep the best.
- **Refit temperature** on val: grid search one scalar T per option-count bucket to minimize val NLL. Write it
  into `rl_agent_config.json` (`temperature`, `temperature_by_options`). Values must be in [0.5, 5]
  or laya clamps them.
- Save to `checkpoints/laya-poker-{flat,tree}/`. Record the data size, steps and val metrics under the
  config's `training` key.
- **Check:** overfit 64 rows to near-zero loss in a few hundred steps before a full run. That proves the
  loss, masking and targets are wired right.

## Phase 3: evaluate (existing tools, no new code)

Run each with `--laya-model checkpoints/laya-poker-flat` (and `-tree`):

1. `quiz.py --n 200`: regret and near-best vs zero-shot Laya and pot_odds. Also the held-out test split.
2. `sim.py --deals 1000` (2,000 hands), **seeds 1 and 2**, guard **off**:
   vs pot_odds, vs calling_station, vs `laya:range=1,guard=1` (the current best).
3. `guard_stats.py` on a `--log` run with guard **on**: how often the guard still changes a move.
4. Diagnostics from the zero-shot investigation. The fold/play answer should now correlate strongly with
   equity - pot odds, and the sizes should be used, not just third_pot/allin.

**Done when:** the fine-tuned model with **guard off** beats `laya:range=1,guard=1` head to head, with the CI
excluding 0, on both seeds, and the guard changes < 5% of its moves.

## Phase 4 (optional, only if Phase 3 passes)

- **Round 2 / policy iteration:** relabel with the fine-tuned model as the continuation policy and as a villain,
  then train again. Stop when a round no longer beats the previous one.
- **MLP control:** a small torch MLP on `features()` numbers, trained on the same targets. If it plays as
  well as Laya, Laya's value here is its flexibility (words, new inputs), not strength. Worth knowing.

## Risks

| Risk | Sign | Response |
|---|---|---|
| Labels only as good as the rollout policy | Model plays like pot_odds and doesn't beat it by much | Phase 4 round 2 |
| Exploits the 3 training villains only | Wins vs pot_odds, loses vs the zero-shot + guard bot | More villain styles, self-play villain |
| Out of GPU memory | OOM in Phase 0 | Freeze more layers, gradient checkpointing, batch 4 + accumulation |
| Labelling too slow | Phase 0 estimate > 1 night | Fewer samples (32), fewer situations, cache `strengths` across rollouts |
| Mis-set temperature | Sampled play too random or too greedy vs val | Phase 2 refit; compare `--greedy` |
| Too slow locally | Training > 3 h | The Kaggle 2xT4 notebook from the original brief (`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`); T4 has no bf16, use fp16 |
