# RL post-training as a custom dataflow Program

Five parallel worked examples — one per model family — of the same
scenario, each verified against an isolated plain-autograd trainer.
This is the executable companion to `docs/extending_programs.md`.

## Scenario

An inference engine generated a rollout and saved, per layer:

- the **activation checkpoint** — the block's INPUT hidden state;
- any **discrete state** the layer produced: MoE routing decisions
  and/or sparse-attention index selections (the runtime's `M` objects).

The trainer receives reward-derived per-token advantages, the sampled
actions, and the behavior policy's logprobs. It **never runs a forward
pass**: it starts at the head (an RL objective replaces cross-entropy),
walks the layers in reverse, recomputes each layer's float context from
its checkpoint with CURRENT weights, and backprops — with the saved
routing/selections consumed verbatim, so the backward sees the
inference engine's exact discrete choices. PressureFit schedules the
streaming of weights, checkpoints, and optimizer state under any device
budget, which is the point of running RL post-training on a
small-memory box.

## Run

```bash
python examples/rl_training/<family>/run.py \
    [--loss ppo|reinforce] [--steps 3] [--device-gib 2.0] [--out-dir D]
```

Families: `llama3`, `qwen35`, `qwen3moe`, `dsv32`, `glm52`. Each run
leaves three inspectable intermediates in the family directory:

| file | what it is |
|---|---|
| `program.json` | the bare custom Program (upload to the webapp simulator, or `load_program` it) — READ THIS to see what a dataflow program is |
| `plan.json` | the same program after PressureFit annotation (offload/prefetch directives per task) — diff against program.json to see exactly what planning adds |
| `rollout.pt` | the saved inference artifacts (regenerated if deleted; gitignored) |

The run ends with the parity verdict: per-step losses and every
parameter field vs the isolated autograd reference (`PASS: engine ==
isolated autograd`). CI runs all five: `tests/examples/test_rl_training.py`.

## What each family demonstrates

| family | discrete state consumed from inference | notes |
|---|---|---|
| `llama3` | none | the minimal weave: checkpoints + recompute only |
| `qwen35` | none | heterogeneous layer kinds (DeltaNet + gated attention) through the same generic builder |
| `qwen3moe` | MoE routing pack | routing pinned via the golden's `route_ids` seam |
| `dsv32` | DSA index selection + routing | per-layer selections; `train_indexer=False` — selection is data, not a trainable path |
| `glm52` | shared selections (IndexShare) + routing | leader M consumed by follower layers; the richest metadata case |

## What reaches the engine (the whole handoff)

`harness.run()` ends in exactly one call — everything before it is
preparation you can inspect:

```python
result = Engine(backend).execute(
    planned.program,      # plan.json: the PressureFit-ANNOTATED program
                          # (program.json + offload/prefetch directives)
    resolver=resolver,    # task -> executable: the family resolver for
                          # recompute/bwd/optimizer tasks + the one
                          # custom RLHeadLoss for compute key "rl_head_loss"
    initial_buffers=values,   # pinned host buffers, one per initial
                          # object: weights + zeroed optimizer state,
                          # the per-layer checkpoints (y_*), the M
                          # payloads, tokens/actions/logprobs/advantages
    pool_prewarm=dry.pool_demand,  # from a FakeBackend dry run
)
```

Losses and final weights are read back from `result.objects[...]`.
Nothing else is passed; there is no hidden model object — the program,
the resolver, and the buffers ARE the model.

## How it works (shared machinery)

- `builder.py` — family-GENERIC surgery on the standard lowering:
  walk the standard chain, drop forward tasks, insert
  `block_recompute` tasks derived from each dropped forward (same
  inputs + the layer's M), swap the CE head for `rl_head_loss`, flip
  the now-unproduced checkpoints/M into initial objects. No family
  names, kinds, or optimizer ids appear — they are read off the
  standard program.
- `rl_ops.py` — the one custom executable: fused final-norm + head +
  PPO-clip/REINFORCE loss + head backward, chunked like the builtin
  `HeadLoss` (no (tokens, vocab) materialization). The loss uses an
  explicit `where`-form so the clip-branch derivative is unambiguous;
  the same function is the reference's autograd objective.
- `harness.py` — artifacts, engine run, the isolated reference, parity.
- `<family>/adapter.py` — the only per-family code: how to capture and
  pin that family's discrete state (~40-110 lines; llama3's is 60
  lines of mostly pass-through).

## The semantic subtlety (read twice)

After step 1 the weights have changed, and "train on this rollout"
forks into two DIFFERENT trainers:

1. **Frozen-rollout** (these examples, and the scenario's meaning):
   each layer forwards from its FIXED checkpointed input with CURRENT
   weights; the head reads the FIXED last-block output. Layer-local
   Jacobians, chained by explicit VJPs.
2. **Re-forward each epoch**: updated weights ripple through
   activations (what a naive autograd reference computes — and a
   LEGITIMATE trainer, just a different one, requiring an actual
   forward pass each step).

They agree exactly at step 1 and diverge after. The engine program
implements (1) BY CONSTRUCTION — recompute reads the checkpoint. The
reference implements (1) with explicit per-layer VJPs; our first draft
accidentally implemented (2) and the parity gate caught it at step 2.
If you want (2), refresh the rollout between steps — that is an
inference pass, which is exactly what this design avoids.

## Known tolerance envelopes

Two sign-lottery parameter classes compare under absolute envelopes
instead of rel_l2 (`harness.BIAS_ATOL`): `w_router_bias` (noaux count
race) and qwen35's `dt_bias` (sub-noise gradients; a one-step AdamW
update is ±lr times a coin flip on both sides). Everything else holds
rel_l2 ≤ 3e-2, typically ~1e-3.
