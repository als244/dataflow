# RL post-training on the dataflow engine — a custom Program, for real

The worked example behind `docs/extending_programs.md`: an RL trainer
that **never runs a forward pass**. An inference engine (simulated here)
already generated the rollout and saved, per layer, the activation
checkpoint (block input), the MoE routing pack, and the sparse-attention
index selection. Training consumes reward-derived advantages and starts
straight at the head, recomputing each layer's context from its
checkpoint on the way down.

```bash
python examples/rl_trainer/run.py                 # PPO-clip, 3 steps
python examples/rl_trainer/run.py --loss reinforce
python examples/rl_trainer/run.py --device-gib 2  # PressureFit budget
```

Output ends with a parity verdict: after N optimizer steps, EVERY
parameter field of the engine run matches an **isolated plain-autograd
trainer** run on the same artifacts (rel_l2 ≤ 3e-2 per field, ~1e-3
typical; per-step losses match to ~1e-5).

## Why this example earns its keep

- **The metadata grammar fits RL natively.** The saved routing and
  index selections arrive as the runtime's `M` objects; with
  `train_indexer=False` they are consumed verbatim — the backward sees
  the inference engine's EXACT choices, never a re-derived
  approximation, by construction rather than by care.
- **Almost nothing is new.** Recompute-from-checkpoint and every block
  backward/optimizer are the builtin executables, bound through the
  family resolver via `compute_block_key`. The new surface is one loss
  executable (`rl_ops.RLHeadLoss`, mirroring the builtin `HeadLoss`
  chunking) and a ~120-line program builder.
- **The builder is SURGERY, not authorship** (`program_builder.py`):
  lower the standard glm52 chain, delete the forward tasks, flip the
  per-layer inputs + M objects to initial (inference-supplied) objects,
  add explicit `block_recompute` tasks, swap the CE head for the RL
  head. All naming, dW wiring, optimizer interleaving, and
  final-location conventions are inherited from the standard lowering.

## Files

| file | role |
|---|---|
| `fake_inference.py` | rollout forward (capture-mode golden); saves checkpoints, M payloads in the exact runtime layouts, actions/old-logprobs/advantages, starting weights |
| `program_builder.py` | the custom Program (surgery on `lower_glm52`) |
| `rl_ops.py` | PPO-clip / REINFORCE head-loss executable + the where-form loss contract shared with the reference; math pinned vs autograd standalone |
| `reference_trainer.py` | the isolated witness: plain autograd, per-layer VJPs from fixed checkpoints, golden AdamW replica |
| `run.py` | build → PressureFit → `Engine.execute` → parity table |

## The one semantic subtlety (worth reading twice)

After step 1 the weights change, and there are two *different* trainers
one could mean:

1. **Frozen-rollout training (this example, and the spec):** each layer
   forwards from its FIXED checkpointed input with CURRENT weights; the
   head reads the FIXED last-block output. Layer-local Jacobians at
   `(x_ckpt_i, W_current)`, chained by explicit VJPs.
2. **Re-forward each epoch:** activations ripple with the updated
   weights (what a naive autograd reference does — and what PPO
   implementations that re-run the policy per epoch compute).

They agree exactly at step 1 and diverge after. The engine program
implements (1) by construction — recompute reads the fixed checkpoint.
Our first reference implemented (2) and the per-step losses split
immediately; the committed reference implements (1) with explicit
per-layer VJPs and matches to ~1e-5. If you want semantics (2), refresh
the checkpoints between steps (re-run `fake_inference` with the updated
weights — i.e., an actual inference pass, which is the point of that
design).

## Notes

- `train_indexer=False` is load-bearing: no KL objective, no `dM`
  accumulators, indexer weights bit-frozen — selection is data, not a
  trainable path. (Enabled for glm52 as part of this example.)
- The parity gate runs in CI: `tests/examples/test_rl_trainer.py`.
- Scale knobs: the example runs glm52-tiny for speed; the builder takes
  any glm52 config (`build_rl_program(cfg, steps=N)`), and PressureFit
  handles streaming when the model outgrows `--device-gib`.
