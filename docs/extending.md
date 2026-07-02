# Extending: custom ops, blocks, and models

The walkthrough we ourselves follow when adding builtin families. The chain
of custody: **op → block executable → lowering → planned program → runtime**,
with a gradcheck gate at each level.

## 1. Write an op (`dataflow/tasks/ops.py`)

An op is two functions plus cost knowledge:

- **launch form** — eager torch (or Triton/custom kernel) writing into
  provided tensors where possible. It runs inside the executable's
  `torch.cuda.stream(external_stream(ctx.stream))` scope; never synchronize,
  never allocate runtime-owned memory (op-internal torch scratch is fine —
  it gets measured).
- **reference form** — pure, autograd-able torch. This is what gradcheck
  differentiates and what golden models are composed from; keep the same
  numerics discipline (bf16 storage, fp32 reductions) so tolerances stay
  tight.

Gate it with a level-1 test (see `tests/tasks/test_llama3_math.py`): launch
outputs vs reference forward; hand-written backward vs autograd on the
reference; use `rel_l2` with bf16-honest tolerances (2-3e-2).

## 2. Compose a block (`dataflow/tasks/<family>_blocks.py`)

A block executable resolves its buffers positionally from the task's
input/output order (document the convention next to the class), views them
through `PackedLayout`s, and calls op launch forms. Rules:

- Everything backward needs beyond (inputs, weights) goes into the saved
  context layout — and the recompute executable must reproduce that context
  bit-comparably from the same inputs.
- Accumulating variants (grad accumulation) must ADD exactly when the task
  mutates an existing gradient object and WRITE when it creates one — the
  engine exposes which via `ctx.task.mutates`.
- Sizes come from layouts; never hand-compute bytes anywhere else.

Gate: `check_block_backward(dims)` — verifies dx + every packed dW field vs
autograd, recompute-equivalence, and 2x-accumulation.

## 3. Define the model + golden reference (`dataflow/models/`)

The golden reference composes the ops' *reference* forms with plain autograd
and replicates the optimizer update exactly (including bf16 state
round-trips). It must share the packed-weight layouts so state is comparable
buffer-to-buffer — see `llama3_reference.py`.

## 4. Lower it (`dataflow/training/`)

Lowering emits the task chain (see `shaped_llama3.py` for the structural
conventions: naming, grad-accum mutation pattern, recompute tasks +
`RecomputeRewrite`s, optimizer `step` params, multi-step-invariant
`final_locations`) with sizes taken from the layouts, plus an executable
resolver keyed by `compute_block_key` (never task id — planner-inserted
recompute tasks must bind automatically).

## 5. Measure, plan, run, verify

```python
profiles = profile_program(program, resolver, backend)   # runtimes + workspace
planned  = plan_program(apply_measured_costs(program, profiles),
                        fast_memory_capacity=cap, recompute=..., build_variant=...)
```

Gate: `check_model_step` at a few budgets (plan-invariance!), plus the
poison-on-free and interleaving-stress runs from
`tests/tasks/test_m3_gate.py`. A model that passes those is ready for
throughput work (budget sweeps vs the simulator — see tools/m2_gate.py's
replay-fidelity metric).
