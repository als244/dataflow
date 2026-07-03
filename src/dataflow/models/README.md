# dataflow.models — golden references

**Purpose.** One hand-written, plain-autograd model per family: the
independent witness every runtime result is checked against. A golden model
catches composition-graph mistakes (wrong residual wiring, missed final
norm, optimizer-state drift) that per-op and per-block checks cannot.

## Contract

- **Compose from op references.** The forward uses the same *reference*
  forms as `tasks/ops.py` (bf16 storage, fp32 reductions) so ladder-3
  tolerances stay at bf16-honest levels (2–3e-2 rel L2) instead of
  absorbing numerics drift.
- **Replicate the optimizer exactly**, including bf16 state round-trips and
  the step-dependent bias correction — final `W` and `O` are compared
  buffer-to-buffer after N steps, not just losses.
- **Share the packed layouts.** `from_packed_bytes(dims, n_layers, ...)`
  constructors consume the exact pinned bytes `initial_values()` produced,
  so golden and runtime start from identical weights, and final state
  compares without any re-packing step.
- Keep it boring: eager torch, no fused kernels, no custom autograd — its
  entire value is being obviously correct.

## Files

- `llama3_reference.py` — `GoldenLlama3` (`from_packed_bytes`,
  `train_step(tokens, targets) -> loss`).

Adding a family: see `docs/extending.md` §3 and §6.
