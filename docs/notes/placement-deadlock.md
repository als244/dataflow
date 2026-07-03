# Note: the placement lifetime-inversion deadlock (found 2026-07-02, fixed 2026-07-03)

## The incident

First-ever runtime deadlock, seq-1K sweep, `8b-s1k-bs2ga32 @ 20 GiB`:

```
DeadlockError: task 'block_recompute_0_3_16' waiting to reserve 176439296
fast bytes (used=20030521344, cap=21474836480); no in-flight work can
unblock it (from_slow queue=[], to_slow queue=[], deferred=[])
```

The telltale: the **ledger had room** (20.03 + 0.18 < 21.47 GB) and **all
transfer queues were empty**. The task was blocked purely by the static
placement's per-offset admission (`pool.can_get == False`): its assigned
offset range was still occupied by a live buffer — and nothing in flight
could ever free it.

## Root cause: cross-tag lifetime inversion (NOT key mismatch)

First hypothesis — that instance keys `(object_id, incarnation)` get
assigned in pool-arrival order and real timing reorders arrivals versus the
dry run ("miskeying") — is **provably wrong**: for any single object id,
gets are totally ordered by the chain (task-driven gets are serialized by
strict pacing; per-object prefetches are serialized by the FIFO
one-in-flight transfer engine; a produce following a prefetched copy cannot
dispatch before that copy's consumer, which cannot run before the prefetch
started). Per-tag key sequences are identical in the dry run and any real
run.

The real mechanism is **cross-tag**: the packer overlaps two instances
whose *dry-run* lifetimes are disjoint (P dies at ordinal k, Q born at
k+1). Real durations differ from simulated ones, and an h2d prefetch can
legally start much earlier relative to other tags than it did in the dry
run (its only preconditions are its anchor event, ledger room, and a free
offset). So under real timing:

1. Prefetch R (for an object consumed by a *later* task) starts early and
   takes its assigned offset — which the packer overlapped with Q's,
   because in the dry run R was born only after Q died.
2. Task T_Q arrives at its output reservation for Q. Its offset range
   overlaps live R. `can_get` says wait.
3. R's release depends on its consumer — a task *after* T_Q in the chain.
   The dispatcher can't pass T_Q. **Cycle.** Queues drain, tokens stop,
   quiescence.

The "progress guaranteed" argument in the original placement design ("the
blocker is always an in-flight transfer or an earlier task") is exactly
wrong for early-started prefetches of later-consumed objects. The
condition became reachable at ga=32 because 2,048 context incarnations per
step make dry-vs-real interleaving divergence essentially certain.

## The fix: quiescent escape valve

`pool.get_escaped(location, size, tag)` + engine/transfer hooks:

- The engine already detects true quiescence (`next_completion() is None`
  while blocked). At that moment — and only then — if the ledger admits
  the bytes and the block is purely placed-offset busyness, the blocked
  instance is served from a **dynamic allocation** instead of its assigned
  offset. Applied at both admission points: task output reservation
  (dispatcher) and the h2d queue head (prefetch destination).
- The incarnation counter still advances on an escaped get, so every LATER
  instance of that tag keeps its recorded key and offset.
- Each escape increments `pool.placement_escapes`, emits a
  `placement_escape` trace event, and is reported end-to-end
  (`RunResult.placement_escapes` → `TrainReport` → sweep summary rows).
- Genuine capacity deadlocks (ledger cannot admit) still raise
  `DeadlockError` loudly, as before.

Properties: **deadlock-freedom by construction** under placement (any
pure-placement quiescent block converts to forward progress); correctness
unaffected (any free VRAM is as good as the assigned offset — poison and
parity suites unchanged); the zero-escape steady state remains the norm
and is visible per run (a nonzero count is a signal the packing is
timing-fragile, not an error).

Regression test:
`tests/runtime/test_placement.py::test_quiescent_lifetime_inversion_escapes_instead_of_deadlocking`
distills the incident to a 3-task chain with an adversarial hand-built
`Placement` (two concurrently-live instances share an offset): without the
valve the engine deadlocks; with it the run completes with
`placement_escapes == 1`.

## Alternatives considered

- **Deterministic plan-position keying**: solves a problem we don't have
  (keys are already order-invariant per tag; see above).
- **Order-gating h2d starts to dry-run birth order**: restores the packing
  invariant but serializes transfer admission across tags — gives back the
  latency-adaptivity that makes real runs beat the plan, and adds global
  bookkeeping. Rejected for cost.
- **Temporal padding in the packer** (treat near-adjacent dry-run lifetimes
  as overlapping): reduces incidence, no guarantee, pays extent on every
  run. Possible future knob if escape counts are ever routinely nonzero.
- **VMM chunk-backing** (M5): removes fixed offsets entirely, and with
  them both the geometry tax and this whole class of inversion — the real
  long-term fix; the escape valve is the guarantee until then.
