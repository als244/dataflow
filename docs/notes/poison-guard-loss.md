# Post-mortem: the poison-gate NaN flake (guard loss on range recycling)

**Symptom.** `test_poison_on_free_changes_nothing` intermittently failed
with `W_0: rel_l2=nan` in the POISONED run — NaN bytes in the final
layer-0 weight — at ~10–18% per iteration when the three GPU test files
ran in one process (fresh process per iteration), essentially never in
isolation. Loss always matched, so the corruption entered after round 0,
in the gradient/optimizer/writeback tail. Confirmed pre-existing (4/40
at 6203a8d, before the M4.9 interleave/valve work).

**Mechanism.** Poison-on-free memsets a freed fast buffer to 0xFF on the
COMPUTE stream and records `buffer.guard_event` there. The safety
contract has two legs: kernels reuse buffers on the same compute stream
(stream-ordered after the memset), and transfers into a reused buffer
wait `guard_event` first. Both legs attach to the **Buffer object**. Two
paths discarded the object while its **address range** lived on:

1. **Fragmentation flush** (`pool._carve`): on a carve miss, free-listed
   slab buffers are returned to the slab and the coalesced range is
   re-carved as a NEW `Buffer` — `guard_event=None`. If the discarded
   buffer's poison memset was still queued behind a busy compute stream,
   the new owner's h2d fill raced it, and the straggler could land 0xFF
   on top of freshly-copied live data. Deterministic repro
   (`test_slab_flush_preserves_pending_poison_guard`): a 100 ms spin
   before the memset, then flush + re-carve + unguarded write —
   1,048,576 of 1,048,576 bytes corrupted on the broken code.
2. **Placed-offset reuse** (`pool.put`/`get` in placement mode): placed
   buffers are identity-managed — `put()` drops the object, the next
   incarnation at the same offset gets a fresh Buffer with no guard.
   Same race, production-placement shape (only reachable with the
   poison debug mode on).

Why the weird reproduction profile: the race needs the compute stream to
lag the host far enough that a poison memset is still pending when a
flush + reuse happens — cold-process dispatch overhead (first-call
lazy-init, Triton compiles) with warm GPU clocks stretches that window;
a warm process closes it. Why `W_0`: backward runs 31→0, so layer-0
buffers free last, exactly amid the tail's churn where 8 MiB budgets
force carve misses; the straggler that hits `W_0`'s (or `dW_0`'s) buffer
poisons the bytes the writeback then persists. Why production was never
at risk: `guard_event` is produced ONLY by the poison debug mode, and
real runs use static placement (no fragmentation flush) — but the debug
mode's verdicts must be trustworthy, which is the point of fixing it.

**Fix.**
- Flush skips buffers whose guard is still pending (new backend
  `event_complete(event)` — non-blocking cudaEventQuery / fake-clock
  check); they stay in the free list, where reuse preserves the object
  and the guard. They flush on a later miss once the memset lands.
- Placed mode stores pending guards by ADDRESS RANGE at `put()` and
  re-attaches them (tuple if several overlap) to the next incarnation
  carved over the range; transfers honor tuple guards.

**Validation.** Deterministic repro test red→green; placed-carryover
unit test; full suites 75 CPU + 41 GPU green; the original grouped-run
flake loop: **0 failures in 80 iterations** post-fix vs 11/60 pre-fix
(P ≈ 1e-7 of that under the old rate).

**Rule worth remembering:** a guard protecting BYTES must live with the
bytes, not with whichever Python object currently wraps them. Any future
path that retires a `Buffer` while its range remains reachable (VMM
remapping is the obvious next one) must carry pending guards across.
