# The step boundary: invariants, costs, and the fix (2026-07-03)

Reference workload throughout: llama3-8B bf16 AdamW, seq 1024, 65,536
tok/step, bs=8/ga=8, RTX 5090, budgets 12–24 GiB. Everything quantified
below is measured (summary rows / nsys) or sim-derived from the actual
annotated plans — no estimates.

## 1. Why replaying one plan per step is correct at all

`train()` replays a single-step annotated chain once per optimizer step.
That is sound if and only if the plan's **postcondition re-establishes its
own precondition** — an inductive invariant on the boundary state. The
boundary state that matters is:

- **fast contents**: which objects (and with static placement, which
  offsets) hold live copies, and their versions;
- **backing contents + freshness**: every persistent object has a pinned
  home copy that is *current* (no fast copy is dirty);
- **in-flight work**: none — `execute()` drains before returning
  (quiescent seam);
- **per-step inputs**: tokens/targets staged in their pinned buffers.

The current chain establishes, at both ends, the maximally conservative
invariant **B0**: *fast empty, backing complete and fresh*. The lowering
pins it structurally — persistent objects are `initial: backing` +
`final_locations: backing`, the sim's mutation rule forces a writeback
after the optimizer's in-place update, and every fast copy is released by
last use. Replay-correctness is then induction: `B0 → step → B0`.
The pinned buffers double as the carry: the plan's final offloads
overwrite them in place, so step N+1's initial objects ARE step N's
results (no host copies, no replanning).

B0's virtue is that nothing about the previous step needs to be trusted
at the seam. Its cost is that the seam moves ~30 GiB through PCIe per
step with the GPU largely idle. Measured decomposition:

| seam component | where it lives | 12 GiB | 16 | 20 | 24 |
|---|---|---:|---:|---:|---:|
| setup upload (greedy pre-placement set, synchronous, host-serial) | wall − makespan | 0.32 s | 0.42 | 0.42 | 0.42 |
| optimizer-tail drain (GPU-idle share of the window from first optimizer task to chain end) | inside makespan, sim-agreed | 2.00 s | 1.83 | 1.67 | 1.52 |
| total seam waste (≈ share of a ~21 s step) | | 2.3 s (11%) | 2.3 (11%) | 2.1 (10%) | 1.9 (9%) |

Two distinct mechanisms:

1. **The setup-copy subsidy.** PressureFit pre-places backing objects on
   fast at t=0 greedily (earliest-first-use fill to capacity — 14.96 GiB
   of W at 20/24 GiB, 11.54 at 12). Initial placement is *free in
   simulated time*, so the policy maximizes it; the runtime then realizes
   it as synchronous `memcpy` before the chain's clock starts. The bytes
   are honest, the *accounting* was not: the sim never charged them, the
   wall metric paid them.

2. **The optimizer-tail drain.** All 34 optimizer tasks sat after all
   grad-accum rounds. The phase's compute is trivial (~70 ms of fused
   AdamW) but its traffic is not: O in (29.9 GiB) + W back for the update
   (8.7–13.6 GiB re-prefetched, the planner having evicted W after bwd)
   + W writebacks (14.96) + O writebacks (29.9). With no compute left to
   hide under, it runs as a pure PCIe pipeline: the sim shows the GPU
   idle for 1.5–2.0 s in that window at every budget. This cost was
   *sim-visible all along* — real-vs-sim agreed within −3% — which is
   exactly why it hid: we kept validating real against sim instead of
   asking whether the plan's tail was necessary.

## 2. The fixes (both are planning/authoring changes, no new semantics)

**Optimizer interleaving** (`ShapedLlamaConfig.optimizer_placement =
"interleaved"`, the default; `"tail"` = legacy). Each optimizer task is
emitted immediately after the LAST mutation of its gradient — i.e.
inside the final grad-accum round's backward: `optimizer_head` right
after the last `head_bwd`, `optimizer_i` right after the last
`block_bwd_i`, `optimizer_embed` after the last `embed_bwd`. Task ids,
inputs, and the task SET are unchanged (profiles and golden math carry
over); only chain order moves. Consequences:

- O_i prefetches and W_i/O_i writebacks overlap the remaining backward
  compute (~2.3 s of it) instead of draining serially at the end;
- the optimizer's W re-prefetch disappears (W_i is still resident from
  `block_bwd_i` a moment earlier);
- dW_i dies at `optimizer_i` while still on fast (previously it was
  mutation-offloaded and re-prefetched into the tail at tight budgets —
  7.3 GiB each way at 12 GiB).

Correctness argument: `dW_i`'s final mutation is the last round's
`block_bwd_i` (grad accumulation is complete there by construction);
no task after `optimizer_i` within the step reads `W_i`, `O_i`, or
`dW_i` (backward proceeds toward layer 0; recompute/bwd of layer j<i
touch only W_j). Structural tests pin both properties, and the golden
multi-step gate + plan-invariance run on the interleaved default.

**Honest head** (`plan_program(..., preplace="task0")`, the default for
the runtime; the simulator's own default stays `"greedy"`). PressureFit
pre-places only task 0's inputs; everything else arrives as planned
prefetch triggers — charged by the sim, overlapped under early forward
compute by the inbound schedules, and visible in the trace. The setup
copy shrinks from ~15 GiB to W_embed + tokens (~1 GiB, ~30 ms). This
also closes most of the standing wall-vs-makespan gap, so sim
predictions and wall throughput become directly comparable.

## 3. What the invariant becomes

Unchanged in kind: the steady-state contract is still **B0** (fast
empty, backing fresh, quiescent seam). Both fixes only *reprice* it —
they move the seam's traffic to where compute can hide it. That is why
neither touches the runtime engine, the pool, placement, or golden math.

The stronger invariant — **B(R): a resident set R carried in fast across
the seam** (`final_locations: "fast"`, session-owned carryover slots,
placement-pinned offsets, deferred write-back with an explicit flush) —
was analyzed and deliberately **deferred**. The sim schema already
supports most of it (`final_locations` accepts `"fast"`; PressureFit's
emit skips release AND writeback for final-fast last intervals; the
validator's dirty-tracking rejects stale-backing reads *provided the
chain declares resident objects fast-only in initial memory*). But the
measured traffic says its marginal value at these budgets is small once
the two fixes land: W is NOT fast-resident mid-step anyway — the planner
deliberately streams it (161–206 GiB/step of W prefetch against 15 GiB
resident) to keep capacity for saved context, so a pinned R would fight
the recompute planner for exactly the bytes it uses best. Revisit-if:
budgets ≫ working set (W residency becomes free), or CUDA-graph capture
lands (stable addresses across steps become structurally valuable).

Requirements recorded for that future variant: R declared fast-ONLY in
initial memory (a live backing entry would let the policy legally
release-and-reprefetch from a stale home copy — the validator starts
`dirty=∅` and cannot see cross-step staleness); residency intervals
extended to chain end and made uncuttable; engine skip of setup-copy /
end-free for session-carried slots; escape valve forbidden for R;
`session.flush_resident()` before any backing read (checkpoint, final
values, goldens).

## 4. Is a 2-step (k-step) planning window needed?

No. A window does not create a repeatable plan — the boundary invariant
does, and a 1-step plan under B0 is already a fixed point. What a k-step
window buys is *cross-seam overlap of whatever the invariant still
leaves exposed*: with the two fixes, that residual is the last
optimizers' writebacks (O_embed + tail ~2.5–3 GiB ≈ 0.1–0.2 s) which
have no later compute in-step to hide under, plus the ~1 GiB task-0
pre-place. Against that, doubling the window doubles planning/packing
cost, doubles plan artifacts, and needs either a seam invariant at the
window edge anyway (same problem, amortized) or whole-run lowering
(unbounded). The equivalent runtime-side win — starting the next step's
head transfers before the previous `execute()` fully drains — is a
bounded, later micro-optimization (~1%), superseded if CUDA-graph
capture changes execute() structure anyway.

## 5. The ledger inversion the interleave exposed (and the eviction valve)

Interleaving moved the O_i prefetch triggers into the backward's
pressure window and made a latent engine race reachable: **individually
legal h2d admissions can collectively strand the ledger**. The sim
proves a feasible schedule under ITS timing; the engine admits the h2d
head whenever capacity exists at real-clock moments. When real timing
diverges from sim timing (transfer completions arriving early relative
to compute — extreme at test scale where sim runtimes are analytic µs
and real dispatch is 100× that), the one-in-flight FIFO can grab freed
bytes for a far-future prefetch at a moment the sim's still-busy h2d
engine could not, in front of a nearer task's reservation whose release
the prefetch now depends on. Quiescent deadlock; the plan was valid, the
timing was not the sim's. (Same class as the placement-pool inversion of
docs/notes/placement-deadlock.md, one level down: bytes, not offsets.)

Fix: a **pressure-eviction valve** at the quiescent-deadlock site (after
the placement escape): evict ONE fast-resident object that is (a) clean
— fast/backing versions match the record, backing live; (b) not needed
by the stalled task; (c) not touched by any plan directive before its
next use (so no release/offload/prefetch misfires on the emptied slot);
choosing the farthest-next-use victim (Belady), then queueing its reload
ahead of that use. Freed bytes go to a pre-existing blocked queue head if
one exists (the simulator's own heads-first priority), else to the
stalled reservation. The result is a state the sim itself could have
produced under different timing — an eviction+reload is exactly a
*deferred prefetch* decided late — so plan semantics, the budget cap
(evictions only free), and golden math are all preserved. Counted and
traced (`pressure_evictions`, `pressure_evict` events), 0 in healthy
runs; a thrash guard (10× task count) lets genuine capacity deadlocks
still raise loudly. Deterministic regression: `FakeBackend(time_scale=)`
distorts virtual timing (h2d 1000× faster) to force the inversion —
valve on: completes within budget, evicting exactly the Belady victim;
valve off: the quiescent deadlock, as required.

## 6. Results

See results/m4/seq1k-boundary-v1/ (real + wall + sim rows, old vs new).
