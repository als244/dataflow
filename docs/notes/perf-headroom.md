# Known performance headroom (measured, 2026-07-03)

Reference workload: llama3-8B full bf16 AdamW, seq 1024, 65,536 tok/step,
bs=8/ga=8, RTX 5090. Ours ≈ 3,200–3,250 tok/s; flextrain ceiling on the
same machine ≈ 3,410–3,435 (VRAM-insensitive from 17 → 29 GiB alloc — its
limiter is pipeline, not memory). Every number below is measured, not
estimated (`DATAFLOW_DISPATCH_STATS=1`, isolated enqueue benchmarks,
per-phase accumulators).

## 1. Per-task-boundary host tax: 815 µs (≈2.8%/step at seq-1K)

Strict pacing exposes host work between tasks that pipelined engines hide:

| component | µs/boundary | mechanism |
|---|---:|---|
| eager op dispatch | ~510 | ~30 op enqueues × ~17 µs (cuBLAS heuristic+launch 15–25, Triton launcher marshaling 15–25, allocator ~8, adds/copies ~10) |
| completion-token detect | ~134 | poll-loop lag from event completion to host notice |
| DLPack view fan-out | ~46 | ~30 layout field views × 2.2 µs (cacheable to 0.06 µs) |
| bookkeeping (ledger/pool/events) | ~20 | |

Cross-check: 662 boundaries × 815 µs ≈ 0.54 s/step ≈ the entire
real-vs-sim deficit (−3…−4%) — the sim's unmodeled term is exactly this.

**Remedies, in order:**
- **Dispatch-ahead (built, measured, REVERTED — recover at 21243be)**:
  verdict: +1.4% at the compute-bound top, a wash at 16/20 GiB, −1.8%
  at the transfer-bound bottom — not worth the engine complexity.
  The gaps' true fix is CUDA-graph capture per dataflow-task (below).
  Design record: plan-derived sync points; free boundaries enqueue immediately,
  hiding the tax under GPU execution. Two instructive failed versions:
  v1 pre-charged the ledger at enqueue and STARVED transfer admission
  (−20%: prefetches compete for the same ledger); v2 "drained
  opportunistically" via next_completion, which BLOCKS while work is in
  flight (−15%: serialized every boundary). v3: token-paced ledger in
  placed mode (physical safety = the placement proof; transfers behave
  exactly as strict) + a genuinely non-blocking poll_completion.
  STRUCTURAL CEILING: tight packing reuses offsets densely, so ~50% of
  boundaries must sync (free-running fraction 49.5% on the reference
  plan) — packing tightness and dispatch-ahead are in direct tension.
  Raising the ceiling needs slack-aware packing (trade extent for
  anti-adjacency) or the graphs endgame below. Bitwise-equal to strict
  (same chain order -> same pool sequence -> same buffers).
- **CUDA-graph capture (endgame, M5)**: replaying a captured task span
  costs ~5–10 µs total vs ~510 µs of re-enqueue — a 50–100× reduction that
  also wins in small-task regimes (seq ≤ 512) where the tax grows.
  Static placement's fixed addresses satisfy graph capture's stable-pointer
  precondition by construction; capture compute spans between sync points,
  keep transfers/admission event-driven outside.

## 2. Dispatch-count bloat in composed blocks

`block_bwd` = ~37 kernel dispatches (host enqueue 565 µs); a fused-layout
implementation needs ~8–12. Expansion sources, largest first:

1. **9 separate dW GEMMs** (one per weight tensor). Fused `[wq|wk|wv]` and
   `[w1|w3]` weight packing → ~5 GEMMs, each larger and more efficient.
   Touches layouts + lowering + golden model + checkpoints; high value,
   real surgery.
2. **rmsnorm_bwd = 4 dispatches** (zeros + kernel + partial-sum + copy) ×2
   per block; a direct-write two-pass kernel makes it 2 total.
3. **Unfused matmul+add chains** (`dh1`/`dh2`: 8 dispatches → 3 with
   GEMM-accumulate).
4. Residual adds/copies not folded into adjacent kernels.

Per-family measured enqueue (idle GPU): block_bwd 565 µs, block_fwd 297,
block_recompute 251 (2.2× cheaper after the w1/w3 truncation), optimizer 38.

## 3. Step-boundary state round-trip (~6%/step)

34 weight tensors (14.96 GiB) are initial-fast AND final-backing: every
step offloads updated weights to pinned memory and re-uploads the same
bytes at the next step's setup — serially, between `execute()` calls.
Fix: fixed-point steady-state plans (final fast state ≡ initial fast
state) with session-resident carryover buffers; also unlocks overlap of
the remaining boundary traffic. Not yet built.

## 4. Recompute truncation (DONE, 81c0043)

Recompute must rebuild only the saved context: stopping at the w1/w3 GEMMs
(the block output y is never a backward dependency) measured
18.94 → 13.12 ms per recompute task (−31%), workspace 2.5 → 0.42 GiB,
host enqueue 2.2× cheaper. Follow-up: staged-forward authoring so the
truncated recompute is DERIVED per block (run stages until the last
context-emitting stage) instead of hand-written — see the extending-guide
plan; plus a waste tripwire (warn when recompute cost ≈ full-forward cost).

## 5. Embed-on-host (built, measured, REVERTED — recover at 53cf971)

Shein's design: the embedding's sparse access lets the table live on CPU
(host gather -> pinned X staging, dy_embed offloaded home, host
scatter-add + host AdamW). Mechanism verified in nsys: the exposed ~30 ms
W_embed prefetch per round boundary and ~4.2 GB/step of embed traffic
genuinely disappeared, and makespans improved +1-2%.

Why it lost anyway — three lessons:
1. **Measurement error (caught by Shein in nsys)**: v1 reported
   makespan-based tok/s while the host update ran OUTSIDE the measured
   window (~2.2 s/step invisible). The wall metric was fixed to cover the
   full step (kept permanently: `wall_tokens_per_s` in every row).
2. **The host update is bytes-bound**: eager torch chain 1,503 ms; fused
   via torch.compile 299 ms — right at the host-bandwidth floor Shein
   estimated (~15 GB of W/M/V/g traffic).
3. **Synthetic-random tokens defeat sparsity**: 65k draws over a 128k
   vocab touch ~40% of the table, so even the split-update (sync rows the
   next gather needs; threaded dense remainder under GPU execution) left
   ~0.9 s of synchronous boundary. Honest wall verdict: +0.7% @ 16 GiB,
   −1.0% @ 20 GiB — below the agreed keep bar (≥1% at two budgets).

Revisit-if: real token distributions (SFT data is far sparser than
uniform-random — the touched set shrinks 5-20x and the economics flip),
or a resident-embed GPU variant at generous budgets (flextrain's choice).

## 6. Smaller, known, deliberately deferred

- Torch reserved-vs-allocated slack can exceed the 256 MiB device-envelope
  pad (~1 GiB observed once); post-run envelope check is the backstop.
- `del`-at-last-use in block_bwd: MLP-section giants stay referenced
  through the attention section (~1 GiB of peak scratch; raises the
  placement physical limit when fixed).
- Duty-cycle-matched contended profiling (unbiased cost model, ±5% band
  accepted instead).
- VMM chunk-backing: removes the geometry tax (×1.03–1.19) AND the
  offset-coupling replay gaps AND the lifetime-inversion escape class.
