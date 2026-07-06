# VMM slab v1 — non-contiguous backing for fast memory

Status: IMPLEMENTED (2026-07-05) — gates green (6 unit/E2E incl. bitwise
loss equality vs static mode; suite 167). Benchmarks vs baselines below. Microbench: `tools/bench_vmm.py`.
Baselines to beat/match: `artifacts/m5/llama3-s1k-v2/README.md` (llama grid),
qwen35 bs32ga2 records (3,001 @14 ledger; 24.19 GiB device @16 w/ expandable).

## 1. Problem

Fast-memory buffers are backed today by contiguous ranges in one `cudaMalloc`
arena. Contiguity in (address × lifetime) is a packing problem, and we pay
for it three ways, all measured:

1. **Extent tax ×1.05–1.21**: packed extent > ledger peak (llama @16: ledger
   16 → extent 17.39 GiB; dev-24 qwen35: 15.88 → 17.08). 1–2.5 GiB of a
   32 GiB card spent on address geometry.
2. **Placement infeasibility**: configurations whose ledger fits at every
   instant but admit no contiguous assignment (M4.9 llama flat-24). A whole
   failure class that exists only because of the formulation.
3. **Accidental complexity**: the placement dry run + interval packer with
   48 random restarts, the m4_train extent-shave replan loop (≤6 attempts),
   the `--extent-budget` knob, dual ledger/extent reporting, dynamic-mode
   headroom + flush-retry + overflow arenas + vendor escapes, assigned-mode
   busy-gating with quiescent-deadlock escapes.

Both `slab.py` and `placement.py` name VMM chunk mapping as the intended
successor in their module docstrings.

## 2. Mechanism

CUDA VMM (`cuMemAddressReserve` / `cuMemCreate` / `cuMemMap` /
`cuMemSetAccess` / `cuMemUnmap`) decouples what kernels need (per-object
virtual contiguity) from what the budget owns (physical pages).

- **VA arena**: one reservation per Session (tens of GiB is free; 48-bit VA).
  Every object id gets a **stable virtual range for the life of the run**,
  assigned first-touch, sized to the object (2 MiB-aligned). Stability makes
  replays deterministic and guard tracking per-object rather than
  per-address-overlap.
- **Physical pool** (REVISED during implementation): `cuMemMap` cannot map
  a sub-range of a physical handle — offset must be 0 and size must equal
  the allocation (`CUDA_ERROR_NOT_SUPPORTED` otherwise; the microbench
  passed only because it mapped whole handles). Extent-carving over big
  handles is therefore impossible. The design that survives is SIMPLER:
  **exact-size handles, free-listed by size, mapped whole (one call)**.
  Shape-stable programs make size demand periodic, so free lists stabilize
  after round one and steady state does zero create/release work.
- **Budget by reflow**: `created_bytes ≤ pool` is enforced by releasing
  cached handles of other sizes (largest first) before creating a new one
  (~150 µs per create — warmup-round only). This is the escape the
  contiguous slab never had: physical bytes flow across size classes on
  demand instead of summing per-class maxima.
- **get(object)** → free-listed (or created-under-budget) handle →
  one `cuMemMap` + `cuMemSetAccess` into the object's VA →
  `Buffer(ptr=VA)`. **put(object)** → unmap, handle back to its size's
  free list. See §4 for ordering.

Physical occupancy ≡ ledger occupancy + per-object 2 MiB rounding. The
extent tax is eliminated by construction, not by a better heuristic; the
packing solver, its dry run, and the shave loop have nothing left to decide.

## 3. Microbench (RTX 5090, 2026-07-05)

| operation | cost |
|---|---|
| granularity | 2 MiB (min = recommended) |
| cuMemCreate+Release | 12.6 µs @2 MiB → 153 µs @1 GiB (init-only; pooled) |
| map+setAccess+unmap, one range | **38.9 µs @2 MiB → 52.4 µs @1 GiB** — per-call, not per-page |
| fragmented map of 416 MiB | K=1: 40 µs, K=2: 59, K=4: 99, K=16: 353 — ~20 µs/piece |
| churn vs bandwidth kernel | −1.8 % (noise) across 124 concurrent map/unmap cycles |
| stable-VA physical swap | isolated + bytes persist per handle (verified) |
| sub-range map of a handle | **NOT SUPPORTED** (offset must be 0, size == allocation) — forced the exact-size-handle design |

Steady-state rate: ~2–5 object births per task, tasks 2–160 ms → mapping
overhead ~0.1–0.5 % of wall, host-side and overlappable with device compute.
First-map lazy-init spike (one-off 4 ms) is absorbed by a warmup map at
arena construction.

## 4. Unmap ordering (the one subtle piece)

`cuMemUnmap` is host-ordered; in-stream address-reuse arguments do NOT apply
(a queued kernel touching an unmapped VA faults). Two engine facts make the
discipline cheap:

- `pool.get` is called before the consuming task is enqueued (the kernel
  needs the pointer), so **map strictly precedes device execution**. ✓
- `pool.put` for releases fires in `_on_task_done` — the host has already
  observed the completion event of the releasing task, and the single
  compute stream orders all prior consumers before it. Offload puts fire on
  transfer-completion handling. Pressure evictions fire only when the
  anchor task is complete. So at put() time the device has provably passed
  every consumer **except** a guard (debug poison memset queued at
  release-time itself).

Rule: **unmap inline at put(), unless `guard_event` is pending — then defer
(object, extents, event) to a reclaim list** drained lazily before any carve
that would otherwise fail, and at step boundaries. This is the existing
`_placed_pending_guards` pattern with per-object keys instead of
address-overlap scans.

## 5. What dies, what stays

Dies (in VMM mode; legacy modes remain until the default flips):
- `compute_placement` packing + restarts, `PlacementRecorder` dry run,
  `PlacementError` at planning, m4_train extent-shave loop + `--extent-budget`,
  assigned-mode busy-gating (`can_get` overlap scan), `get_escaped` lifetime-
  inversion valve, `_placed_pending_guards` range scans, slab `headroom_factor`,
  flush-retry, overflow arenas, vendor-escape counters — for fast memory.
- The extent/ledger split-brain: sweep rows report one number again
  (`placement_extent_gib` ≡ ledger peak + rounding).

Stays unchanged:
- **Ledger** (logical admission; sim parity semantics) — in VMM mode the
  physical pool is sized to the ledger budget, so ledger admission ≈
  physical admission; a loud assert covers the rounding delta.
- Buffer abstraction (`ptr` is a VA like any other), task executables,
  pinned-host backing pools, FakeBackend (sim/tests), the planner stack.
- Determinism: VA assignment and extent carving are pure functions of the
  (deterministic) allocation sequence.

## 6. Integration surface

- `runtime/device/vmm.py` (new): `VmmArena` — VA reservation, physical
  handle pool, extent allocator, map/unmap, deferred reclaim, stats
  (maps, fragmented_maps, reclaim_stalls, rounding_bytes).
- `pool.py`: third fast-memory mode `enable_vmm(arena)`; `get` maps,
  `put` routes `raw=("vmm", ...)` to the arena. `can_get` returns True
  (ledger is the gate).
- `engine.py`: accepts `placement_mode="vmm"`; skips recorder/placement.
- `train_loop.py` / `m4_train.py`: `--placement vmm`; device-gib derivation
  drops the extent iteration (envelope − fixed − scratch = ledger, directly).
- Session owns the arena (created once, persists across steps/executes).

## 7. Failure modes & honesty

- **Rounding headroom**: pool = budget + max(256 MiB, 2 MiB × live-object
  high-water). If a carve fails after reclaim draining, that's a loud
  invariant error (ledger admitted bytes the pool can't back) — never a
  silent fallback.
- **Fragmentation**: does not exist — handles are exact-size and mapped
  whole. The analogous metric is `handle_reflows` (budget-forced releases),
  expected ≈0 after the first round.
- **What VMM does NOT fix**: PressureFit *annotation* infeasibility (e.g.
  bs8ga8@24's planning failure is sim-side, before placement) and torch
  scratch (already VMM'd via expandable_segments). Claims stay scoped.

## 8. RESULTS & VERDICT (2026-07-05)

Benchmarks vs recorded static baselines (same profiles, `artifacts/m5/vmm-v1`):

| config @ ledger | static: wall / device | vmm: wall / device | reflows/3steps |
|---|---|---|---|
| llama bs16ga4 @24 | 3,600 / 31.11 | 3,487 (−3.1%) / **28.08** (−3.0) | 625 |
| llama bs32ga2 @16 | 3,602 / 25.29 | 3,558 (−1.2%) / **20.94** (−4.3) | 338 |
| llama bs32ga2 @24 | 3,629 / 30.60 | 3,552 (−2.1%) / **29.00** (−1.6) | 299 |
| qwen35 bs32 @16 | 2,973 / 24.19–25.51 | 2,928 (−1.5%) / **23.32** | 418 |

Fidelity degraded to 3.3–5.8% (baseline 0.25–0.46%) — fully attributed:

1. A FRESH handle costs **~1.9 ms/GiB** beyond the create call (hidden page
   sanitization, measured 2.88 ms at 1.5 GiB fresh-vs-reused), and that
   sanitization **contends with bandwidth-bound compute** (+11.5% on a copy
   kernel; +0.1% on matmul) — so creates cannot be hidden by async prep.
2. Creates never stop: the ledger touches the pool ceiling every round, so
   every size-class's cached handles are destroyed at each peak crossing and
   recreated after — ~100–150 churn cycles/step, PERIODIC, not warmup.
   Cache allowance barely helps (reflows 418 → 303 at +3.5 GiB; the cycling
   class zoo is ~10 GiB wide). ~120 × 2.9 ms ≈ the whole wall gap.

**Verdict: a memory↔throughput TRADE, not a strict win.** Genuine −1.6 to
−4.3 GiB device at −1.2 to −3.1% wall. Root cause is a driver limitation:
physical handles are unsplittable (no sub-range map), so exact-size handles
are forced, and exact-size + tight budget = periodic churn. Structural wins
stand regardless: no placement-infeasible class, no packing/shave loop,
feasibility guaranteed for any admitted ledger.

**Disposition: `--placement vmm` stays as a documented opt-in** for
memory-bound configs (it runs shapes static cannot place, and buys real
device headroom when −2% wall is acceptable); **static remains the
default**. What would flip it: driver support for sub-range or batched
mapping (then physical bytes reflow without create/sanitize), or a no-zero
create flag — worth re-testing on future CUDA releases.
`DATAFLOW_VMM_HEADROOM_GIB` tunes the cache allowance.

## 9. Improvement iterations (2026-07-05, after the v1 verdict)

Shein asked for attribution + improvement + the device-normalized fight.
Three iterations, each measured (dev-24 envelope, static vs vmm):

| iteration | mechanism | bs8ga8 | bs16ga4 | bs32ga2 | qwen35 bs32 |
|---|---|---|---|---|---|
| v1 exact-size | map per birth | — | −2.3%* | −2.8% | −1.7% |
| + smallest-first victims, demand prewarm | cheap creates | −5.6% | −2.3% | −2.8% | **parity @16 ledger** |
| + tag-bound parking | zero-call re-gets | −7.9% ✗ | −4.2% ✗ | −2.7% | −2.5% |
| + **slot adoption** (VAs belong to slots) | zero-call same-size births | **−3.0%** | **−0.8%** | **−1.0%** | **−1.5%** |

(*ledger-matched rows for the pre-matrix iterations.)

Key findings along the way:
- **No implicit device sync**: the cost is host dispatch-path driver calls +
  device-side page sanitization of FRESH handles contending with
  bandwidth-bound work. Attributed with per-call-class timers.
- **Tag-bound parking backfired** (48 hits vs 4,271 steals): same-SIZE
  siblings dominate rebirth order. Slot adoption (bind VA to the handle,
  not the object) is the correct inversion — the slab free-list reborn at
  VA level, zero driver calls per steady-state birth.
- **--device-gib works in vmm** (ledger = envelope − fixed − scratch −
  headroom, no extent iteration) and grants +0.4–1.9 GiB ledger over
  static's derivation. The planner converted it into less recompute ONLY
  at bs8ga8 (165→128 of 256) — elsewhere recompute sits on the M5.2
  contention plateau where more memory buys nothing.
- **Residual floor**: ceiling reflows — at pool==ledger-peak, cross-class
  evictions destroy+recreate handles every step (creates≈reflows≈300–1,400
  per 3 steps). Fidelity stays 1.5–5.6% (sim doesn't price driver work).

**Complete device-normalized matrix (same verified --device-gib, h=0.5)**,
Δ = vmm vs static wall:

| config | dev-16 | dev-18 | dev-24 |
|---|---|---|---|
| llama bs8ga8 | **+0.5%** (rc 201→128) | — | −3.0% |
| llama bs16ga4 | −0.9% | **+1.4%** | −0.8% |
| llama bs32ga2 | −0.4% | −1.5% | −1.0% |
| qwen35 bs32ga2 | — | −2.3% | −1.5% |

vmm wins exactly the cells where the extra ledger CHANGED the plan or its
pressure (bs8ga8@16: static's shave-taxed ledger forced 201/256
recomputes vs vmm's 128 — the designed mechanism end-to-end; bs16ga4@18).
Everywhere else the planner's choices are ledger-insensitive (bs32's rc
stays 32/64 from ledger 9.9 to 19.1!) and only the churn shows. Slack
does NOT fix the residual churn (+1 GiB pool slack: reflows 469 → 405 —
the cycling class working set is far wider than any affordable slack).
Side-finding: static qwen35 @ dev-18 = 3,023 tok/s — a new qwen35 record
at a SMALLER envelope (recompute-relieves-contention, again).

**Final verdict (regime-split)**: at GENEROUS envelopes vmm loses 1–3%
(the planner is on the recompute-contention plateau; extra ledger buys
nothing) — static stays default there. At TIGHT envelopes (the regime
offloading exists for) vmm ties or wins outright, and it categorically
runs shapes static cannot place. Recommended: static default, vmm the
documented choice for memory-tight configs; revisit the default if the
driver gains sub-range mapping / no-zero creates (kills the residual
churn entirely).

## 10. Gates & rollout

1. Unit: arena carve/coalesce/fragmented-map/reclaim-order/stable-VA.
2. GPU integration: mini + tiny E2E in vmm mode — goldens bit-identical to
   dynamic-slab mode (same kernels, same values; only addresses differ).
3. Full suite green with vmm-mode GPU tests added alongside existing modes.
4. Benchmarks vs recorded baselines (same profiles): llama3-8b s1k
   {bs8ga8, bs16ga4, bs32ga2} × {12, 16, 20, 24} and qwen35 bs32ga2 @16 —
   compare wall tok/s, measured device peak, planning time, and the @24
   placement-infeasible rows. Flip the default only if strictly ≥ on
   throughput and memory with fidelity intact; otherwise report and keep
   static placement.
