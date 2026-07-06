# VMM slab v1 — non-contiguous backing for fast memory

Status: DESIGN → implementing (2026-07-05). Microbench: `tools/bench_vmm.py`.
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
- **Physical pool**: a few large `cuMemCreate` handles (≤ 2 GiB each)
  totaling the ledger budget + rounding headroom, created at Session init
  (~150 µs/GiB, once). A best-fit extent allocator (the existing
  `SlabAllocator` hole logic, reused verbatim) carves **physical** offsets.
- **get(object)** → carve physical extents → `cuMemMap` each extent into the
  object's VA (+`cuMemSetAccess`) → return `Buffer(ptr=VA)`. Contiguity
  failure is impossible by construction: the allocator may return K disjoint
  extents; K maps instead of 1.
- **put(object)** → unmap + return extents to the pool. See §4 for ordering.

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
- **Fragmentation**: cannot fail; K>4-piece maps are counted and reported
  (expected ≈1 steady-state with repeating sizes + coalescing).
- **What VMM does NOT fix**: PressureFit *annotation* infeasibility (e.g.
  bs8ga8@24's planning failure is sim-side, before placement) and torch
  scratch (already VMM'd via expandable_segments). Claims stay scoped.

## 8. Gates & rollout

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
