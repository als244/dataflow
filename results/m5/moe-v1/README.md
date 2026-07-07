# MoE first perf rows (M-E) — olmoe-7b + qwen35moe-20l

Stack: pluggable MoE module v1 (docs/notes/moe-design.md), head_loss +
scratch-discipline + recompute-default baseline, expandable_segments,
contended profiles + 1s soak. Quoting convention: DEVICE envelopes
(`--device-gib`; actual measured peak <= quoted). 65,536 tok/step
(s1k shapes), 3 steps, steady wall tok/s. NOTE: a concurrent process held
~1.4 GiB on the GPU during these runs; it is counted inside
fixed_overhead (envelope math stays honest) but slightly pressures the
tightest envelopes.

## OLMoE-7B-A1B (6.92B params, ~1.2B active; W bf16 = 13.4 GiB; pinned 53.9 GiB)

### bs16ga4 (m4-olmoe-7b-s1k-bs16ga4-12_16_20_24dev-rc.summary.json)

| dev GiB | wall tok/s | real-vs-sim | fidelity | recompute | device peak |
|---|---|---|---|---|---|
| 12 | 7,757 | -5.8% | 3.4% | 58/64 | 11.74 |
| 16 | 8,641 | -4.4% | 6.2% | 50/64 | 15.75 |
| 20 | 8,989 | -5.3% | 6.2% | 44/64 | 19.46 |
| 24 | 9,685 | -2.7% | 5.0% | 35/64 | 23.12 |

### bs32ga2 (fewer/larger rounds — the M5.2 h2d-bound lesson applied)

| dev GiB | wall tok/s | real-vs-sim | fidelity | recompute | device peak | note |
|---|---|---|---|---|---|---|
| 16 | 10,892 | -9.5% | 0.60% | 28/32 | 15.49 | solo card (10,986 under idle co-tenant, rc-29 plan — reproducible within 0.9%) |
| >=20 | — | — | — | — | — | BLOCKED: deterministic engine deadlock (see fragility finding) |

bs32ga2@16 = **+26% over bs16ga4@16**: halving the rounds halves the
per-step weight restreaming, exactly the M5.2 prediction; fidelity
collapses 6.2% -> 0.60% (traffic-minimal). The -9.5% real-vs-sim WITH
0.6% fidelity isolates the residual gap to the plan-sim's
transfer-overlap optimism, not task costs.

### bs64ga1 (single round: the expert stack streams ONCE per step)

| dev GiB | wall tok/s | real-vs-sim | fidelity | recompute | device peak |
|---|---|---|---|---|---|
| 24 | 12,269 | -3.7% | 0.25% | 13/16 | 23.91 |
| 26 | 12,690 | +2.3% | 0.27% | 15/16 | 26.01 |
| 28 | **12,992** | -2.2% | 0.31% | 15/16 | 27.85 |
| 30 | 12,838 | -5.2% | 0.34% | 14/16 | 28.40 |

**12,992 wall tok/s @ dev-28 = the OLMoE record so far** (28->30
plateau/noise, rc 14-15/16 = recompute-nearly-everything even at
generous envelopes). Infeasible <= 20 (single-round working set).
Fidelity 0.25-0.34% at one round/step — the traffic thermometer at its
floor, i.e. sim-vs-real is TIGHT exactly when transfer traffic is
minimal, confirming the transfer-model attribution of the bs16 rows'
3-6% gaps. bs64ga1 profiling needed best_config to adopt m4_train's
expandable_segments convention (2 GiB tail temporaries fragment the
default allocator).

### Plan-fragility finding (bs32ga2 at ledger ~14.1 GiB — REPRODUCIBLE)

The envelope derivation lands on a ledger-14.1 rc-20/32 plan for BOTH
dev-20 and dev-24 (solo card), and that plan DEADLOCKS in the real
engine DETERMINISTICALLY (twice, bit-for-bit: eviction valve churned
1080x, then block_fwd_0_1_15's 1.89 GB y+A reserve — MoE's huge
A-objects — found nothing evictable; the designed loud failure). The
conservative rc-29/32 plan at the same envelope trains fine at ~11k.
MORE ledger produced a MORE FRAGILE plan — an M4.9-class byte-timing
inversion the valve cannot escape, amplified by ~1.9 GB A-objects.
The v1 replay-fidelity "diagnostic crash" at this exact task/bytes was
the same fragility surfacing in the sim (fidelity=null rows deserve
suspicion, not just tolerance). ENGINE FOLLOW-UP (flagged for Shein):
eviction-churn detector -> ledger shave + replan, or valve candidate
widening. bs32 rows >=20 left unquoted; the 24-30 frontier is owned by
bs64ga1 regardless (the oracle's flagged 0.9% dev-24 margin
bs32-vs-bs64 is UNSETTLEABLE until the fragility is fixed).
POST-LEAN-TAIL PROBE: the workspace cleanup changed the envelope
derivation (dev-20 ledger 14.10 -> 13.97, rc-20 -> rc-22 plan) and BOTH
bs32@20/24 now train clean (12,342 / 12,654 wall; bs64 still wins those
envelopes). The specific fragile plan is no longer produced; the CLASS
is unfixed — valve completion (dirty-evict via writeback enqueue) is
the real cure, plan-time fragility check the cheap guard.

## flextrain comparison curve (APPLES-TO-APPLES, leeway releveled)

Matched workload: synthetic s1024, 65,536 tok/step, full bf16. Both
systems TARGET a peak-device budget: ours via --device-gib (verified
peak <= B), flextrain via --max-gpu-mem-gib B --leeway-gpu-mem-gib 1.5
(their default leeway 5 strangles them at tight budgets: 9,878 ->
11,178 tok/s at B=12 after releveling). Their card peaks land
~0.4-0.75 GiB ABOVE B, so the honest pairing is by MEASURED peak on
both sides (1 Hz sampler; logs flextrain-olmoe-lw1.5-dev*.log):

| ours peak GiB -> tok/s (shape) | flextrain peak GiB -> tok/s | delta |
|---|---|---|
| 11.74 -> 7,757 (bs16ga4@12) | 12.39 -> 11,178 | flextrain +44% |
| 15.75 -> 10,892 (bs32ga2@16) | 16.49 -> 11,692 | flextrain +7% |
| 19.46 -> 8,998 (bs16ga4@20)* | 20.48 -> 11,839 | flextrain +32%* |
| 23.91 -> 12,269 (bs64ga1@24) | 24.76 -> 11,922 | ours +2.9% |
| 26.01 -> 12,690 (bs64ga1@26) | 26.76 -> 11,886 | ours +6.8% |
| 27.85 -> 12,992 (bs64ga1@28) | 28.75 -> 11,934 | **ours +8.9%** |

*dev-20 quotes our best RUNNABLE shape: bs32@20's ~12.3k (sim) plan is
blocked by the deterministic fragility corner; bs64 infeasible there.
flextrain's curve is FLAT (~11.2-11.9k) from 12 GiB up — their
stream-once-at-any-memory schedule's signature. Crossover ~23 GiB;
above it the dataflow planner's recompute freedom wins, below it our
per-round weight restreaming pays (next section). The default-leeway
curve (flextrain-olmoe-dev*.log) is retained for reference.

### The structural finding (the top planner follow-up)

flextrain streams the expert stack ONCE per step at ANY memory: its
chunked schedule token-chunks WITHIN the step, so weight traffic is
independent of the effective "rounds". Our chain grammar restreams W
every grad-accum round — bs16ga4 pays 4x (53.7 GiB/step of expert
weights), and our stream-once shape (bs64ga1) only FITS at >=24 GiB
because a single round's activation working set is 4x larger. That is
the entire low-memory deficit. Fix directions, in rising ambition:
(a) round-resident weights — the B(R) carryover the M4.10 window
oracle REJECTED for llama; the MoE weights>>activations regime flips
that tradeoff hard, so re-run the oracle under MoE dims before
building; (b) intra-round token chunking in the grammar (flextrain's
shape). Either would attack the 12-20 GiB regime where flextrain
currently leads by 7-44% (releveled curve; the 20 GiB point also needs
the fragility fix to field bs32 there).

## Workspace-lean tail (Shein ask: kill fp32 materializations + aten copies)

moe_scale_rows (in-place route-weight scaling, fp32 in registers only),
moe_rowdot (fused dprob dot), dual-mode grouped ops (scratch destinations
use aten's returned tensor — no duplicate + copy pass; ctx write-through
keeps its copy), dh2 accumulates bf16 via addmm_ (the dense convention).
All 47 MoE tests unchanged-green incl. bitwise determinism.

Measured scratch: bs64 14.26 -> 6.76 GiB (-53%), bs32 7.38 -> 3.63
(-51%); compute-only ceilings ROSE ~20% (bs64 20.0k, bs32 19.0k tok/s) —
the copies and fp32 passes were time, not just space.

### The new frontier (lean tail) — beats flextrain at EVERY budget

| dev GiB | wall tok/s (shape) | fid% | rc | peak | vs flextrain releveled |
|---|---|---|---|---|---|
| 12 | 12,403 (bs32ga2) | 0.67 | 28/32 | 11.73 | +11% (11,178) |
| 14 | 12,263 (bs32ga2) | 0.81 | 25/32 | 13.10 | — |
| 16 | 12,887 (bs32ga2) | 0.59 | 26/32 | 15.26 | +10% (11,692) |
| 18 | 13,765 (bs64ga1) | 0.34 | 13/16 | 17.27 | — |
| 20 | **14,615 (bs64ga1)** | 0.38 | 13/16 | 19.57 | **+23% (11,839)** |
| 24 | 14,347 (bs64ga1) | 0.29 | 12/16 | 23.69 | +20% (11,922) |
| 28 | 13,941 (bs64ga1) | 0.38 | 8/16 | 27.23 | +17% (11,934) |

bs32@12 was INFEASIBLE pre-cleanup (7.4 GiB scratch reserve ate the
envelope); bs32@16 gained +18% on identical plans-space. Record:
**14,615 wall tok/s @ dev-20**. The structural round-restreaming fix
(round-resident weights / token chunking) remains as further upside at
12-16, no longer needed for parity.

### Sim ranking caveat (new; oracle users take note)

bs64 REAL inverts above 20 (14,615 > 14,347 > 13,941) while sim says
monotone (16.3k -> 16.8k): larger envelopes -> planner picks LESS
recompute (rc 13 -> 8/16) -> more h2d traffic -> the transfer-overlap
optimism grows (real-vs-sim -10% -> -17%) while per-task fidelity stays
0.3-0.4%. First RANKING-level sim error: within-shape, across envelopes,
at traffic-heavy (rc-light) plans. Contention-aware plan costing (the
M5.2 deferred item) would fix both this and the absolute bias.

## RESOLUTION: sync-free tuned triton grouped GEMM — the final curve

The saga in one line: hidden host syncs (torch.bincount, then
F.grouped_mm's per-call offs readback — spin-kernel audited: all four
aten grouped forms block; every op of OURS is async) starved the GPU
inside task windows, plan-dependently, invisible to profiles. Fix =
custom triton grouped GEMM (device-side tile-prefix + binary search,
STATIC worst-case grid, single-owner deterministic tiles, in-place
wgrad epilogue) with STATICALLY swept tiles (128x256x64 wins at both
bs16 and bs64 shapes; no runtime autotune — nondeterministic across
runs). v1 mis-tiling cost bs64 +12/+48% per op (parity at bs16 — why
bs32 improved while bs64 regressed); the sweep fixed it.

FINAL CURVE (fresh profiles, 3-step wall, fid 0.4-0.6, esc 0):

| dev GiB | wall tok/s (shape) | sim | real-vs-sim | vs flextrain releveled |
|---|---|---|---|---|
| 12 | 13,351 (bs32ga2 rc28) | 13,031 | +2.6% | +19% (11,178) |
| 16 | 14,337 (bs32ga2 rc25) | 13,742 | +4.5% | +23% (11,692) |
| 18 | 14,459 (bs64ga1 rc13) | 14,113 | +2.7% | — |
| 20 | 14,766 (bs64ga1 rc14) | 14,389 | +2.8% | +25% (11,839) |
| 24 | 15,492 (bs64ga1 rc8) | 15,632 | -0.7% | +30% (11,922) |
| 28 | **15,948 (bs64ga1 rc8)** | 16,336 | -2.2% | **+34% (11,934)** |

- MONOTONE again; RECORD 15,948 @ dev-28; beats flextrain +19..+34%
  at every budget.
- real-vs-sim collapsed from -5..-18% to +/-5%: the plan-dependent
  cost-model error WAS the syncs. Residual -2% at rc-light plans =
  the true contention term (small; sim-side costing now optional).
- Planner note: replaying the sync-era rc-13 plan at dev-20's ledger
  gives 15,426 > the greedy derivation's 14,766 — greedy recompute
  selection leaves ~4% at mid envelopes under the new cost surface
  (follow-up: selection quality, M4.10 thread).
- Kernel A/B at bs64 ragged shapes (ms, triton/aten): fwd 21.0/22.3,
  dgrad ~parity, wgrad 26.0/23.9 — plus aten's hard sync in-run.
  aten-grouped demoted to A/B-only (priority 5, allocates="vendor").

## WHY the bs64 curve inverts (dev-20 > 24 > 28): measured decomposition

gap_analysis on the dev-20 (rc-13) and dev-28 (rc-11) plans; webapp
uploads olmoe-b64-dev{20,28}.measured.json; raw dirs gap-b64-dev{20,28}/.

| | dev-20 | dev-28 |
|---|---|---|
| sim -> real tok/s | 15,961 -> 14,500 | 16,852 -> 14,260 |
| TOTAL task-time inflation vs profiles | +14.1% | **+20.8%** |
| moeattn_recompute inflation | +42.3% | **+64.2%** |
| moeattn_fwd / bwd | +14.9% / +6.7% | +16.4% / +14.5% |
| exposed compute idle | 3.5% | 1.8% |
| achieved PCIe h2d/d2h (planned 25.5/25.8) | 35.3 / 43.8 GB/s | 32.2 / 35.9 GB/s |
| replay fidelity | n/a (reserve-infeasible, fragility signal) | **+0.33%** |

VERDICT (corrected by nsys kernel forensics — the earlier HBM-contention
attribution was FALSIFIED): scheduling + transfer models are EXONERATED
(replay +0.33%, idle <=3.5% BETWEEN tasks, transfers over-achieve). The
gap is GPU IDLE INSIDE task windows (rc 48%, fwd 35%, bwd 15% of span)
caused by HIDDEN HOST SYNCS that starve the enqueue pipeline:

1. torch.bincount in moe_sort: aten's input-validation min() reads 1 B
   back to PAGEABLE host = full compute-stream drain (22 ms of queued
   kernels in the traced recompute) + starved refill. FIXED (scatter_add_
   counting, sync-free, integer-atomic deterministic): +0.6-1.0%, new
   record 14,746 @ dev-20 — small because the drain just shifted to:
2. F.grouped_mm: reads its offs tensor back to host per call (cutlass
   builds group descriptors host-side) — 4 syncs/bwd, 1-2/fwd/rc. THE
   DOMINANT TERM; unfixable from our side of aten.

Why profiles miss it: back-to-back profiling reps keep the queue deep on
both sides of a sync (drain costs ~0); in the strict-paced engine each
task starts from an empty queue and pays drain+starve at every sync,
scaled by how much work was enqueued ahead — which is what made it look
plan-dependent (traffic-heavy plans = deeper queues at sync points).

FIX: the custom triton grouped GEMM from the original plan (device-side
offsets, binary-search tiles, in-place epilogue, deterministic wgrad) —
deferred when aten looked sync-free, now REQUIRED: kills 4-6 syncs/task
+ the out-copy tax, projected to recover most of the -9..-18%
real-vs-sim and likely restore monotonicity (re-verify before building
any sim-side contention costing — the kappa numbers quoted earlier
conflate sync drains with contention and are NOT calibration-grade).

## nsys captures (dev-12, for inspection)

- nsys/olmoe-7b-s1k-bs16ga4-dev12-replay.nsys-rep — our plan replay
  (NVTX per task / from_slow:W_i transfer / step; PRE-cleanup capture)
- nsys/flextrain-olmoe-dev12.nsys-rep — their steps 3-5 at max-gpu 12 /
  leeway 1.5 (cudaProfilerApi bounded)

## qwen35moe-20l curve (17.8B hybrid, E=256+shared, ~107.7 GiB pinned W/dW/O)

Oracle (qwen35moe-oracle.json, triton kernel set): bs32ga2 wins >=20,
bs16ga4 at 16, everything h2d-bound (compute ceilings ~10.5-10.7k tok/s;
35.6 GiB weights/round = 2x olmoe's restream burden at similar active
FLOPs). bs64ga1 cannot even profile on 31.5 GiB (single-task working
sets exceed the card) — structurally out for 20L.

| dev GiB | wall tok/s (shape) | sim | real-vs-sim | fid% | rc | peak | backing peak |
|---|---|---|---|---|---|---|---|
| 16 | 3,967 (bs16ga4) | 3,482 | +14.1% | 2.37 | 73/80 | 15.07 | 134.1 |
| 20 | 5,472 (bs32ga2) | 5,451 | +0.6% | 2.82 | 28/40 | 18.61 | 142.7 |
| 24 | 5,899 (bs32ga2) | 5,521 | +7.1% | 3.22 | 26/40 | 23.37 | 138.7 |
| 28 | 6,283 (bs32ga2) | 5,811 | +8.4% | 2.39 | 31/40 | 26.88 | 127.3 |

Monotone; real ABOVE sim everywhere (conservative post-sync-fix
calibration, like olmoe); fidelity 2.4-3.2% = the traffic thermometer
reading warm on ga2/ga4 restreaming (olmoe bs64 runs at 0.3-0.6). At
~55-59% of compute ceiling, this family is the round-restreaming
grammar's clearest customer (intra-round token chunking / round-resident
weights), and with E=256 the per-expert segments are ~4x smaller than
olmoe's — the regime where a cublasLt-per-expert backend (flextrain's
matmul_dispatcher) becomes interesting once host-visible counts exist.

### bs64ga1 correction + final row (the stream-once ending)

The oracle's original "bs64 can't profile" was a PROFILER ARTIFACT:
torch's allocator held the base pass's cached segments while the
rc-variant pass profiled new size classes in the same process
(best_config now empty_cache's between the two passes as well as
between shapes). Re-oracled: bs64 profiles fine (scratch 10.79 GiB) and
is planning-verified INFEASIBLE at dev-20/24 (single-round working set
+ scratch reserve) but WINS dev-28. Verified real:

| dev-28 | wall 7,380 tok/s | fid 0.17% | rc 20/20 | peak 27.43 | backing 102.2 |

+17.5% over bs32@28 (6,283); 68% of the 10.9k compute ceiling vs bs32's
54%. The sim's own decomposition of the bs32 rows (useful/recompute/
idle = 54/8/38% at dev-28; h2d 80-92% busy at every envelope; recompute
overhead only 5.5-7.6% everywhere) said the degradation is transfer-
exposed idle, not recompute cost — bs64ga1 confirms it by construction:
W streams ONCE, ctx saves ZERO (rc-20/20), fidelity collapses to 0.17%,
and throughput jumps 17.5%. Same lesson as olmoe, sharper: this family
is the round-restreaming grammar's clearest customer below 28 GiB.

FINAL qwen35moe-20l curve: 3,967 / 5,472 / 5,899 / **7,380** @ dev
16/20/24/28 (bs16ga4 / bs32ga2 / bs32ga2 / bs64ga1).

**dev-12 is structurally below this model's floor** — two independent
failure signatures across attempts (derivation timing deadlock streaming
1.9 GiB weight layers through an 8.4 GiB ledger; PressureFit boundary
unpackable at the shaved 6.65): the designed loud failures, not bugs.
Feasibility floor sits between 12 and 16.

**Backing-cap finding (m4_train --backing-gib)**: the cap pre-pins the
FULL capacity as one slab at startup, so near-RAM caps are unusable:
cap 130 < plan need (134-143 measured) => derivation infeasible; cap 155
> cudaHostAlloc lockable => startup crash. Plans genuinely need 127-143
GiB backing here (measured peaks). FOLLOW-UP (small): decouple the
planner's backing bound (sim-side plan rejection) from the runtime slab
size — a plan-only cap. Until then this model runs uncapped and host
safety rides on swap headroom + oomd (whose pressure-kills during
pinning storms are exactly what interrupted this session's runs).

No flextrain head-to-head for this family: their stock config is the
40L/35B (out of host reach on this box); a 20L variant on their side
would need a config hack. olmoe remains the cross-system comparison.


## Reading

- Planner goes recompute-HEAVY (35-58/64) — the MoE regime as designed:
  active compute is tiny (~1.2B) while the full expert stack streams;
  saved-ctx bytes trade badly against re-running cheap forwards.
- Fidelity 3-6% = the contention thermometer (M5.2): weight restreaming
  dominates h2d. The fewer-rounds shape (bs32ga2) attacks exactly this.
- real-vs-sim -2.7..-5.8% = host tax + the known aten grouped_mm
  out-copy tax (~5% of expert time; moe-design.md par 3 has the followup).
- 0 escapes, 0 evictions, every envelope verified.
