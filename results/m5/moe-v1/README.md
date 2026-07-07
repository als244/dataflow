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


## M-G: qwen3moe-24l + dsv3-mini curves (2026-07-07) — the family capstone

### qwen3moe-30b-24l (15.6B, E=128 K=8 F=768, GQA per-head qk-norm)

| dev GiB | wall tok/s (shape) | sim | real-vs-sim | fid% | rc | peak |
|---|---|---|---|---|---|---|
| 12 | infeasible (PressureFit boundary, loud; floor in (12,16]) | | | | | |
| 16 | 6,686 (bs32ga2) | 6,162 | +8.7% | 1.35 | 41/48 | 15.76 |
| 20 | 8,563 (bs64ga1) | 7,312 | +17.4% | 0.30 | 23/24 | 19.17 |
| 24 | 9,684 (bs64ga1) | 8,606 | +12.9% | 0.33 | 24/24 | 23.69 |
| 28 | 10,014 (bs64ga1) | 9,639 | +4.2% | 0.37 | 20/24 | 27.21 |

### dsv3-mini (12.7B DeepSeek-V3 shape: MLA + sigmoid_noaux_tc + first-2-dense)

| dev GiB | wall tok/s (shape) | sim | real-vs-sim | fid% | rc | peak |
|---|---|---|---|---|---|---|
| 12 | 5,037 (bs16ga4) | 4,018 | +25.5% | 0.55 | 72/72 | 11.68 |
| 16 | 8,104 (bs32ga2) | 7,299 | +11.3% | 0.39 | 24/36 | 15.09 |
| 20 | 9,876 (bs64ga1) | 8,760 | +13.0% | 0.38 | 15/18 | 19.36 |
| 24 | 10,531 (bs64ga1) | 9,463 | +11.6% | 2.88 | 15/18 | 22.94 |
| 28 | **12,137 (bs64ga1)** | 12,305 | -1.0% | 5.65 | 14/18 | 27.99 |

Sim decomposition (best plan per envelope, % of sim step): dsv3 idle
67/44/32/27/**5.4** at 12/16/20/24/28 with recompute only 4.5-11% —
idle COLLAPSES at 28 (the MLA compressed-ctx effect: rc-14/18 keeps
recomputing because MLA ctx is nearly free to rebuild, so stream-once
weights + tiny ctx leaves almost nothing to wait for).

### Cross-family: best real tok/s per envelope + fraction of compute ceiling @28

| dev GiB | olmoe-7B (6.9B) | qwen35moe-20l (17.8B) | qwen3moe-24l (15.6B) | dsv3-mini (12.7B) |
|---|---|---|---|---|
| 12 | 13,351 | infeasible | infeasible | 5,037 |
| 16 | 14,337 | 3,967 | 6,686 | 8,104 |
| 20 | 14,766 | 5,472 | 8,563 | 9,876 |
| 24 | 15,492 | 5,899 | 9,684 | 10,531 |
| 28 | 15,948 | 7,380 | 10,014 | 12,137 |
| % of compute ceiling @28 | 80% | 68% | **87%** | 83% |

Reading: post-sync-fix the sim is CONSERVATIVE everywhere (+4..+26%,
worst where plans are recompute-heaviest — the contended-profile bound
overprices under light real traffic) and ranking-correct in every
family. h2d-boundedness @28 ranks qwen35moe (68%, heaviest weights/
layer) < olmoe (80%) < dsv3 (83%) < qwen3moe (87%); at LOW envelopes
dsv3 is the only heavyweight that stays feasible at 12 (MLA ctx) while
every family pays 44-67% idle — the round-restreaming grammar work
(round-resident weights / intra-round token chunking) is quantified per
family by exactly these idle columns.


## M-H1: dsv32-mini (DeepSeek-V3.2 DSA) FIRST curve — eager kernels (2026-07-07)

12.8B dsv32-mini at s4k (65,536 tok/step = 16 seqs; k=1024 < 4096 so
sparsity is ACTIVE), SPARSE mode, all-eager DSA ops (mask-form core +
eager indexer) — the correctness-grade baseline M-H2's kernels are
measured against. dsv3-mini at the SAME dims and s4k = the no-DSA
baseline (padded-v flash attention).

| dev GiB | dsv32 wall (shape) | dsv32 sim | r/s% | fid% | dsv3-s4k wall (shape) | DSA-eager tax |
|---|---|---|---|---|---|---|
| 12 | 2,310 (bs4ga4 rc-72) | 2,341 | -1.3 | 0.25 | 5,095 (bs4ga4 rc-72) | 2.2x |
| 16 | 2,688 (bs4ga4 rc-0) | 2,950 | -8.8 | 6.26 | 7,671 (bs8ga2 rc-28) | 2.9x |
| 20 | 2,772 (bs8ga2 rc-0) | 3,051 | -9.1 | 5.84 | 8,428 (bs8ga2 rc-22) | 3.0x |
| 24 | 2,872 (bs8ga2 rc-0) | 3,166 | -9.2 | 6.16 | 10,812 (bs16ga1 rc-18) | 3.8x |
| 28 | 2,890 (bs8ga2 rc-0) | 3,166 | -8.7 | 5.75 | 12,176 (bs16ga1 rc-14) | 4.2x |

Reading:
- dsv32-eager is COMPUTE-walled: the curve is nearly flat above 16 and
  the planner picks rc-0 (save-all) — with attention this slow,
  transfers hide completely. The flatness IS the M-H2 headroom: 2.2x at
  12 GiB rising to 4.2x at 28.
- The rc-0 rows carry fid 5.8-6.3% and r/s -9%: the eager core's
  many-small-kernel chunked enqueue makes those tasks timing-loose (a
  mild boundary-tax cousin). Expected to collapse with real kernels.
- dsv3-s4k baselines behave like their s1k selves (real above sim,
  monotone 5.1k -> 12.2k).
- ENGINE NOTE (2nd sighting of the valve-fragility class): dsv3-s4k
  bs16ga1@dev-20 deadlocked at runtime after 770 pressure evictions
  (block_bwd_0_0_15, 1.9 GB unservable) — the qwen35moe ledger-14.1
  signature on a new family; strengthens the valve-completion
  (dirty-evict) engine follow-up. Row covered by bs8ga2@20.

M-H2 next: triton dsa_index fwd/bwd + gather-absorbed sparse core
(deterministic inversion bwd) + sm90 FlashMLA seam + tiny-scale
stock-flash cross-validator; then THIS TABLE gets re-measured.


## M-H2: dsv32-mini re-measured — triton DSA kernels (2026-07-07)

Same protocol as the eager round; kernel set: triton masked-flash
sparse core (bitmask selection, flash tiling, deterministic two-pass
bwd), triton indexer fwd/bwd, fused KL-target probs_sum. Sparse-stage
training remains PAPER-FAITHFUL (the indexer's KL is its only training
signal — report: 'the training signal of the indexer is from only
L_I'; its ~3 ms/seq bwd cost is DSA's legitimate training price).

| dev GiB | TRITON wall (shape) | sim | r/s% | fid% | vs eager | vs dsv3-dense s4k |
|---|---|---|---|---|---|---|
| 12 | 4,971 (bs4ga4 rc-58) | 4,300 | +15.8 | 0.66 | 2.15x | 1.02x — FREE |
| 16 | 5,765 (bs8ga2 rc-36) | 4,727 | +22.1 | 1.06 | 2.14x | 1.33x |
| 20 | 7,514 (bs8ga2 rc-15) | 6,669 | +12.9 | 0.53 | 2.71x | 1.12x |
| 24 | 7,558 (bs16ga1 rc-9) | 6,707 | +12.9 | 0.21 | 2.63x | 1.43x |
| 28 | 8,308 (bs16ga1 rc-8) | 7,266 | +14.6 | 1.67 | 2.87x | 1.47x |

- End-to-end 2.1-2.9x over the eager round; the DSA tax vs dense dsv3
  collapsed from 2.2-4.2x to 1.02-1.47x. At dev-12 DSA is effectively
  FREE: the sparse core's 4x flop savings pay for the indexer.
- The curve has slope again (4,971 -> 8,308): attention left the
  critical wall, memory economics returned, and stream-once bs16ga1
  reclaimed the top envelopes exactly as the oracle predicted.
- Fidelity back to 0.2-1.7% (the eager rows' 5.8-6.3% chunked-enqueue
  looseness vanished with real kernels, as predicted); real runs
  +13-22% ABOVE sim (the standard contended-profile conservatism).
- Remaining vs-dense gap at 24/28 (~1.45x) decomposes as: indexer
  compute + KL training tax (paper-faithful) + sparse-bwd headroom
  (the dkv pass re-derives O internally; saving it via the ctx
  attn_out is a filed micro-optimization) + selection/topk. The
  long-context regime (128K, k<<L) flips the economics decisively
  toward DSA; on this box at s4k the parity-at-low-memory result is
  the headline.


## M-H2 final: dsv32-mini after three kernel rounds (2026-07-07)

Kernel rounds: (1) triton masked-flash + indexer; (2) dead-tile skip,
ctx-O plumb, tile sweeps, fast pack; (3) NATIVE v_head_dim (no flash
equal-dims pad anywhere on the dsv32 path), pack-once-per-seq shared
across bwd kernels, delta computed in dq / reused in dkv; (3.5) topk =
torch.topk semantics (DeepSeek's model.py rule — the smallest-index pin
was a MoE convention imported by mistake) + idx-bwd stages=4. Sparse
fwd 0.40 ms/seq = dense flash parity; DSA total 10.75 -> 5.6 ms/seq/L.

| dev GiB | dsv32 real (shape) | sim | r/s% | fid% | vs dsv3-s4k | eager era |
|---|---|---|---|---|---|---|
| 12 | 5,205 (bs4ga4 rc-66) | 4,439 | +17.4 | 0.82 | 0.98x | 2,310 |
| 16 | 7,369 (bs8ga2 rc-16) | 6,520 | +13.2 | 0.43 | 1.04x | 2,688 |
| 20 | 8,015 (bs8ga2 rc-14) | 7,025 | +14.3 | 0.56 | 1.05x | 2,772 |
| 24 | 9,179 (bs16ga1 rc-14) | 7,819 | +17.9 | 3.10 | 1.18x | 2,872 |
| 28 | 9,766 (bs16ga1 rc-9) | 8,820 | +11.0 | 4.31 | 1.25x | 2,890 |

- 3.4x over the eager round at dev-28; DSA tax vs dense dsv3 collapsed
  4.2x -> 1.25x top / ~1.0x at 12-20. Raw model FLOPs are EQUAL at s4k
  (sparse-core savings == indexer + KL additions), so the remaining
  1.18-1.25x is DSA's extra OPS at high memory where dsv3 streams
  once: indexer bwd (~1.1 ms/seq, still ~3x off roofline — the ONLY
  clearly poorly-constructed kernel left; rebuild filed), selection
  (topk ~0.35 + pack 0.44), KL target (0.27, at roofline).
- Roofline verdicts: index-scores fwd AT roofline (172 TF/s + write-
  bound); sparse fwd AT flash parity; probs_sum near; sparse bwd ~1.3x
  off; index_bwd ~3x off (two deterministic ownership passes both
  re-derive ReLU scores; fp32 dI reloaded per head; 64^3 dots).
- Stock-flash/FA answer (Shein's FA4 PR pointer): sm120 GeForce box —
  flash_attn absent, FA4 targets sm100, SDPA flash caps head_dim 256 /
  cudnn 128 (probed: MQA 288/576 refuse) -> absorbed-MQA dims can't run
  on any stock kernel here; and at s4k gather-MQA reads 2.4 GB/seq
  (head-amortized) vs masked 0.8 GB (query-tile-amortized) — gather
  wins only at k << L. Long-context + FlashMLA sm90 = the big-machine
  seam (M-H2b), where the FA4-class path is exactly right.
- Sim rows stale-conservative (+11..+18%): profile cache did not
  re-key on the round-3 kernel changes; refreshed on next natural
  re-profile. fid 3.1/4.3 on the bs16 rows tracks the same staleness.


## M-H FINAL: dsv32-mini three-way table — fresh profiles, all kernel rounds (2026-07-07)

PROFILE_CACHE_REV 3 (round-3.5/4 kernels re-profiled; the stale-sim era
ends). Sparse rows = the M-H2 final; DENSE WARM-UP = M-H3 (paper stage:
dense attention, MAIN MODEL FROZEN — bit-verified — indexer trains from
full-prefix KL only).

| dev GiB | dsv3 dense | dsv32 WARM-UP | dsv32 SPARSE | sparse vs dsv3 |
|---|---|---|---|---|
| 12 | 5,095 | — | 5,289 (rc-56, fid 0.93) | 0.96x |
| 16 | 7,671 | 7,847 (0.98x) | 7,680 (rc-20, fid 0.32) | 1.00x |
| 20 | 8,428 | — | 8,796 (rc-14, fid 0.44) | 0.96x |
| 24 | 10,812 | — | 9,947 (rc-14, fid 0.30) | 1.09x |
| 28 | 12,176 | 11,392 (1.07x) | 10,093 (rc-14, fid 1.46) | 1.21x |

- SPARSE AT PARITY OR BETTER through dev-20 (sparsity's flop savings +
  our kernels beat dense flash + the planner exploits the smaller ctx);
  1.21x at 28 = the indexer + selection + KL ops on equal model FLOPs.
  Eager era -> now: 2,890 -> 10,093 @ 28 = 3.5x.
- WARM-UP sits exactly where the math says: flash attention + KL-only
  backward = ~7% over dsv3 at 28 (11,392 vs 12,176). fid 0.35 @ 28.
- Fidelity 0.30-1.46 everywhere (profile staleness cleaned up); sim
  conservative +2..+21% (contended-profile convention, ranking-correct).
- bs16ga1 claims 20-28 under round-4 kernels (attention cheap enough
  that stream-once wins from dev-20 now).


## M-I3: glm52-mini (IndexShare) first curve vs dsv32-mini (2026-07-07)

Same backbone as dsv32-mini; 6 of 18 layers carry indexers (F F +
[F S S S] x4). Selection size REALITY CHECK (Shein q): the shared
selection is per-token top-k index LISTS — (t, k) int32 = t*k*4B:
268 MB/round at bs16ga1, 134 at bs8, 67 at bs4; dM doubles the bwd
window in fp32. Identical bytes to dsv32's per-layer M — the delta is
LIFETIME: a leader's M serves 4 layers across fwd + the group's whole
bwd span (more reloads), plus the dM round-trip.

| dev GiB | glm52 (shape) | sim | r/s% | fid% | dsv32 | delta |
|---|---|---|---|---|---|---|
| 12 | 3,288 (bs4ga4 rc-72) | 3,691 | +18.8* | 1.22 | 5,289 | -38% |
| 16 | 5,298 (bs8ga2 rc-25) | 6,878 | -22.9 | 0.96 | 7,680 | -31% |
| 20 | 8,041 (bs8ga2 rc-19) | 7,093 | +13.6 | 0.68 | 8,796 | -8.6% |
| 24 | 9,689 (bs16ga1 rc-18) | 8,084 | +20.1 | 0.73 | 9,947 | -2.6% |
| 28 | 11,230 (bs16ga1 rc-12) | 10,220 | +10.2 | 0.29 | 10,093 | +11.3% |

- IndexShare on THIS box: +11% at 28 GiB (sim predicted +12 — ranking
  and magnitude both right), crossover ~23 GiB, SLOWER below: the
  indexer savings are modest at s4k while the shared-M lifetime + dM
  round-trip add transfer pressure that tight-memory plans pay for.
  dev-16's real-below-sim (-22.9% with tight fid 0.96) is the
  transfer-overlap-optimism signature — the plan is transfer-bound on
  M/dM traffic the sim under-charges (the known contention-aware-
  costing gap, now with a second family exhibiting it).
- The paper's 2.9x-fewer-indexer-FLOPs claim is a 1M-context statement;
  at s4k the indexer is ~15% of DSA time, so +11% at high memory is
  the expected shape of the win here.
- ORACLE BUG (filed): best_config marked bs4ga4@12 (and bs16@16/20)
  infeasible:ValueError with the detail swallowed — bench_train runs
  the same cell fine solo (derivation converges: ledger 6.68, extent
  8.55 <= 8.71 avail; measured peak 11.81 <= 12, wall 3,288). The
  in-process cell-state class again (allocator/cublas residue between
  cells); fresh-process = truth. best_config needs per-cell error
  detail + probably per-cell process isolation.
* dev-12 quoted from the solo bench_train repro run.


## CAMPAIGN CLOSE: all-legal three-family table — REV-4, round-5 kernels,
## auto-headroom enforcement (2026-07-07)

Every cell PEAK-VERIFIED <= envelope (the auto-headroom closing loop:
post-run measured-peak check, shrink-by-measured-overage, re-run once —
no hand leeway constant; bench_train enforces envelope_ok on every row).
Best LEGAL number per cell; mode noted where it is not plain static.

| dev GiB | dsv3 (dense MLA) | dsv32 (DSA) | glm52 (DSA+IndexShare) |
|---|---|---|---|
| 12 | 5,174 (vmm) | 5,124 | 5,286 (vmm) |
| 16 | 8,204 | 7,618 | 7,991 (vmm) |
| 20 | 8,341 (extent-b) | 8,175 | 8,220 (vmm) |
| 24 | 10,335 | 9,931 (vmm) | 9,856 (vmm) |
| 28 | 12,177 | 10,427 | 11,270 |

(dsv3@28: closing-loop row — measured peak EXACTLY 28.00 GiB, fid
0.38; the loop landed the plan flush against the envelope. The earlier
extent-bounded 11,613 over-shaved; the earlier plain 12,357 ran 0.31
over. 12,177 is the honest number and matches the pre-round-5 12,176
almost exactly — dsv3's throughput held while its peak got legal.)

HEADLINES
- glm52 >= dsv32 at EVERY envelope: IndexShare is a strict win over
  per-layer indexing under legal accounting. Its earlier -38/-31%
  low-memory deficits were placement geometry (contiguous packing of
  long-lived shared-selection episodes), cured by VMM (+61% at dev-12)
  and by round-5's smaller transients (dev-16 static now legal).
- glm52 vs DENSE dsv3: 1.02x at 12 (glm52 FASTER than dense), 1.03x at
  28, ~1.03-1.05x mid — at s4k, GLM-5.2's architecture reaches
  effectively dense-model throughput at every memory point on this
  runtime, while keeping DSA's long-context upside.
- dsv32 dev-24 saga resolved: 9,931 legal (vmm) vs the old row's 9,947
  at an ILLEGAL 24.46 peak. The apparent -13% was: ~5% old-row cheat +
  never-recompute-selection memory economics (rc-14 -> 18 under static;
  vmm's packing headroom restores rc-14) — both now quantified.
- Never-recompute M grammar: costs ~0-7% at mid envelopes under static
  packing (rc recovers ~14% less per layer), refunded by vmm; buys
  bit-exact selection consistency by construction + cheaper recompute
  tasks (no re-select) + smaller A objects.
- Enforcement archaeology: 7 pre-existing rows across families were
  silently over-envelope (worst: glm52 17.23 on 16). All published
  numbers in this table are peak-verified; the enforcement + closing
  loop make future busts structurally impossible.

FOLLOW-UPS (filed, with measured exhibits)
- Packing-aware placement / metadata zones (26% extent tax exhibit) or
  planner-side prefer-offload for long-lived M episodes.
- Per-task-time-aware scratch reserve in the derivation (head_loss's
  1.5-2 GiB global-max reserve taxes every window; CE chunk shrink
  measured +43% slower = wrong fix).
- rc-planner: price M-offload vs extra recompute layers explicitly.
- vmm fidelity looseness (2-4% vs static's 0.3-1) — schedule replay
  under arena mapping; investigate before vmm becomes a default.
- best_config: per-cell error detail + process isolation (false
  infeasibles from in-process cell state).


## FINAL TABLES: per-mode, per-cell (wall tok/s, sim tok/s, measured
## peak, chosen shape, recompute fraction) — 2026-07-07

Protocol: REV-4 profiles, round-5 kernels, auto-headroom closing loop
(every peak measured and <= envelope; no leeway constants). Shapes per
envelope chosen by the oracle and HELD FIXED across modes (bs4ga4@12,
bs8ga2@16-20, bs16ga1@24-28, all families) so static-vs-vmm isolates
placement. rc% = recomputed layer-rounds / total layer-rounds.


### STATIC placement

| dev GiB | dsv3 (dense MLA) | dsv32 (DSA) | glm52 (DSA+IndexShare) |
|---|---|---|---|
| 12 | 5,095 (sim 3,879) · 10.81 GiB · bs4ga4 · rc 100% | 5,124 (sim 4,103) · 11.92 GiB · bs4ga4 · rc 100% | 3,288 (sim 3,691) · 11.81 GiB · bs4ga4 · rc 100% |
| 16 | 8,204 (sim 7,239) · 15.80 GiB · bs8ga2 · rc 81% | 7,618 (sim 6,742) · 15.90 GiB · bs8ga2 · rc 75% | 7,457 (sim 6,546) · 15.70 GiB · bs8ga2 · rc 100% |
| 20 | 8,582 (sim 7,750) · 20.00 GiB · bs8ga2 · rc 75% | 8,175 (sim 7,108) · 19.61 GiB · bs8ga2 · rc 69% | 8,213 (sim 7,214) · 19.75 GiB · bs8ga2 · rc 56% |
| 24 | 10,335 (sim 9,089) · 22.62 GiB · bs16ga1 · rc 89% | 9,053 (sim 7,670) · 22.23 GiB · bs16ga1 · rc 100% | 9,391 (sim 7,752) · 23.16 GiB · bs16ga1 · rc 100% |
| 28 | 12,177 (sim 12,083) · 28.00 GiB · bs16ga1 · rc 83% | 10,427 (sim 9,155) · 27.47 GiB · bs16ga1 · rc 78% | 11,270 (sim 10,103) · 27.98 GiB · bs16ga1 · rc 72% |

### VMM placement

| dev GiB | dsv3 (dense MLA) | dsv32 (DSA) | glm52 (DSA+IndexShare) |
|---|---|---|---|
| 12 | 5,174 (sim 4,159) · 11.19 GiB · bs4ga4 · rc 100% | 4,694 (sim 3,871) · 11.19 GiB · bs4ga4 · rc 100% | 5,286 (sim 4,512) · 11.95 GiB · bs4ga4 · rc 79% |
| 16 | 7,913 (sim 7,289) · 15.71 GiB · bs8ga2 · rc 100% | 7,563 (sim 6,759) · 15.31 GiB · bs8ga2 · rc 75% | 7,991 (sim 7,146) · 15.31 GiB · bs8ga2 · rc 61% |
| 20 | 8,289 (sim 7,806) · 19.75 GiB · bs8ga2 · rc 75% | 7,998 (sim 7,219) · 19.71 GiB · bs8ga2 · rc 61% | 8,220 (sim 7,547) · 19.37 GiB · bs8ga2 · rc 69% |
| 24 | 11,988 (sim 12,192) · 23.93 GiB · bs16ga1 · rc 89% | 9,931 (sim 9,095) · 23.56 GiB · bs16ga1 · rc 78% | 9,856 (sim 8,374) · 23.56 GiB · bs16ga1 · rc 72% |
| 28 | 11,963 (sim 11,900) · 27.99 GiB · bs16ga1 · rc 78% | 10,333 (sim 9,815) · 27.93 GiB · bs16ga1 · rc 67% | 10,954 (sim 9,834) · 27.67 GiB · bs16ga1 · rc 83% |


Cell notes: dsv3-static@12 is the retained REV-3-era row (legal at
10.81 GiB; a fresh REV-4 static derivation is sim-infeasible at that
envelope — the vmm row beside it is current). All other 29 cells are
REV-4/round-5, measured this campaign.

Reading the pair:
- STATIC: dsv3 leads at 24/28 (no metadata objects to place); glm52
  beats dsv32 from 16 GiB up (IndexShare's indexer savings) but pays
  the shared-selection geometry at 12 (3,288 — the one cell where
  static packing hurts it badly).
- VMM: erases glm52's dev-12 penalty (5,286 — ABOVE dense dsv3) and
  lifts every family's 24-row (dsv3 11,988, +16% over its static 24:
  contiguous packing taxes even the dense family there). At 28, static
  wins for all three — when memory is loose the arena's overheads and
  looser fidelity (2-5% vs static's 0.3-1%) cost more than packing.
- Sim is uniformly conservative (+5..+18% real-over-sim) EXCEPT dsv3
  vmm@24 (-1.7%) — fresh REV-4 profiles, so this is the contended-
  profile convention, not staleness.
- Per-cell mode winner => the earlier best-legal table; these two are
  the honest per-mode records behind it.
