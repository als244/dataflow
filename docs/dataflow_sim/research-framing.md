# research framing

Paper-facing framing for the auto-policy work, written for systems venues
(MLSys / SOSP / OSDI class). Pairs with [problem.md](problem.md) (the exact
executable contract), [related-work.md](related-work.md) (formulation,
complexity, and the full literature map), and the
[policy catalog](policy/README.md).

## 1. Pitch

Training and inference stacks have complete forward visibility: task order,
tensor sizes, kernel runtimes, and link bandwidths are known before the
first kernel launches. Production memory management mostly ignores this —
UVM paging is reactive, ZeRO's offload splits are static rules, runtime
caches are LRU. `dataflow` treats data movement as a **compiled artifact**:
a planner reads a bare task chain and emits per-task release / offload /
prefetch directives; a deterministic event simulator (three FIFO streams,
continuous-time capacity including in-flight transfers) verifies and prices
every candidate plan; the engine executes the selected plan on real
hardware; and the sim-vs-real gap is tracked as a first-class fidelity
metric.

Headline claim shape: *at fast-memory budget B, planned movement sustains
X% of unconstrained-memory throughput, within Y% of a bound on any possible
plan, where reactive and rule-based baselines reach Z%* — X measured on the
real engine, Y from the oracle / lower-bound machinery, Z from baselines.

## 2. Why now

- Workloads crossed VRAM long ago (large-model fine-tuning on consumer
  GPUs, long context, MoE); every serious stack now ships an offload path.
- The systems lineage converged toward compile-time planning — vDNN's
  static heuristics (2016) → SwapAdvisor's search and AutoTM's ILP (2020)
  → G10's compile-time migration plans (2023) — yet no system *states* the
  optimization problem it heuristically solves, none models the duplex
  FIFO copy-engine timeline exactly, and none reports distance to optimal
  ([related-work.md §6.6](related-work.md)). The field built the machine;
  the planning problem is unclaimed.
- Bandwidth economics keep shifting (PCIe 4/5, C2C superchips, NVMe
  tiers). Hand-tuned policies rot; planners that re-derive schedules from
  stated constants don't.

## 3. Contributions the paper defends

1. **The problem as an executable contract.** A precise machine model —
   duplex one-transfer-per-direction FIFO streams overlapping serial
   compute, capacity charged from inbound-transfer start to
   outbound-transfer completion, mutation/write-back semantics, output +
   workspace reservation at task start — as a deterministic simulator that
   is both the planning cost model and the correctness gate, with the
   engine held to the same contract and the sim-vs-real gap measured
   ([problem.md](problem.md); precedent for the methodology: StarPU-on-
   SimGrid, [related-work.md §6.5](related-work.md)).
2. **PressureFit.** A workload-agnostic portfolio planner (residency-cut
   strategies × prefetch rules, every candidate simulator-verified,
   bounded work) that plans any ordered task chain — DNN training, RL
   pipelines, hand-built programs — with no model knowledge
   ([pressurefit.md](policy/pressurefit.md)).
3. **Oracle-gap methodology.** Exact oracle on small chains
   ([tools/bench](../../tools/bench/README.md)) plus a lower-bound ladder
   (compute, channel-volume, window, LP/flow —
   [related-work.md §5](related-work.md)) at scale, so quality is reported
   as *distance from any possible plan*, not policy-vs-policy deltas.
   Checkmate did this for rematerialization; nobody has for offload.
4. **A layered system.** Recompute selection as a simulator-priced
   pre-pass, planning on the object-only capacity, static-arena
   physicalization beneath, checkpoint/resume and data-parallel fleets
   around — each layer separately testable, with a parity ladder
   (`verify_family`) pinning numerics against plain autograd.

## 4. Decision axes

The planner has eight degrees of freedom, not three:

1. **Initial placement** — fast/backing residency at t=0.
2. **Release** — drop a fast copy whose value is dead or backing-backed.
3. **Offload** — write-back (semantically required for mutated state).
4. **Prefetch** — reload before next use.
5. **Trigger boundary** — *which* task completion fires each directive;
   late triggers minimize occupancy, early ones give streams slack.
6. **Stream FIFO order** — co-fired transfers unblock different downstream
   tasks depending on queue order.
7. **Cascades** — "issue after that offload lands" dependencies, not just
   "issue at boundary k".
8. **Recomputation level** `k_o ∈ {0..K}` — per-intermediate point on a
   (stored-size, recompute-time) curve; generalizes binary checkpointing.
   Chosen by the recompute pre-pass in the current architecture.

Axes 5–7 are the ones a "what to evict / what to load" framing collapses;
they are where stalls actually appear or vanish at tight budgets.

## 5. Positioning against prior systems

Full map with citations: [related-work.md §6–7](related-work.md). The
review-table version:

| System | Plan time | Mechanism | Duplex FIFO timeline | In-flight cap accounting | Mutation / write-back | Output+workspace reservation | Optimality gap reported | Beyond-DNN programs |
|---|---|---|---|---|---|---|---|---|
| vDNN (MICRO'16) | offline | layer heuristics | – | – | – | – | – | – |
| SwapAdvisor (ASPLOS'20) | offline | GA over a simulator | ≈ (sim has both channels) | – | partial | allocator co-search | – | – |
| AutoTM (ASPLOS'20) | offline | ILP | – (per-kernel slots) | – | partial | – | – | – |
| Capuchin (ASPLOS'20) | online | profiled benefit rules | – | – | partial | – | – | – |
| FlashNeuron (FAST'21) | offline | overlap-aware selection | – (one NVMe path) | – | – | – | – | – |
| ZeRO-Infinity (SC'21) | static | partitioning rules | – | – | ✓ (fixed) | – | – | – |
| G10 (MICRO'23) | offline | greedy vs bandwidth-over-time | ≈ | – | partial | partial | – | – |
| DeepUM (ASPLOS'23) | online | page correlation prefetch | – | – | – | – | – | – |
| FlexGen (ICML'23) | offline | LP over block schedules | modeled | – | ✓ | – | – | – (inference loop) |
| StarPU + DARTS (IPDPS'22–JPDC'25) | online | data-yield greedy + Belady-adapted eviction | ✓ (runtime) | ✓ (runtime) | ✓ | ✓ | – | ✓ (task graphs) |
| POFO (NeurIPS'21) | offline | DP (chain, restricted class) | – (one channel) | – | ✓ | – | within its class | – |
| **dataflow** | offline | portfolio + simulator verification | ✓ | ✓ | ✓ | ✓ | ✓ (oracle + LBs) | ✓ (any ordered chain) |

No prior row fills the four middle columns at once, and none fills the
optimality-gap column for offload at all — that conjunction is the paper's
territory. (Closest kin, and worth generous citation: G10's planner
independently derived a latest-safe-then-advance prefetch rule; SwapAdvisor
independently chose search-over-a-simulator; the StarPU line independently
adapted Belady to task sets.)

## 6. What the theory buys (support, not the story)

- **Hardness licenses the architecture.** Offloading on a chain with
  overlapped transfers is strongly NP-hard (Beaumont et al., Euro-Par
  2020); variable-size caching is strongly NP-hard even with instantaneous
  transfers; the transfer-sequencing core alone is strongly NP-complete
  ([related-work.md §4.2](related-work.md)). Exact solvers cap out at toy
  scale, so *heuristics + simulator verification + oracle gap reporting*
  is the defensible design, not a compromise.
- **Tractable islands supply principled ingredients.** Belady/KTNS for
  set-valued eviction keys, the aggressive-vs-conservative prefetching
  spectrum (the portfolio's `latest-safe` … `packed-fifo` axis, named and
  bounded in the classical literature), min-cost-flow lower bounds, and
  the prefetch/write-back reversal duality
  ([related-work.md §6.1–6.2, §8](related-work.md)).
- **Feasibility is exact and cheap.** Let
  `W = max_i Σ_{o ∈ in(T_i) ∪ out(T_i)} s(o)` be the widest single-task
  footprint. If `C_fast ≥ W` (and backing has room), a feasible schedule
  exists — evict everything between tasks, reload per task; the bound is
  tight since `C_fast < W` makes that task unrunnable. This is the
  engine's admission guarantee.
- **Open question (kept, correctly scoped).** Is Belady-style greedy with
  stream-load modulation within one transfer of optimal *on structured
  training chains* (nested lifetimes, ≤2 uses per object)? Unrestricted,
  such a bound collides with caching hardness
  ([related-work.md §4.3](related-work.md)); on the structured subclass it
  is open. A theory bonus if it lands; the paper does not depend on it.

## 7. Evaluation plan

The three-tool workflow ([benchmarking.md](../benchmarking.md)) is the
evaluation harness; the paper's tables come from it.

- **Workloads.** Builtin families (GPT-2 → DeepSeek-class MoE), the RL
  training example, and at least one hand-built non-training chain to
  substantiate generality.
- **Sweeps.** Geometry × budget grids (`predict_step` /`measure_step`):
  budgets from comfortable to just above the feasibility bound `W`;
  per-cell s/step, tok/s, stall %, link utilization %, recompute %.
- **Baselines.** Internal: the policy catalog (sliding-window hand policy,
  belady_reactive, max_reduce, min_grow) — an ablation of planning
  sophistication. External: unplanned execution (UVM/managed-memory mode
  and a keep-all-OOM upper envelope), a ZeRO-style static split, and a
  layer-streaming window (L2L/STRONGHOLD-shaped) expressed as chains in
  the simulator; G10-style greedy as a policy port if time permits.
- **Optimality.** Exhaustive oracle on small chains; LP/flow +
  channel-volume lower bounds at scale ([related-work.md §5](related-work.md));
  report `makespan / LB` for every headline cell.
- **Fidelity.** Sim-predicted vs measured timelines diffed per cell
  (`trace_real_run` + webapp panels); state the gap distribution, and use
  Nsight captures to attribute residuals (copy-engine contention, allocator
  jitter, kernel variance).
- **Artifact story.** Deterministic simulator, exported programs/plans,
  the webapp replay, and the `verify_family` parity ladder make the whole
  paper replayable without GPUs except for the measured columns — a strong
  artifact-evaluation narrative.

## 8. Honest limits

- **Fixed task order.** MODeL/SwapAdvisor show reordering buys memory;
  the chain contract buys determinism, checkpointing, and parity testing.
  State the trade; reordering is future work.
- **Single-accelerator planning.** Fleets are data-parallel around the
  planner; model-parallel offload (Mobius/Harmony territory) is out of
  scope.
- **Strict per-direction FIFO.** Real copy engines allow more (TURNIP
  deliberately exploits reordering); FIFO is a modeling choice that keeps
  plans executable and deterministic — quantify what it costs via the
  simulator when relaxing it.
- **Fragmentation delegated.** The planner treats capacity as fungible;
  the static-arena physicalizer pays the packing bill and feeds back a
  reserve when it can't ([pressurefit.md](policy/pressurefit.md)); DSA
  theory says the bill is small when objects ≪ B
  ([related-work.md §6.1](related-work.md)).
- **Cost-model risk.** Profiled runtimes assume shape-stable kernels;
  data-dependent control (MoE routing, sparse attention) is pinned via
  `AuxTemp` objects so recompute replays choices verbatim — but
  data-dependent *sizes* remain future work.
- **Host-side effects.** Pinned-buffer pressure and CPU memory-bandwidth
  interference shift measured bandwidths from nominal; calibrate constants
  from measurement, not spec sheets.

## 9. Open extensions

Online/dynamic shapes; NVMe third tier and multi-path channels
(MLP-Offload-style) which relax one-transfer-per-direction; joint
recompute+movement in a single solve (replacing the layered pre-pass);
task reordering within dependence slack; multi-accelerator model
parallelism; approximation guarantees on the structured subclass
([related-work.md §4.3](related-work.md)).

---

*This live doc is the framing a reader needs alongside
[problem.md](problem.md), [related-work.md](related-work.md), and the
[policy catalog](policy/README.md).*
