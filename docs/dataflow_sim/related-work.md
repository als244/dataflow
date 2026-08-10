# The planning problem as mathematics, and its neighbors in the literature

This note does three things:

1. states the `dataflow_sim` planning problem as a mathematical program
   (§1–§3), in a form that connects to standard scheduling/caching notation;
2. maps its complexity landscape — which special cases are polynomial, which
   are NP-hard, and where the hardness actually comes from (§4–§5); and
3. surveys the related research literature across the five communities that
   have studied pieces of this problem, with pointers to what each community's
   results do and do not cover (§6–§9).

It pairs with [problem.md](problem.md) (the concrete simulator contract and
solver-formulation options) and [research-framing.md](research-framing.md)
(decision axes and positioning). Recomputation planning — the layer that
chooses task runtimes and object sizes — is deliberately **out of scope**
here; tasks and sizes are fixed, as in [problem.md](problem.md). See
[recompute.md](recompute.md) for that layer.

---

## 1. Instance

An instance is `(O, s, λ⁰, λᶠ, T, B, B_H, β_in, β_out)`:

| Symbol | Meaning |
|---|---|
| `O` | finite set of objects, sizes `s_o > 0` (bytes) |
| `λ⁰(o) ∈ {fast, backing}` | initial location (given for the bare chain; pre-placement of backing-init objects may also be a decision, see §2) |
| `λᶠ(o) ∈ {backing, –}` | terminal requirement: latest bytes durably on backing, or disposable |
| `T = (τ_1 … τ_n)` | tasks in **fixed total order** (no reordering) |
| task `τ_t` | reads `X_t ⊆ O`, mutates `M_t ⊆ O` (in place), creates `Y_t ⊆ O` (fresh ids), uses opaque workspace `w_t ≥ 0` bytes, runs for `R_t > 0` |
| `B` | fast-memory (GPU) capacity; `B_H` backing capacity |
| `β_in, β_out` | bandwidths; transfer durations `δ_in(o) = s_o/β_in`, `δ_out(o) = s_o/β_out` |

Derived notation: `A_t = X_t ∪ M_t` (must be live in fast memory at start of
`τ_t`), `q_t = Σ_{o∈Y_t} s_o + w_t` (bytes reserved at start), and for each
object the ordered use list `uses(o) = {t : o ∈ A_t ∪ Y_t}`, with
`birth(o)` its creating task (or 0 for initial objects) and `death(o)` its
last use (or `n+1` when `λᶠ(o) = backing`).

**Machine semantics** (exactly the simulator contract of
[problem.md §1](problem.md) / PRESSUREFIT_ALGO.md §A.2): one serial compute
stream executing `τ_1 … τ_n` in order; one inbound and one outbound transfer
stream, each single-server FIFO and non-preemptive; all three overlap. An
object occupies fast memory from **inbound-transfer start** (not enqueue) or
creation-reservation, until **release instant** or **outbound-transfer
completion** (the "to-slow tail"). Dirty semantics: an object whose latest
bytes exist only in fast memory (created, or mutated since last write-back)
must complete an outbound transfer before its fast copy may be dropped;
clean objects may be dropped by a free `release`.

## 2. Decision variables

A **plan** consists of three coupled decisions:

1. **Initial placement** `P₀ ⊆ {o : λ⁰(o) = backing}` — backing-init objects
   pre-placed in fast memory at `t = 0` (cost-free; they are clean), subject
   to `Σ_{o∈P₀∪Fast₀} s_o ≤ B`.
2. **Residency** — for every object `o` and every gap `g = (u, u′)` between
   consecutive uses: `keep_{o,g} ∈ {0,1}`. `keep = 0` induces
   an eviction (a `release` if a valid backing copy exists, otherwise an
   **offload event**) and a **prefetch event** that must complete before
   `σ_{u′}`. The same applies to the initial gap (`λ⁰ = backing`, first use)
   and the terminal gap (`last use, λᶠ = backing` — offload occurrence forced,
   timing free).
3. **Transfer schedule** — for every induced transfer event: an enqueue
   boundary `φ(e) ∈ {0 … n}` (directives fire at task completions) and a
   queue position among same-direction events; equivalently, per direction, a
   non-preemptive sequence.

Given (1)–(3) the timeline is fully determined by the deterministic
event-driven semantics; makespan `M` and per-task stalls are outputs of
`Simulate`. That is the *operational* form. The *mathematical-program* form
below replaces the simulator with explicit timing variables, which is what a
MILP/CP solver (or a proof) manipulates.

## 3. The mathematical program

Continuous variables: task start times `σ_t ≥ 0`, transfer start times
`τ_e ≥ 0` for every event `e` that the residency decisions induce, release
instants, and makespan `C`.

```
minimize  C                            (equivalently: stall S = C − Σ_t R_t)

subject to
 (C1) serial compute, fixed order:     σ_{t+1} ≥ σ_t + R_t
 (C2) availability:                    τ_e + δ_in(o) ≤ σ_u        for the prefetch e of o serving use u
 (C3) writeback / source validity:
        offload starts after last write:   τ_f ≥ σ_p + R_p        (p = producer/last mutator before the gap)
        reload starts after backing valid: τ_e ≥ τ_f + δ_out(o)   (when the backing copy comes from offload f)
        no release of dirty objects; terminal offloads exist for λᶠ = backing
 (C4) per-direction channel exclusivity (disjunctive, non-preemptive):
        τ_e + δ(e) ≤ τ_{e′}  ∨  τ_{e′} + δ(e′) ≤ τ_e              for e ≠ e′ in the same direction
 (C5) capacity, at every instant θ:
        Σ_o s_o·live_o(θ) + Σ_t w_t·1[σ_t ≤ θ < σ_t + R_t] ≤ B
        where live_o(θ) = 1 on each residency segment of o:
          from   {0 (pre-placed) | σ_birth (created: reserved at start) | τ_e (inbound start)}
          until  {release instant | τ_f + δ_out(o) (outbound completion) | C}
 (C6) backing capacity analog of (C5) with B_H
 (C7) makespan:                        C ≥ σ_n + R_n  and  C ≥ τ_e + δ(e) for all events
```

Remarks.

- **(C5) is finite.** Occupancy changes only at event points (task starts and
  ends, transfer starts and completions, releases), so the continuous-time
  capacity constraint reduces to `O(n + |E|)` linear constraints once the
  event order is fixed — this is what makes the event-time MILP of
  [problem.md §7.1.2](problem.md) and the CP-SAT model of §7.1.3 workable.
  The pool-tail nuance ([problem.md §1](problem.md)) is captured exactly by
  the "until outbound **completion**" clause.
- **Boundary anchoring is a restriction, not the relaxation.** The program
  above lets `τ_e` be any real; the implementable system enqueues at task
  boundaries and serves FIFO, so its optimum is `≥` the program's optimum.
  The gap is small in practice (there are `n+1` boundaries and a transfer at
  the FIFO head starts the moment capacity admits it) and zero whenever each
  program-optimal `τ_e` is at or after its event's earliest legal boundary,
  but it exists; the exact oracle
  ([tools/bench/pressurefit_exact_oracle.py](../../tools/bench/pressurefit_exact_oracle.py))
  searches the implementable (boundary-anchored) space directly.
- **Existence binaries.** `keep_{o,g}` gates whether events exist; in MILP
  form this is standard optional-activity modeling (big-M on (C2)–(C5)); in
  CP form these are optional interval variables under `Cumulative` — the
  formulation choices and size estimates are already worked out in
  [problem.md §7](problem.md).

### 3.1 Objective decompositions

Two equivalent views of the objective, both useful:

- **Stall form.** `C = Σ_t R_t + Σ_t stall_t` with
  `stall_t = max(0, ready_t − end_{t−1})` where `ready_t` is when `τ_t`'s last
  arrival lands and its reservation admits. Minimizing makespan = minimizing
  total stall — the objective of the *integrated prefetching/caching*
  literature (§6.2).
- **Volume-with-deadlines form.** Fixing residency decisions fixes the byte
  volumes `V_in, V_out` and per-event time windows; makespan optimality is
  then a two-machine sequencing feasibility question (§4.2). This is the
  cleanest way to see where the combinatorics live: *what* to move
  (caching-like), and *when* to move it (sequencing-like), coupled through
  (C5).

---

## 4. Complexity landscape

### 4.1 Polynomial islands

| Special case | Status | Via |
|---|---|---|
| unit sizes, instantaneous transfers, single-object reads, count misses | P | Belady's MIN (1966) |
| unit sizes, per-object fetch costs, count weighted misses | P | offline weighted caching as min-cost flow (k-server offline, Chrobak–Karloff–Payne–Vishwanathan 1991) |
| **set**-valued reads, unit sizes, fixed sequence, count switches | P | KTNS "Keep Tool Needed Soonest" (Tang–Denardo 1988; Crama et al. 1994) — the tool-switching problem |
| unit sizes, fetch **duration** F, single channel, stall objective | P | Albers–Garg–Leonardi (JACM 2000); total-unimodularity proof: Dürr–Hurand (ESA 2006) |
| same, D ≥ 2 channels, ~2(D−1) blocks of extra cache | P under augmentation | Albers–Garg–Leonardi 2000; Albers–Büttner 2003 |
| given residency plan + equal transfer times per channel | P | per-channel EDF/Simons-style sequencing |
| feasibility only (any makespan) | P | lazy schedule: evict-all/reload-all between tasks, feasible iff `max_t (Σ_{o∈A_t∪Y_t} s_o + w_t) ≤ B` and backing admits — the feasibility bound of [research-framing.md](research-framing.md) |

### 4.2 Two independent hardness cores

The problem is NP-hard for two *separate* reasons; each survives when the
other is switched off.

**Core 1 — eviction choice (what to move).** With variable sizes, even
instantaneous-transfer caching is hard: offline *general caching* is
strongly NP-complete in **all four** variants — {fault, bit} miss costs ×
{forced, optional} caching (Chrobak, Woeginger, Makino, Xu; ESA 2010 /
Algorithmica 2012), and stays strongly NP-hard even with every size in
{1, 2, 3} (Folwarczný & Sgall 2015/17). It is constant-factor approximable
(local-ratio 4-approximation, Bar-Noy et al. 2001). Embedding into this
problem: make each task read one object with `R_t → 0`; makespan →
serialized inbound volume; minimizing makespan = bit-model general caching.
The set-request variant is independently hard: tool switching with
**non-uniform tool sizes is NP-hard even for a fixed job sequence** (Crama,
Moonen, Spieksma, Talloen; EJOR 2007) — a task chain requesting sets of
variable-size objects, with zero-time switches, is already intractable. So
*variable object sizes* alone put the residency subproblem in NP-hard
territory — this is where [problem.md §5.2](problem.md)'s "weighted
caching" pointer should be sharpened: uniform-size weighted caching is
polynomial (min-cost flow); it is the **variable-size** generalization that
is hard, with or without weights.

**Core 2 — transfer timing (when to move).** Fix all residency decisions
(evictions forced, e.g. every object dirty with `λᶠ = backing`). Asking
"does a zero-stall plan exist?" makes every `σ_t` a constant, every offload
a job with a fixed **release time** (its producer's end) and a fixed
**deadline** (the instant its bytes must be gone so a later reservation
fits), on one non-preemptive machine (the outbound channel). Non-preemptive
single-machine sequencing with release times and deadlines is strongly
NP-complete (Garey–Johnson SS1; Lenstra–Rinnooy Kan–Brucker 1977). So even
with *nothing to decide about residency*, deciding stall-freeness is hard in
general. (This also bounds what "optimal-given-residency-plan" in
[problem.md §9.2](problem.md) can promise: EDF is exact only in the
preemptive/equal-length relaxations; at chain scale the instances are small
and structured, so this is a worst-case caveat, not a practical obstacle.)

Membership in NP holds (a plan plus queue orders is a poly-size certificate
verified by one simulator replay), so the decision version is NP-complete.

**Direct evidence from the nearest published formalization.** Beaumont,
Eyraud-Dubois, Shilova (Euro-Par 2020) study activation offloading on a
*linear chain* with compute-overlapped transfers and prove that problem
**strongly NP-hard** — a strict special case of this one (one object class,
no in-place mutation, no output/workspace reservation, no duplex-queue
coupling). The hardness of the `dataflow_sim` problem is therefore not an
artifact of its extra generality; it is inherited from the smallest
interesting kernel of the problem. (The "Optimal" in that paper's title
refers to relaxations they solve exactly, not the integral problem.)

**Adjacent evidence on channel multiplicity.** Even at uniform sizes with no
writeback, integrated prefetching/caching with D ≥ 2 channels is NP-hard
without augmentation, and stall minimization is APX-hard (Ambühl & Weber,
STACS 2004). The reduction does not directly transfer here (their channels
partition *reads*; this problem's second channel serves *writes*), but it
marks the single-channel polynomiality of Albers–Garg–Leonardi as fragile:
every known step toward this problem's machine model — a second channel,
variable sizes, or set requests — independently forfeits exactness.

### 4.3 What is genuinely open

- **Approximation.** No approximation guarantee is known for the combined
  objective (stall/makespan with variable sizes, duplex FIFO channels,
  writeback, reservation). The nearby inapproximability landscape is
  discouraging in the worst case: one-shot red-blue pebbling is UGC-hard to
  approximate within 2−δ for I/O count (Papp–Wattenhofer 2020), and
  parallel-channel stall minimization is APX-hard at uniform sizes
  (Ambühl–Weber 2004). Resource-augmented or structured-instance guarantees
  are the realistic targets.
- **The structured subclass.** Chain structure alone does not rescue
  tractability — Beaumont et al.'s strong NP-hardness is *on* a chain, with
  adversarial sizes. But training chains have more structure than that: near
  two uses per object (`f_i`/`b_i`), nested (LIFO) reuse distances for
  activations, monotone phase structure, and size distributions far from
  adversarial. Which of these restrictions (uniform-ish sizes, nested
  lifetimes, bounded uses per object) buys polynomial solvability — as
  tree/SP-graph restrictions do for peak-memory scheduling (§6.4) — appears
  **unstudied**: a concrete, publishable open question.
- **Additive-error guarantees.** The Greedy-Belady conjecture of
  [research-framing.md](research-framing.md) ("within one transfer of
  optimal") has the right *shape* — analogous results exist in the uniform
  size world (Cao et al.'s aggressive-prefetching bounds; Kimbrel–Karlin;
  AGL's exact single-channel result) — but any proof must respect Core 1:
  on adversarial variable-size instances an additive bound at the granularity
  of one *smallest* transfer would decide strongly NP-hard gaps. A safe
  target is: additive `O(τ_max)` **on the structured subclass**, or with
  resource augmentation (plan at capacity `B`, compare against optimum at
  `(1−ε)B` — the classic escape hatch in caching theory).

---

## 5. Lower bounds (cheap, reportable, and mostly unimplemented)

A ladder of makespan lower bounds, each computable from the bare chain, each
usable in benchmarks as denominators for "% of optimal" claims:

1. **Compute.** `LB₀ = Σ_t R_t`.
2. **Channel volumes.** Compulsory inbound bytes: every `λ⁰ = backing`
   object that is used must cross once, minus the best case `≤ B` bytes of
   pre-placement; mandatory outbound: every dirty object with `λᶠ = backing`
   (all `dW_i`, mutated `W/O` state). `LB₁ = max(LB₀, V_in^min/β_in,
   V_out^min/β_out)`.
3. **Prefix/window (Hong–Kung-style) bounds.** For any task window
   `[i..j]` whose distinct-use footprint `V(i,j)` exceeds `B`, at least
   `V(i,j) − B` inbound bytes must land inside `(start_i − lead, start_j)`;
   summed over disjoint windows and combined with (C1) this yields
   segment-interference bounds strictly above `LB₁` in tight regimes. This is
   exactly the red-blue-pebbling I/O lower-bound argument (§6.3) transplanted
   to a timed chain.
4. **LP / flow relaxation.** Drop integrality of `keep` and the disjunctions
   of (C4): a polynomial LP whose optimum lower-bounds every implementable
   plan — the makespan analog of the FOO/PFOO flow bounds that made offline
   optima computable for variable-size CDN caching at millions of requests.
   [problem.md §7.2(3)](problem.md) already recommends this; the caching
   literature (§6.1) confirms the same relaxation is tight in practice on
   realistic (non-adversarial) instances, and flow formulations of
   *stall-time* scheduling specifically have precedent (Albers–Witt's
   multicommodity-flow model for parallel-channel prefetching, and the
   total-unimodular LP behind the single-channel optimum).

---

## 6. The literature, by community

Five research communities own pieces of this problem. For each: what their
model captures, the load-bearing results, and the delta to `dataflow_sim`.

### 6.1 Offline caching and paging

The residency half of the problem, with instantaneous transfers:

- Belady. IBM Systems Journal 1966 — MIN/OPT for uniform pages, polynomial
  (optimality treatment: Mattson, Gecsei, Slutz, Traiger 1970).
- Chrobak, Karloff, Payne, Vishwanathan. SODA 1990 / SIDMA 1991 — offline
  *weighted* caching (uniform sizes, arbitrary fetch costs) is polynomial
  via min-cost flow: cost heterogeneity alone stays easy.
- Irani. STOC 1997 — the bit model (cost ∝ bytes) and fault model (uniform
  cost) for variable-size pages; O(log²k)-competitive online, O(log k)
  offline approximations. Transfer time ∝ size puts this problem's
  residency core in the bit model.
- **Chrobak, Woeginger, Makino, Xu. ESA 2010 / Algorithmica 2012 — offline
  general caching is strongly NP-complete in all four variants ({fault,
  bit} × {forced, optional})**; Folwarczný & Sgall. ISAAC 2015 /
  Algorithmica 2017 — still hard with all sizes in {1, 2, 3}. Variable
  sizes are *the* hardness ingredient, and near-uniform sizes don't escape.
- Bar-Noy, Bar-Yehuda, Freund, Naor, Schieber. STOC 2000 / JACM 2001 —
  local-ratio 4-approximation for general caching (still the best
  unaugmented factor); Albers, Arora, Khanna. SODA 1999 — LP rounding with
  O(1)-largest-page augmentation; Cristi & Wiese. STACS 2020 — improved
  guarantees under resource augmentation via the UFP-cover equivalence.
- Berger, Beckmann, Harchol-Balter. SIGMETRICS 2018 — **FOO/PFOO**: offline
  OPT for variable-size caching as a min-cost-flow relaxation with provably
  tight bounds at scale (hundreds of millions of requests) — the practical
  "compute a near-exact OPT baseline anyway" answer to the NP-hardness.
- Tool switching — offline caching with **set-valued requests**: Tang &
  Denardo. Operations Research 1988 — for a fixed job sequence with
  uniform tool slots, **KTNS** ("keep tool needed soonest" = Belady for
  sets) is polynomial and optimal; Crama, Kolen, Oerlemans, Spieksma. IJFMS
  1994 — choosing the sequence is strongly NP-hard; **Crama, Moonen,
  Spieksma, Talloen. EJOR 2007 — with non-uniform tool sizes, the
  fixed-sequence problem is already NP-hard** — the sharpest OR-side match
  to a task chain requesting sets of variable-size objects (still with
  zero-time switches).
- Modern bridges: Lykouris & Vassilvitskii. ICML 2018 / JACM 2021 —
  paging with predictions (this problem sits at the perfect-prediction
  endpoint); **Atre, Sherry, Wang, Berger. SIGCOMM 2020 — "delayed hits":
  once fetches take real time, Belady's rule is no longer latency-optimal
  even with uniform sizes** — independent confirmation that makespan/stall
  objectives break classical eviction reasoning.
- The fragmentation layer beneath (abstracted away by treating `B` as a
  scalar, restored by the static-arena physicalizer): dynamic storage
  allocation — Robson. JACM 1971/1974 (Θ(log n) worst-case overhead);
  Gergov 1999 (3-approx); Buchsbaum, Karloff, Kenyon, Reingold, Thorup.
  STOC 2003 — OPT ≈ LOAD within (1+ε) when items are small relative to
  capacity, which justifies scalar-capacity modeling exactly in this
  problem's regime (objects ≪ B).

### 6.2 Integrated prefetching and caching (stall minimization)

The community whose *objective* is this problem's objective: known request
sequence, fetches take time and overlap computation, minimize stall:

- Cao, Felten, Karlin, Li. SIGMETRICS 1995 (+ TOCS 1996) — the founding
  model (uniform blocks, fetch time F, one channel): four rules every
  optimal schedule satisfies; **conservative** (≤ 2·OPT elapsed, tight) vs
  **aggressive** (< 2, F/k-dependent) — the same axis PressureFit's
  prefetch-rule portfolio spans.
- Kimbrel & Karlin. FOCS 1996 / SICOMP 2000 — D parallel channels:
  aggressive degrades by ≈ D; **reverse-aggressive** near-optimal —
  planning the *reversed* sequence is the trick (cf. the prefetch/writeback
  duality below).
- **Albers, Garg, Leonardi. STOC 1998 / JACM 2000 — single channel:
  minimum total stall is computable in polynomial time** (LP whose
  fractional optima round to integral ones; Dürr & Hurand. ESA 2006 —
  clean total-unimodularity reformulation). For D channels: polynomial
  with ~2(D−1) blocks of extra cache (resource augmentation); Albers &
  Büttner. SPAA 2003 improve the augmentation.
- **Ambühl & Weber. STACS 2004 — without augmentation, D ≥ 2 channels is
  NP-hard, and stall minimization is APX-hard.** So even at uniform sizes,
  *two channels already forfeit exactness* — directly relevant: this
  problem has two channels by construction.
- **Albers & Büttner. WADS 2003 — reads and writes: dirty blocks must be
  written back before eviction**; tight ratios for conservative/aggressive
  with writes. The only classical-theory treatment of this problem's
  writeback constraint inside the stall model.
- Flow methods: Albers & Witt. APPROX 2001 — multicommodity-flow
  formulation of parallel-channel prefetching; Kallahalla & Varman. SPAA
  2001; Shah, Varman, Vitter. SPAA 2004 — optimal read-once prefetching
  and online variants in the parallel-disk model.
- **Hutchinson, Sanders, Vitter. SICOMP 2005 — prefetching and queued
  writing on parallel disks are formally dual**: optimal prefetch
  schedules arise from greedy write schedules on the reversed sequence. A
  candidate structural lemma for this problem: the offload channel is the
  time-reversed twin of the prefetch channel.
- Systems ancestors: Patterson, Gibson, Ginting, Stodolsky, Zelenka
  (TIP). SOSP 1995 — disclosed future accesses + cost-benefit allocation
  between caching and prefetch-in-flight; Dryden, Böhringer, Ben-Nun,
  Hoefler (NoPFS). SC 2021 — "clairvoyant" prefetching for ML training
  I/O: the known-sequence regime is precisely the ML-workload regime.

**The open cell in this community's matrix is exactly this problem**:
variable-size objects × set requests × duplex channels × writeback ×
space reservation. No paper combining variable sizes with timed,
compute-overlapped transfers was found on either the theory or the survey
side (§6.8's sweep concurs from the systems side).

### 6.3 Pebble games and I/O complexity

The combinatorial skeleton (transfers counted, not timed):

- Hong & Kung. STOC 1981 — the red-blue pebble game: red = fast memory
  (≤ S pebbles), blue = slow; S-partition technique gives the classic I/O
  lower bounds (Ω(n³/√S) matmul, FFT, …). This problem is a red-blue game
  with byte-weighted pebbles, timed duplex moves, workspace, in-place
  mutation, and no recomputation.
- Complexity of playing the game optimally: Gilbert, Lengauer, Tarjan.
  SICOMP 1980 — black pebbling (recomputation allowed) is
  PSPACE-complete; Demaine & Liu. SPAA 2018 — red-blue trade-off
  PSPACE-complete in general, NP-complete under no-deletion, W[1]-hard
  parameterized by transfers; **Papp & Wattenhofer. SPAA 2020 — the
  one-shot variant (every node computed exactly once — this problem's
  regime, since recomputation is planned in a separate layer) is NP-hard
  and, under UGC, inapproximable below factor 2** for I/O count; greedy
  strategies can be far from optimal.
- Weighted/timed extensions are recent and partial: Böhnlein, Papp,
  Yzelman. SPAA 2024 — multiprocessor red-blue pebbling charging compute
  and communication *cost*; Blelloch et al.'s asymmetric-read/write (ARAM)
  line — asymmetric transfer costs (≈ separate β_in/β_out). **No red-blue
  variant with asynchronous, compute-overlapped transfers exists** — that
  overlap lives only in §6.2's model; the combination is unclaimed.
- Lower-bound machinery that transfers to this problem's §5.3 window
  bounds: Ballard, Demmel, Holtz, Schwartz. SIMAX 2011 — communication
  lower bounds for essentially all direct linear algebra; Elango,
  Rastello, Pouchet, Ramanujam, Sadayappan. POPL 2015 (+ PLDI 2020) —
  automated CDAG I/O lower bounds via min-cut arguments.
- Memory-hierarchy cost models: Aggarwal & Vitter. CACM 1988 (external
  memory); Aggarwal, Alpern, Chandra, Snir. STOC 1987 (HMM); Aggarwal,
  Chandra, Snir. FOCS 1987 (block transfer — duration ∝ length, but
  serial); Alpern, Carter, Feig, Selker. Algorithmica 1994 (UMH — the one
  classical model whose inter-level buses run *concurrently* with
  compute, the closest model-level ancestor of overlapped DMA).

### 6.4 Peak-memory and I/O-aware DAG scheduling (the Lyon/Bordeaux school)

A three-decade thread of scheduling theory asks: execute a task graph whose
tasks produce/consume sized data, under a memory bound — minimize peak
memory, I/O volume, or makespan. This is the theory community whose *model*
is closest to `dataflow_sim` (tasks with sized inputs/outputs, not unit
pages):

- Sethi & Ullman. JACM 1970 — register-minimal evaluation order for
  expression trees; the ancestor of every peak-memory ordering result.
- J.W.H. Liu. TOMS 1986 and SIAM J. Alg. Disc. Meth. 1987 — peak-memory of
  out-of-core multifrontal factorization as a function of elimination-tree
  traversal; a polynomial optimal traversal for trees with arbitrary node
  demands (generalized tree pebbling).
- Jacquelin, Marchal, Robert, Uçar. IPDPS 2011 — on trees: **MinMem
  (minimize peak) is polynomial** (Liu's algorithm, improved), but **MinIO
  (given capacity B, minimize transferred volume) is NP-hard, even
  restricted to postorders**. The cleanest statement in the literature that
  *fitting* is easy but *moving the minimum* is hard — the exact split
  between this problem's feasibility result and its optimization hardness.
- Marchal, McCauley, Simon, Vivien. APDCM 2017 / IJFCS 2022 — MinIO on
  trees refined: which data to write out (the offload-victim choice) and
  when.
- Agullo, Guermouche, L'Excellent. SISC 2010 (+ Parallel Computing 2008) —
  in production out-of-core multifrontal solvers, **the traversal minimizing
  peak storage is not the traversal minimizing I/O volume**, with large
  measured gaps: empirical proof that peak-memory heuristics are the wrong
  proxy for a transfer objective.
- Kayaaslan, Lambert, Marchal, Uçar. TCS 2018 — peak-memory-optimal
  scheduling is polynomial for series-parallel DAGs; Herrmann, Marchal,
  Robert. Euro-Par 2013 — trees over *two* memories (CPU + accelerator):
  complexity and inapproximability; Eyraud-Dubois, Marchal, Sinnen, Vivien.
  TOPC 2015 — parallel tree scheduling under shared memory: hardness + list
  scheduling with guarantees; Marchal, Simon, Vivien. JPDC 2019 — add
  fictitious edges so *any* dynamic order respects a memory bound.
- **Beaumont, Eyraud-Dubois, Shilova. Euro-Par 2020** — activation
  offloading on the training chain with overlapped transfers: **strongly
  NP-hard**; relaxation-derived heuristics. The single most on-point
  hardness result for this problem (§4.2). Their **POFO** (NeurIPS 2021)
  adds recomputation and computes the optimal plan *within a schedule
  class* by dynamic programming — the richest exactly-solved cousin;
  MadPipe (ScaDL 2022) and their Euro-Par 2021 pipelining complexity study
  extend the line to pipelined model parallelism.
- Memory-constrained HEFT descendants for heterogeneous platforms:
  Kulagina, Meyerhenke, Benoit. ICPP 2024 and CCGrid 2025 — makespan
  scheduling of large workflows with per-processor memory capacities and
  eviction of intermediates.

### 6.5 Task-based runtimes and out-of-core practice

The systems that *execute* this problem's machine model daily, with online
policies where `dataflow_sim` plans offline:

- **StarPU** (Augonnet, Thibault, Namyst, Wacrenier. Euro-Par 2009 / CCPE
  2011): data handles with MSI coherence, per-device allocation/eviction
  (LRU-ish) and prefetch, transfer-cost-aware schedulers (DMDA/DMDAR,
  Augonnet et al. ICPADS 2010); out-of-core disk nodes since 1.2 (feature,
  no dedicated paper); submission throttling to bound footprint (Sergent et
  al. IPDPSW 2016).
- **The Gonthier–Marchal–Thibault line** — the closest academic neighbor of
  the whole repo:
  - COLOC@Euro-Par 2021: ordering independent data-sharing tasks for reuse
    under GPU capacity;
  - IPDPS 2022: multi-GPU dynamic data selection + custom eviction beating
    DMDAR under scarcity;
  - **FGCS 2023: adapts Belady's furthest-in-future rule to the task-set
    setting as the offline-optimal eviction reference** (the same move as
    `belady_reactive`, published), ~8–11% mean gain over StarPU's best;
  - JPDC 2025: the DARTS scheduler generalized to task graphs and
    out-of-core execution;
  - Gonthier's PhD (ENS Lyon, 2023) consolidates the line — the best single
    background document, and its experimental method (StarPU + simulated
    replays) parallels the sim-verified-portfolio method here.
  Delta: dynamic/runtime decisions, tasks arrive scheduler-driven, objective
  is transfer volume / achieved GFLOPs; no exact-makespan planning against
  a declared bare chain, no duplex-FIFO event optimality, no oracle gap.
- Other runtimes: **Sequoia** (SC 2006) — programmer-expressed memory-
  hierarchy task trees, compiler-generated transfers; **Legion** (Bauer et
  al. SC 2012) — logical regions + a *mapping interface* that is exactly a
  pluggable placement/movement planner (default mappers greedy); **PaRSEC**
  (Bosilca et al. CiSE 2013; IJHPCA 2025 retrospective; Euro-Par 2025
  user-vs-runtime data-placement study) — parameterized task graphs, i.e.
  compile-time-known access sets; **OmpSs/Nanos++** (Bueno et al. IPDPS
  2012) — software cache + prefetch for GPU clusters.
- Out-of-core dense linear algebra — the hand-crafted ancestor of the
  planned-prefetch pattern: Toledo's survey (1999); POOCLAPACK (Reiley, van
  de Geijn 1999; Gunter et al. IPDPS 2001; TOMS 2005); D'Azevedo & Dongarra
  (CCPE 2000) OOC ScaLAPACK; Quintana-Ortí et al. TOMS 2012 — an OOC
  runtime-by-tiles with software cache and automatic double buffering;
  Marqués et al. 2009 and Yamazaki, Tomov, Dongarra (ICCS 2012; CCPE 2017)
  — GPU-as-small-memory factorizations with lookahead panel prefetch; KAUST
  arXiv 2024 — mixed-precision OOC GPU Cholesky with a **static task
  schedule** driving staging (recent evidence that offline planning beats
  dynamic staging on regular chains).
- **Simulation methodology precedent**: Stanisic, Thibault, Legrand,
  Videau, Méhaut. CCPE 2015 — StarPU-on-SimGrid reproduces real hybrid
  executions within a few percent using calibrated kernel/PCIe models — the
  same "simulator as ground truth + fidelity gap tracked" methodology as
  `dataflow_sim`'s sim-vs-real workflow.
- **Channel discipline citation**: the per-direction serial FIFO engines are
  the **full-duplex one-port model** of HPC scheduling — Bhat, Raghavendra,
  Prasanna. JPDC 2003; used throughout steady-state scheduling (Beaumont,
  Legrand, Marchal, Robert. IJFCS 2005). Useful vocabulary when writing for
  scheduling audiences.

### 6.6 DNN-training offload systems (the direct ancestors)

The systems lineage that built this problem's machine model, ordered by how
much *planning* replaced *reacting*. Decision mechanism in bold:

| System | Venue | Decision mechanism | Overlap/FIFO modeling |
|---|---|---|---|
| vDNN (Rhu et al.) | MICRO 2016 | **static layer heuristics** (offload all / conv-only) + profiling pass | transfer streams beside compute; phase-separated directions |
| moDNN (Chen, Chen, Hu) | DATE 2018 | **heuristic transfer-start scheduling** + conv algo + sub-batch choice | explicitly times offload/prefetch against the kernel timeline — the earliest "transfer timing as the optimized object" |
| TFLMS (Le et al.) | ISMM 2019 | **graph rewriting** with distance-threshold swap insertion | control-dependency placement only |
| SuperNeurons (Wang et al.) | PPoPP 2018 | liveness + tensor pool + **cost-model recompute/offload by layer class** | async streams, no queue model |
| SwapAdvisor (Huang, Jin, Li) | ASPLOS 2020 | **genetic algorithm** over op order × allocation × swap set | fitness scored by a **dataflow simulator with separate swap-in/swap-out channels** — the closest published match to this simulator's duplex-FIFO model |
| AutoTM (Hildebrand et al.) | ASPLOS 2020 | **ILP** for tensor placement/movement (DRAM↔PMM) | sync + async-move variants; overlap coarsened to one-kernel granularity, no queues |
| Capuchin (Peng et al.) | ASPLOS 2020 | **profiling-guided online** swap-vs-recompute by benefit metrics; prefetch times regressed from measured accesses | empirical overlap, feedback-adjusted |
| FlashNeuron (Bae et al.) | FAST 2021 | **overlap-aware selection**: offload set chosen so D2SSD hides under forward compute | bandwidth-budgeted, single NVMe path |
| ZeRO-Offload / ZeRO-Infinity (Ren/Rajbhandari et al.) | ATC/SC 2021 | **static partitioning rules** (optimizer/params to CPU/NVMe) | engineered pipelines, fixed lookahead |
| L2L (Pudipeddi et al.) | arXiv 2020 | **fixed one-layer window** | pipelined streaming |
| Harmony (Li, Phanishayee, et al.) | PVLDB 2022 | **cost-model-guided task/swap co-scheduling**, multi-GPU commodity box | double-buffered planned swaps |
| STRONGHOLD (Sun et al.) | SC 2022 | **analytic working-window sizing** from transfer/compute rates | window sized so transfers hide; rate matching |
| PatrickStar (Fang et al.) | TPDS 2022 | **chunk-based, warm-up-statistics-guided** movement | chunking amortizes PCIe; demand-driven |
| Mobius (Feng et al.) | ASPLOS 2023 | pipeline-stage granular swap with **contention-aware stage-to-GPU mapping** | per-PCIe-link contention reasoned about — multi-link cousin of per-direction FIFO |
| Angel-PTM (Nie et al.) | PVLDB 2023 | page-granular pool + **lookahead unified scheduler** (production, Tencent) | movement coordinated with compute stream |
| **G10** (Zhang et al.) | MICRO 2023 | **compile-time planner**: tensor-vitality analysis; iterative greedy eviction choice tracking projected memory pressure and per-link bandwidth occupancy over time; prefetches placed at the **latest safe time, then advanced** | plans against a bandwidth-over-time model; emits evict/prefetch/alloc/free instructions |
| DeepUM (Jung, Kim, Lee) | ASPLOS 2023 | **online correlation-table prefetching** over CUDA UM pages | page-fault driven |
| Smart-Infinity (Jang et al.) | HPCA 2024 | static offload + **near-storage compute** (moves compute to data) | changes the cost model, not the schedule |
| SSDTrain (Wu et al.; preprint "TBA") | DAC 2025 | **adaptive fraction control**: offload rate matched to hideable bandwidth | feedback rate-matching |
| Deep Optimizer States (Maurya et al.) | Middleware 2024 | optimizer shards moved in **profiled idle-bandwidth windows** | interleaving into measured gaps |
| LoHan (Liao et al.; "Fuyou") | ICDE 2025 | **cost-model-chosen offload volumes** across SSD-CPU-GPU | multi-tier engineered pipelines |
| ProTrain / Elixir / Colossal-AI Gemini | arXiv 2024 / arXiv 2022 / ICPP 2023+docs | **configuration search** over chunk/swap/recompute strategy spaces | inherited chunk pipelines |

Reading of the table: the field converged from reactive heuristics (2016–18)
through metaheuristic and ILP planning at coarse timing models (2020) to
G10's compile-time planned migration schedules (2023) — i.e., toward exactly
this problem — but no system states the optimization problem it is
heuristically solving, none models the duplex FIFO queues exactly, and none
reports distance-to-optimal. (Two details are striking in hindsight: G10's
"latest safe prefetch time, then advance eagerly" is PressureFit's
`latest-safe` rule, and SwapAdvisor's GA-over-a-simulator is the
portfolio-over-a-simulator method — both derived independently.) The
2025–26 frontier is tiering and topology, not optimality: SuperOffload
(Grace-Hopper-class superchips change the bandwidth constants), MLP-Offload
(multi-path concurrent offload relaxes one-transfer-per-direction),
GreedySnake (SSD-offload scheduling with optimizer overlap), TURNIP
(deliberately *relaxes* FIFO into a nondeterministic event-driven order),
pommDNN (FGCS 2024, the moDNN-lineage journal formulation nearest this
objective).

### 6.7 Joint recomputation + offloading planners (the parked layer)

The recompute axis is deliberately out of scope here ([recompute.md](recompute.md)
runs *before* PressureFit), but this is the literature where exact
formulations of budgeted memory planning live, and where the joint problem —
this problem plus per-object recompute levels — is already partially mapped:

- Chen, Xu, Zhang, Guestrin. arXiv 2016 — √n segment checkpointing, the
  baseline every planner must beat; its ancestor is optimal binomial
  checkpointing for adjoint computation (Griewank & Walther's REVOLVE).
- Checkmate (Jain et al.). MLSys 2020 — store/recompute per stage as a
  **MILP** under a memory budget; the exact-optimization template, no
  transfers, ~hour solves near 100 nodes.
- DTR (Kirisame et al.). ICLR 2021 — online greedy eviction-for-recompute
  with an Ω(√N)-memory guarantee; the runtime counterpoint.
- **POFO** (Beaumont, Eyraud-Dubois, Shilova). NeurIPS 2021 — offload +
  recompute jointly on training chains by **dynamic programming**, provably
  optimal *within its policy class* (offloading during forward only,
  divisible transfers, chain graphs); the published proof that the joint
  problem admits exact algorithms under structure.
- POET (Patil et al.). ICML 2022 — remat + paging in one **MILP** for edge
  devices (energy objective, per-timestep transfer slots).
- XEngine (Schuler, Membarth, Slusallek). ACM TACO 2023 — Checkmate
  extended to device placement as an **MIQP**; no async overlap.
- Moccasin (Bartan, Li, Teague, Lott, Dilkina). ICML 2023 — recompute
  scheduling as **constraint programming with O(n) interval end-time
  variables** — an order of magnitude faster than boolean MILPs; the
  encoding to borrow for residency intervals (note: remat-only, no I/O).
- Rockmate (Zhao, Le Hellard, Eyraud-Dubois, Gusak, Beaumont). ICML 2023;
  HiRemate (Gusak et al.). ICML 2025 — **hierarchical decomposition**:
  ILP inside blocks, DP across the chain — the scaling trick for exact
  methods on long chains, directly transplantable.
- MegTaiChi (Hu et al.). ICS 2022; Coop (Zhang et al.). NeurIPS 2023 —
  eviction/recompute coupled to allocator layout ("freed bytes are not
  fungible") — the coupling the static-arena physicalizer handles here.
- MODeL (Steiner et al.). ICML 2023 — **ILP over operator order** +
  lifetimes to minimize peak memory: the "choose the order" sibling of this
  fixed-order problem.
- Current thread (2024–26): overlapped recomputation (recompute scheduled
  under communication), pipeline-stage remat planning (Beaumont group, HAL
  2025), T-Control (ASPLOS 2026) — the remat line is still active at top
  venues.

Positioning note for the eventual paper: `dataflow_sim`'s K-level recompute
curve ([research-framing.md](research-framing.md) axis 8) generalizes the
binary store/recompute of this entire line; none of these works co-plan
against a duplex-FIFO transfer timeline.

### 6.8 LLM inference offloading

The inference-side literature re-derives pieces of this problem under
content-dependent access patterns (MoE routing, KV importance), which is why
most of it is *online/predictive* where `dataflow_sim` is *offline/exact*.
The two entries closest to a planned, compile-time schedule:

- Sheng et al. "FlexGen: High-throughput generative inference of LLMs with a
  single GPU." ICML 2023. Chooses tensor placements/percentages across
  GPU/CPU/disk by **linear programming over a structured zig-zag block
  schedule** with overlapped transfers — the best-known optimization-based
  offload planner, but throughput-oriented, block-granular, and restricted to
  the regular layer loop (no per-object directive schedule, no FIFO-queue
  model).
- Jeong, Baek, Ahn. "Fast and efficient model serving using multi-GPUs with
  direct-host-access" (DeepPlan). EuroSys 2023. Emits an **offline per-layer
  execution plan** choosing prefetch-to-GPU vs direct-host-access per layer —
  a plan-then-execute structure like PressureFit's, for cold-start latency.

Streaming/pipelining baselines and KV/MoE hierarchies (each one line):

- Aminabadi et al. "DeepSpeed-Inference." SC 2022 (ZeRO-Inference component):
  fixed lookahead-1 layer streaming from CPU/NVMe — the double-buffering
  baseline of this problem.
- Lee et al. "InfiniGen." OSDI 2024: speculative KV prefetch from host using
  a rehearsal of the next layer's attention — online prefetch selection.
- Song et al. "PowerInfer." SOSP 2024: static hot/cold neuron split,
  residency partition solved once offline — the no-transfer degenerate case.
- Kwon et al. "PagedAttention/vLLM." SOSP 2023: paging *within* GPU memory —
  the allocation substrate, orthogonal to cross-device scheduling; Prabhu et
  al. "vAttention." ASPLOS 2025: CUDA-VMM demand paging alternative.
- Gao et al. "CachedAttention." ATC 2024; Qin et al. "Mooncake." FAST 2025
  (best paper): hierarchical DRAM/SSD KV stores with layer-wise pre-loading
  and storage-vs-recompute economics at serving scale; Liu et al. "LMCache."
  arXiv 2025: production KV layer pipelining loads against compute.
- MoE offloading: Eliseev & Mazur (arXiv 2023) LRU + speculative expert
  prefetch; Kamahori et al. "Fiddler." ICLR 2025 (run expert on CPU vs move
  it); Hwang et al. "Pre-gated MoE." ISCA 2024 (retrain the gate so next
  layer's experts are known one layer early — *manufacturing* the
  compile-time-known access sets this problem assumes); Du et al. "SiDA-MoE."
  MLSys 2024 (learned expert predictor).
- CPU-assist decode: Jiang et al. "NEO." MLSys 2025; He & Zhai "FastDecode."
  arXiv 2024; Yu et al. "TwinPilots." SYSTOR 2024; Chen et al.
  "KTransformers." SOSP 2025 — balance CPU compute against transfer instead
  of scheduling transfers alone (an action outside this problem's directive
  set); Xu et al. "Pie." arXiv 2024 — "performance-transparent swapping"
  is precisely this problem's zero-stall condition, enforced adaptively
  online.

**Takeaway.** A 2024–2026 sweep found no inference paper that formulates the
general problem (per-object directives, duplex FIFO channels, exact makespan)
as an optimization; the nearest formal precedents remain FlexGen's LP and the
accelerator-mapping MIPs (§6.9). The gap the repo targets is real on this
side of the literature.

### 6.9 Accelerator dataflow mapping and DMA scheduling

At chip scale the same mathematics — stage known-size tiles through a
capacity-limited buffer hierarchy over known-bandwidth links, overlapping DMA
with compute, minimizing latency/energy — is called *mapping*:

- Parashar et al. "Timeloop." ISPASS 2019 — analytic cost model + mapper
  search; Kwon et al. "MAESTRO" ("Understanding reuse, performance, and
  hardware cost of DNN dataflow"). MICRO 2019 — data-centric reuse analysis.
- Huang et al. "CoSA." ISCA 2021 — the whole mapping space as **one
  mixed-integer program**; the strongest precedent for exact-MIP data
  orchestration under buffer capacities.
- Yang et al. "Interstellar." ASPLOS 2020 (Halide scheduling view); Mei et
  al. "ZigZag." IEEE TC 2021 (uneven per-operand mappings — per-object
  placement freedom); Kao & Krishna "GAMMA." ICCAD 2020 (genetic mapper —
  SwapAdvisor's metaheuristic cousin at chip scale).
- **SoMa** (HPCA 2025): schedules the *timing* of DRAM prefetches/stores
  across fused layers under on-chip buffer capacity — the nearest
  accelerator-scale twin of the transfer-timing half of this problem, shipped
  in a commercial toolchain.
- Ivanov et al. "Data movement is all you need." MLSys 2021 — transformer
  training is data-movement-bound; motivates stall/traffic objectives.
- Dao et al. "FlashAttention." NeurIPS 2022, and Saha & Ye "I/O complexity
  of attention." ICML 2024 — the red-blue-pebbling lower-bound methodology
  applied to one kernel; the same methodology yields this problem's window
  bounds (§5.3).

Difference from this problem: mapping exploits *regular loop nests* (reuse is
algebraic, schedules are parameterized by tiling factors), whereas the task
chain here is an irregular sequence of heterogeneous objects — closer to
general caching than to tiling.

### 6.10 Compilers and embedded software-managed memory

The oldest home of "compile-time-known program + small fast memory + paid
transfers." Three strands:

- **Register allocation.** Sethi. SICOMP 1975 — k-register sufficiency for
  straight-line code is NP-complete (the ancestral hardness result for
  fixed-order residency planning); Chaitin et al. 1981 — coloring + spilling;
  Bouchez, Darte, Rastello. LCTES 2007 — spill minimization stays NP-complete
  even under SSA, isolating *movement choice* (not assignment) as the hard
  core, exactly parallel to §4.2's Core 1; **Braun & Hack. CC 2009 —
  generalizes Belady's furthest-future eviction to CFGs for spilling**, the
  same import `belady_reactive` makes, two decades later, one level up the
  hierarchy.
- **Scratchpad (SPM) allocation.** Banakar et al. CODES 2002 (the case for
  software-managed on-chip memory); Avissar, Barua, Stewart. TECS 2002
  (static residency as 0/1 **ILP**); Udayakumaran & Barua. CASES 2003 / TECS
  2006 (compile-time-placed copy-in/copy-out points — directive insertion);
  Verma, Wehmeyer, Marwedel. CODES+ISSS 2004 (**ILP over time-phased
  overlays with copy costs** — the closest embedded ancestor of the residency
  half of this problem); Kandemir et al. DAC 2001 (loop tiling + DMA staging).
  None model asynchronous duplex FIFO contention; energy, not makespan, is
  the usual objective.
- **Prefetch insertion and static schedules.** Mowry, Lam, Gupta. ASPLOS 1992
  — software-pipelined prefetch issued exactly latency-early: the
  cache-line-granularity ancestor of `latest-safe`/`packed-fifo` trigger
  placement. Sethi & Ullman. JACM 1970 — register-minimal evaluation *order*
  for trees (the ordering sibling of this fixed-order problem). SDF buffer
  sizing: Bhattacharyya, Murthy, Lee 1996; Murthy et al. Formal Methods in
  System Design 1997 — buffer-minimal static scheduling is NP-complete even
  for acyclic dataflow graphs.
- **ML-compiler memory planning.** Checkmate (§6.7) and Steiner et al.
  "MODeL." ICML 2023 (ILP over operator order + tensor lifetimes for peak
  memory) are the published exact planners; XLA buffer
  assignment/rematerialization and TVM/IREE planners are documentation-only
  practice. The address-assignment layer this problem abstracts away
  (`static-arena physicalizer`) is dynamic storage allocation — see §6.1
  notes on DSA hardness.

### 6.11 GPU demand paging and far memory

The *transparent* (non-planned) alternative to this problem's planned
directives — useful as the experimental control:

- Kehne et al. "GPUswap." VEE 2015; Markthub et al. "DRAGON." SC 2018
  (NVMe-backed UVM); Ganguly et al. ISCA 2019 — UVM prefetcher and eviction
  policy interfere at capacity unless co-designed (the online face of this
  problem's prefetch/evict coupling); Li et al. "ETC." ASPLOS 2019 —
  proactive eviction + throttling + compression under oversubscription; Choi
  et al. "HUVM." ATC 2022 (harvest idle neighbor-GPU memory); Jung et al.
  "DeepUM" (§6.6) is the DNN-specialized member.
- Access-in-place instead of transfer: Min et al. "EMOGI." VLDB 2021; Sabet
  et al. "Subway." EuroSys 2020; Qureshi et al. "BaM." ASPLOS 2023
  (GPU-initiated NVMe) — a directive ("access remotely") outside this
  problem's action set, worth noting as a model extension.

### 6.12 OS and database buffer management

- Patterson et al. "Informed prefetching and caching" (TIP). SOSP 1995 — the
  conceptual OS ancestor: applications *disclose* future accesses; a
  cost-benefit model splits buffers between caching and prefetch-in-flight
  against known device latencies. This problem is TIP's offline, exact,
  GPU-DMA analog.
- Cao, Felten, Li. OSDI 1994; Cao, Felten, Karlin, Li. SIGMETRICS 1995 —
  application-controlled caching + the conservative/aggressive prefetching
  analysis (§6.2).
- Chou & DeWitt "DBMIN." VLDB 1985 — per-operator known access patterns
  driving buffer allocation: the database face of "the compiler knows the
  reference stream."

### 6.13 Scheduling models and solvers

The OR framing for the timed layer, and the exact-solver machinery:

- This problem = RCPSP with one unary compute resource, two unary channel
  resources, and one **reservoir** (produced/consumed cumulative) resource
  — the memory pool. Reservoir scheduling: Neumann & Schwindt. MMOR 2002 —
  inventory-constrained project scheduling; Laborie. Artificial
  Intelligence 2003 — CP propagation for cumulative + reservoir resources
  (ancestor of CP Optimizer's native support; OR-Tools CP-SAT's
  `cumulative` + optional intervals covers the same shape, per
  [problem.md §7.1.3](problem.md)).
- Event-based MILP for continuous-time RCPSP: Koné, Artigues, Lopez,
  Mongeau. COR 2011 — the formulation family [problem.md §7.1.2](problem.md)
  instantiates (start/end event variables, no time grid).
- The per-channel sequencing core: nonpreemptive single-machine
  scheduling with release times and deadlines is strongly NP-complete
  (Garey & Johnson problem SS1; Lenstra, Rinnooy Kan, Brucker 1977) —
  §4.2's Core 2; polynomial under preemption (EDF) or equal processing
  times.
- DAG scheduling with communication delays — the classical macro-dataflow
  frame: Papadimitriou & Yannakakis. SICOMP 1990 (NP-complete,
  2-approximation); modern frontier: Davies, Kulkarni, Rothvoss,
  Tarnawski, Zhang. FOCS 2020 / SODA 2021 — polylog approximations via LP
  hierarchies + clustering. Candidate techniques if approximation for this
  problem is ever attempted.

---

## 7. Nearest neighbors, and the exact delta

No single work matches the full conjunction. The closest artifacts, scored
against the six ingredients of this problem:

| Work | var. sizes | set requests | timed transfers + overlap | duplex FIFO channels | dirty write-back | output/workspace reservation | offline makespan | optimality lens |
|---|---|---|---|---|---|---|---|---|
| Belady MIN (1966) | – | – | – | – | – | – | – | exact (P) |
| Tool switching / KTNS (1988) | – | ✓ | – | – | – | – | – | exact (P, fixed seq.) |
| General caching (Bar-Noy et al. 2001; Chrobak et al. 2012) | ✓ | – | – | – | – | – | – | 4-approx; NP-hard |
| Albers–Garg–Leonardi (2000) | – | – | ✓ | single channel | – | – | ✓ (stall) | exact (P) |
| Albers–Büttner (2003) | – | – | ✓ | parallel disks | ✓ (writes) | – | ✓ (stall) | approx |
| Red-blue pebbling, one-shot (Hong–Kung 1981; Papp–Wattenhofer 2020) | – | ✓ | – (I/O count) | – | ✓ (blue=backing) | – | – | hardness known |
| vDNN (2016) / G10 (2023) | ✓ | ✓ | ✓ | ✓ | ✓ | partial | ✓ | none (heuristic) |
| SwapAdvisor (2020) | ✓ | ✓ | ✓ | ✓ | partial | allocator co-search | ✓ | none (GA) |
| AutoTM (2020) / POET (2022) / XEngine (2022) | ✓ | ✓ | coarse (per-op slots) | – | partial | – | ✓ | exact ILP (coarse model) |
| Checkmate (2020) | ✓ | ✓ | – (recompute only) | – | – | – | ✓ | exact MILP |
| Chain activation offloading (Beaumont et al., Euro-Par 2020) | ✓ | chain-structured | ✓ | one modeled channel | – | – | ✓ (stall) | strongly NP-hard; relaxation heuristics |
| POFO (Beaumont et al. 2021) | ✓ | chain-structured | ✓ | modeled bandwidth | ✓ | – | ✓ | exact DP within a schedule class (joint w/ recompute) |
| FlexGen (2023) | ✓ | ✓ | ✓ | modeled | ✓ | – | throughput | LP over restricted schedule family |
| StarPU + Gonthier et al. (2021–25) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | online/dynamic | heuristics + Belady-adapted eviction as offline reference |
| **dataflow_sim planning problem** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | oracle-exact on small chains; portfolio heuristic at scale |

Reading of the table:

- The **theory column** that matches the objective (stall/makespan with real
  transfer durations and overlap) — AGL and its descendants — has never been
  extended to variable sizes, duplex channels, or space reservation. That
  extension is, as far as we found, open; it is also exactly this problem.
- The **systems row** that matches the machine model — vDNN through G10 —
  never states the optimization problem, and none reports distance to an
  optimum, an LP bound, or an oracle. The exact-oracle + gap-reporting
  methodology here (and in [pressurefit-quality.md](policy/pressurefit-quality.md))
  is the Checkmate methodology applied to the *offload* problem instead of
  the *recompute* problem.
- **POFO** is the nearest published relative in spirit (provable planning for
  bandwidth-aware offload+prefetch on a chain) but solves the joint
  recompute+offload problem on the autograd chain's special structure, with a
  simpler transfer model, and does not handle general object sets, in-place
  mutation, or FIFO queue contention.

## 8. What to import (concrete, near-term)

1. **Name and reuse the conservative↔aggressive axis.** The prefetch-rule
   portfolio (`latest-safe` … `packed-fifo`) reinvents the
   conservative-vs-aggressive spectrum of Cao et al.; their analysis (and
   AGL's exactness result) supplies both vocabulary and proof templates for
   per-rule guarantees in the uniform-size regime.
2. **Belady generalizations for CutScore.** For variable sizes and real
   refetch costs, the principled greedy key is Greedy-Dual-Size-style:
   `(bytes freed) × (time to next use) / (refetch penalty if dropped)` — the
   offline analog of the GDS/GDSF family, and the same key KTNS reduces to
   under uniform sizes. `belady_reactive` is the same move register
   allocation made when it imported Belady for SSA spilling (Braun–Hack).
3. **Lower-bound reporting.** Implement §5's `LB₁`/window/LP bounds beside
   the simulator's makespan in `tools/bench`, so every quality run reports
   `makespan / LB` — turning "PressureFit beats min_grow by 4%" into
   "PressureFit is ≤ 9% from any possible plan". FOO/PFOO is precedent that
   flow-relaxation bounds are tight enough to be decision-useful at scale.
4. **CP-SAT oracle before brute force runs out.** The exhaustive oracle's
   `4^(n·|O|)` space dies around a dozen decisions;
   [problem.md §7.1.3](problem.md)'s CP model (optional intervals +
   `cumulative` for the pool, three unary resources, reservoir-style
   write-back precedences) is the native encoding in OR-Tools/CP Optimizer
   (Laborie's reservoir/cumul-function machinery), and warm-starting it from
   PressureFit's plan gives it an incumbent to prune with. Expected reach:
   L=8–16 chains per [problem.md §7.3](problem.md).
5. **Receding-horizon exactness.** Long chains with local pressure structure
   admit windowed-exact planning (solve `[i, i+k]` exactly with boundary
   conditions, slide): the out-of-core linear-algebra lookahead pattern and
   RCPSP decomposition practice both apply directly.
6. **Resource augmentation for the conjecture.** Prove Greedy-Belady-style
   bounds against an optimum with capacity `(1−ε)B` or bandwidth `(1−ε)β` —
   the standard escape from caching hardness — before attempting the
   unrestricted additive claim (§4.3).
7. **Exploit the prefetch/offload duality.** On parallel disks, prefetch
   scheduling and queued-write scheduling are formally dual — optimal
   prefetch schedules arise from greedy write schedules on the *reversed*
   sequence (Hutchinson–Sanders–Vitter, SICOMP 2005; Kimbrel–Karlin's
   reverse-aggressive exploits the same reversal). For this problem: plan
   the outbound queue as an inbound-prefetch problem on the reversed chain
   (a dirty object's offload deadline is a reversed-chain prefetch
   deadline), potentially unifying the `packed-fifo` machinery across both
   directions.

## 9. Precision notes on the existing docs

Collegial nits on [problem.md](problem.md) / [research-framing.md](research-framing.md),
so the eventual paper cites cleanly:

- **Weighted vs general caching** ([problem.md §5.2](problem.md)): weighted
  caching with *uniform sizes* is polynomial offline (min-cost-flow /
  offline k-server); the NP-hardness kicks in with *variable sizes* (general
  caching — strongly NP-hard even with uniform miss costs, "fault model"),
  with a local-ratio 4-approximation. The doc's current sentence merges the
  two regimes.
- **Pebbling citations** ([problem.md §5.3](problem.md)): PSPACE-completeness
  of black pebbling is Gilbert–Lengauer–Tarjan (1980) — Hopcroft–Paul–Valiant
  is the time-space simulation theorem, not the completeness result. More
  importantly, the game matching offload/prefetch is Hong–Kung's **red-blue**
  pebble game (I/O between fast/slow levels), not black-white pebbling
  (whose white pebbles model nondeterministic guessing, not storage); the
  no-recomputation ("one-shot") red-blue variant is this problem's discrete
  skeleton, and its hardness/inapproximability results (Demaine–Liu;
  Papp–Wattenhofer) are the ones to cite.
- **"EDF is provably optimal"** ([problem.md §9.2](problem.md)): with
  nonpreemptive transfers and release times (a transfer cannot start before
  its producing boundary), even single-channel feasibility is strongly
  NP-complete in general; EDF exactness holds under preemption or equal
  transfer lengths. At the scale of one chain the distinction is academic,
  but the paper claim should be scoped.
- **The Greedy-Belady conjecture** ([research-framing.md](research-framing.md)):
  as stated over all instances it collides with general-caching hardness at
  fine additive granularity (§4.3); scope it to the structured training
  subclass or to resource-augmented comparisons.
- **Framing strength**: the claim "no exact match in the literature"
  ([problem.md §5](problem.md)) survives this survey — with the sharpening
  that the *objective* has a mature theory home (integrated
  prefetching+caching / stall minimization) whose open frontier (variable
  sizes, duplex channels, write-back, reservation) is precisely this
  problem. That is a stronger and more citable position than "no match".

## 10. Provenance

Compiled August 2026. Citations were verified against publisher pages, DBLP,
HAL, and arXiv records where reachable; a few flags survive verification and
are noted inline or here: Neumann & Schwindt is 2002 (often miscited 2003);
the POOCLAPACK conference venue (IPDPS 2001) and a handful of page ranges
rest on standard-reference knowledge rather than fresh lookups; several
2025–26 systems are arXiv-only preprints (marked as such). Before citing in
a submission, re-verify §6 entries against DBLP. Classical results
(Garey–Johnson SS1, external-memory models, Sethi 1975) were checked against
standard references only.
