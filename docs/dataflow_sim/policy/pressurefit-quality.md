# PressureFit planning-quality comparison

This guide describes the reproducible cross-version quality gate for
PressureFit. It compares feasibility and simulator-selected makespan for two
planner source trees while holding every bare `TaskChain` and the physical
simulation oracle constant.

## Contents

1. [What the harness proves](#what-the-harness-proves)
2. [Scenario suites](#scenario-suites)
3. [Run it](#run-it)
4. [Classification and acceptance](#classification-and-acceptance)
5. [Reports](#reports)
6. [Current HEAD comparison](#current-head-comparison)
7. [Adding coverage](#adding-coverage)
8. [Limitations](#limitations)
9. [Tiny-chain exact oracle](#tiny-chain-exact-oracle)
10. [Planning-time optimization gate](#planning-time-optimization-gate)
11. [Algorithm-alignment refactor gate](#algorithm-alignment-refactor-gate)

## What the harness proves

[`compare_pressurefit_quality.py`](../../../tools/bench/compare_pressurefit_quality.py)
performs the following isolation protocol:

1. Build each bare chain once with the current scenario generator.
2. Serialize that exact chain and capacity.
3. Materialize the baseline and candidate from a git ref or source path.
4. Run each planner in a separate subprocess with its own `PYTHONPATH`.
5. Replay both returned annotated chains with the **candidate** validator and
   simulator as a common physical oracle.
6. Compare oracle feasibility, makespan, peak bytes, and transfer counts.

This separation matters. An earlier ad-hoc check allowed the editable
worktree package to leak into the baseline process and incorrectly reported
identical results. The isolated subprocesses, chain digests, source identity,
and common replay oracle prevent that failure mode.

The common oracle removes simulator-version drift from the comparison. It does
not prove that the simulator itself is perfect. Counterintuitive results must
be checked against its task and transfer event timeline; simulator behavior is
not an excuse to distort the planner.

## Scenario suites

| Suite | Default cases | Purpose |
|---|---:|---|
| `canaries` | 4 | Exact 122/121-byte final-fast case, backing-only terminal-fast object, and produced-unused terminal-fast object. |
| `synthetic` | 78 | Training-shaped chains with 1, 2, 5, 10, 25, and 100 layers across 13 byte capacities. |
| `models` | 308 | Eleven model-family workload builders, four hardware profiles, five fractions of total logical object bytes, plus task-0 floor and floor-minus-one probes. |
| `all` | 390 | All of the above. |

The model cases use reduced dimensions so the sweep is fast, but retain each
family's real task/object structure, runtime model, optimizer choice, sequence
geometry, bandwidths, and memory ratios. Families currently include Llama 3,
Qwen 3 dense/MoE/hybrid, OLMoE, DeepSeek V3/V3.2, GLM 5.2, GPT-OSS, and
Nemotron-H. Hardware defaults are H100, GB300, RTX 5090, and the SRAM
accelerator preset.

Those family and hardware names exist only in the comparison harness that
constructs coverage. PressureFit receives the resulting bare schema and never
reads a family name, object/task identifier, semantic object type, framework
metadata, or hardware name; it uses only byte sizes, runtimes, bandwidths,
uses, production/mutation facts, locations, and terminal placement.

## Run it

Compare committed `HEAD` with the current worktree and save both reports:

```bash
conda run -n dataflow python tools/bench/compare_pressurefit_quality.py \
  --baseline HEAD \
  --candidate WORKTREE \
  --suite all \
  --output-dir results/pressurefit-quality/head-vs-worktree
```

Compare two future commits:

```bash
conda run -n dataflow python tools/bench/compare_pressurefit_quality.py \
  --baseline <old-commit> \
  --candidate <new-commit> \
  --suite all \
  --output-dir results/pressurefit-quality/old-vs-new
```

Either source may also be an extracted repository path. `WORKTREE` means this
repository's current tracked files.

Useful grid controls:

```text
--hardware H100,GB300,RTX_5090
--model-ratios 1.0,0.75,0.5,0.35,0.25
--synthetic-layers 1,2,5,10,25,100
--synthetic-caps 144,160,192,224,256,320,384,500,640,800,1024,1600,2400
```

Exact non-regression is the default. For an explicitly noise-tolerant study,
the caller may set `--abs-tolerance-us` and/or `--relative-tolerance`; the
report always retains the exact delta.

## Classification and acceptance

| Classification | Meaning | Default gate |
|---|---|---|
| `equal` | Both oracle-valid; candidate makespan is unchanged. | Pass |
| `makespan_improvement` | Both valid; candidate is faster. | Pass |
| `feasibility_improvement` | Baseline planner produced no plan; candidate is oracle-valid. | Pass |
| `correctness_improvement` | Baseline returned a plan rejected by the common oracle; candidate is valid. | Pass |
| `both_nonvalid` | Neither source produced an oracle-valid plan. | Diagnostic, pass |
| `makespan_regression` | Both valid; candidate exceeds the configured tolerance. | Fail |
| `feasibility_regression` | Baseline is valid; candidate produced no plan. | Fail |
| `candidate_invalid` | Candidate planner returned a chain rejected by the common oracle. | Fail |

The process exits nonzero for any failing row. A regression is a diagnostic
that requires root-cause analysis; it is not an instruction to blindly revert
an otherwise sound change. Material regressions should be fixed. Tiny deltas
may be accepted only through an explicit documented tolerance when the simpler
algorithm is preferable.

## Reports

`--output-dir` writes:

- `report.md`: summary, all regressions, the twenty largest relative
  makespan improvements, and every newly valid scenario;
- `report.json`: every scenario, source identities, bare/planned-chain SHA-256
  digests, planner result, planner wall time and evaluated-candidate counts,
  common-oracle result, peaks, transfer counts, and exact absolute/relative
  makespan delta.

The JSON schema is intentionally suitable for future CI or a richer report
renderer. The Markdown report is the normal human review surface.

## Current HEAD comparison

The 2026-08-02 isolated `HEAD` versus worktree run produced:

| Suite | Equal | Faster | Newly valid | Nonvalid in both | Regressions |
|---|---:|---:|---:|---:|---:|
| Canaries (4) | 1 | 0 | 2 | 1 | 0 |
| Synthetic (78) | 50 | 3 | 7 | 18 | 0 |
| Models (308) | 57 | 83 | 0 | 168 | 0 |

Representative model-derived improvements demonstrate that the correction is
not limited to byte-sized canaries:

| Family / hardware | Capacity | HEAD us | Candidate us | Change |
|---|---:|---:|---:|---:|
| DeepSeek V3 / H100 | 247,247,619 | 1,535.831 | 918.837 | -40.17% |
| GLM 5.2 / H100 | 248,065,923 | 1,536.523 | 934.710 | -39.17% |
| DeepSeek V3.2 / H100 | 248,305,923 | 1,536.689 | 934.876 | -39.16% |
| GPT-OSS / H100 | 253,231,875 | 1,536.153 | 945.944 | -38.42% |
| OLMoE / H100 | 276,825,219 | 1,544.624 | 952.352 | -38.34% |

The DeepSeek/H100 event timeline was inspected directly. HEAD retained many
objects, then serialized one 1,311-us late-use-object prefetch after the main
backward work. The candidate instead kept that object resident, released other
objects, and pipelined seven smaller prefetches; the common simulator reports
peaks of 215,303,172 and 246,645,248 bytes respectively, both below the same
247,247,619-byte cap. This illustrates a general size/use/transfer tradeoff,
not a model- or identifier-specific rule. The 40.17% improvement is therefore
an explainable schedule change, not a makespan-reporting artifact.

## Adding coverage

To add a synthetic geometry, extend `_DEFAULT_SYNTHETIC_LAYERS` or
`_DEFAULT_SYNTHETIC_CAPS`, or pass CLI overrides. To add a model family, add a
small `_ModelSpec` with a registry family, preset, shape overrides, sequence
length, and optimizer. To add hardware, use an existing
`HARDWARE_PRESETS` name and include it in `--hardware`.

New scenarios must remain deterministic and serialize through the public
`TaskChain.from_dict` form. Do not build separate baseline/candidate workloads;
that would conflate workload-builder changes with planner changes.

## Limitations

- The harness compares the selected PressureFit plan, not global scheduling
  optimality.
- The model suite samples meaningful structures and budgets; it is not an
  exhaustive search over every model configuration.
- `both_nonvalid` does not prove physical infeasibility. It means neither
  compared planner produced a chain accepted by the common oracle.
- The candidate simulator is intentionally the common oracle. Suspected event
  ordering, allocator, or capacity-accounting defects require independent
  simulator tests and event-timeline analysis.

## Tiny-chain exact oracle

[`pressurefit_exact_oracle.py`](../../../tools/bench/pressurefit_exact_oracle.py)
exhaustively enumerates the four post-task choices (`none`, `release`,
`offload`, and `prefetch`) for every object/task slot, validates each annotated
chain, and selects the lowest simulator makespan. It fixes the exact initial
placement supplied by the caller and rejects search spaces above a configurable
limit. Its exponential complexity makes it appropriate only for small generic
chains.

The oracle establishes a different claim from the cross-version gate:

- the cross-version gate proves feasibility/makespan non-regression across
  realistic sampled workloads;
- the exact oracle measures global approximation gap within a tiny finite
  annotation space.

On the three-task 122-byte terminal-fast canary, exhaustive enumeration checks
262,144 assignments, finds 80 valid plans, and finds a 4-us optimum. Current
PressureFit returns the same legal 122-byte peak at 5 us: it chooses to retain
the terminal-fast object and move the later-use object, while the optimum moves
the terminal object and overlaps its restoration with the final task. This
one-microsecond gap is retained as explicit evidence that PressureFit is a
verified bounded heuristic, not a global optimizer. The planning-time-only
optimization pass does not change the selected plan.

## Planning-time optimization gate

PressureFit may terminate its bounded race when a simulator-verified candidate
exactly equals the serialized compute-only makespan. That value is a global
lower bound for an ordered `TaskChain`; the early exit therefore cannot change
the optimal selected makespan, and stable candidate ordering preserves the
existing tied chain.

Comparing the pre-optimization source snapshot with the optimized worktree over
all 390 scenarios produced zero differences in the 203 valid annotated-chain
digests or selected candidate IDs. Representative five-layer cases that reach
the bound improved from 10-22 ms to 0.45-1.02 ms (about 95%). Ten- and
100-layer tight cases that do not reach the bound remain noise-equivalent and
still evaluate all forty candidates.

The isolated full-corpus run terminated early in 85 of 203 valid scenarios.
Summed planner wall time over valid rows fell 11.1% overall, including 6.8% for
the synthetic suite and 21.6% for the model-derived suite. These aggregate
single-run timings complement, rather than replace, the repeated microbenchmarks
above.

More complex caching/queue experiments are retained only as findings, not
source changes: whole-interval memoization was slower, and an incremental
boundary priority queue improved long-chain medians by only 1.7-3.7%. The
simpler reducer was preferred at that effect size.

## Algorithm-alignment refactor gate

The 2026-08-02 structural refactor separated transition construction,
`PrefetchRule` application, annotation emission, residency refinement, and
candidate verification, then made `pressurefit(...)` the algorithm-shaped
core. A frozen copy of the immediately preceding source was compared after
each step with the complete 390-case suite.

| Gate | Equal valid plans | Nonvalid in both | Digest or selected-candidate differences | Aggregate candidate/baseline planner time |
|---|---:|---:|---:|---:|
| Same-code process-order control | 203 | 187 | 0 | 1.006× |
| Transition/rule/emission split | 203 | 187 | 0 | 1.011× |
| Typed rule and module normalization | 203 | 187 | 0 | 1.016× |
| Final cleaned algorithm-shaped core | 203 | 187 | 0 | 1.011–1.020× |

Two final full-corpus runs measured 1.011× and 1.020× aggregate planner time.
In the latter, the ratios remained similar across canaries (1.019×),
model-derived cases (1.028×), and synthetic chains (1.017×), rather than
growing sharply with workload complexity. This roughly one-to-two-percent
planning-only difference is close to the 0.6% same-code process-order control;
task schedules,
simulator makespans, feasibility, transfer counts, candidate counts, selected
candidate IDs, and all 203 serialized valid plans remained exactly unchanged.
