# PressureFit

PressureFit is the general-purpose planning policy for ordered `TaskChain`s.
It removes optional fast-memory residency until each candidate fits, assigns
legal transfer triggers, verifies the emitted chain with the event-driven
simulator, and returns the valid candidate with the lowest makespan.

PressureFit is workload- and framework-agnostic. It reasons from task order,
runtimes, object sizes, locations, uses, mutations, transfer bandwidths, and
fast-memory capacity. It never branches on model family, tensor role, task
name, object identifier, or `Object.type`.

## Contents

- [Public API](#public-api)
- [Inputs and output](#inputs-and-output)
- [Planner behavior](#planner-behavior)
- [Configuration](#configuration)
- [Diagnostics](#diagnostics)
- [Guarantees and limits](#guarantees-and-limits)
- [Development and validation](#development-and-validation)

## Public API

The normal policy entry point returns only the annotated chain:

```python
from dataflow_sim.policies.pressurefit import apply_pressurefit_policy

planned = apply_pressurefit_policy(
    bare_chain,
    fast_memory_capacity=None,
    preplace="greedy",
    prefetch_rules=None,
)
```

Two equivalent diagnostic surfaces return the selected chain and a
`PressureFitDiagnostics` record:

```python
from dataflow_sim.policies.pressurefit import (
    plan_pressurefit_policy,
    pressurefit,
)

planned, diagnostics = pressurefit(bare_chain)
planned, diagnostics = plan_pressurefit_policy(bare_chain)
```

`pressurefit(...)` is the algorithm-shaped core. The two policy-named
functions are thin compatibility surfaces for callers that expect the common
`apply_<policy>_policy` / `plan_<policy>_policy` convention.

## Inputs and output

The input is a bare `TaskChain` containing:

- ordered task runtimes, inputs, outputs, and `mutates_inputs` metadata;
- object byte sizes and initial/final locations;
- inbound and outbound bandwidths; and
- `fast_memory_capacity`.

The returned `TaskChain` preserves the workload and adds executable movement
annotations:

- initial fast-memory placement;
- post-task release or offload actions; and
- post-task prefetch triggers.

The simulator, not the analytic pressure model, is the physical authority for
capacity, FIFO transfer timing, and makespan. Every selected plan has passed a
full simulator replay.

## Planner behavior

PressureFit evaluates a small deterministic portfolio:

```text
one seed residency plan
  -> five CutScore/pressure strategies
  -> four PrefetchRules, each in ordinary and clean-gap-coalesced form
  -> transition construction and annotation emission
  -> simulator verification and bounded capacity repair
  -> minimum-makespan valid plan
```

The five residency strategies cover conservative prefetch headroom,
capacity-tight accounting, two legal-cut rankings, and one relaxed stopping
rule:

| Strategy | Pressure view | `CutScore` | Stopping behavior |
|---|---|---|---|
| `headroom-stall` | Reserves prefetch headroom | `min-stall` | Continue strict cuts |
| `headroom-transfer` | Reserves prefetch headroom | `min-transfer` | Continue strict cuts |
| `tight-stall` | Required residency only | `min-stall` | Continue strict cuts |
| `tight-transfer` | Required residency only | `min-transfer` | Continue strict cuts |
| `relaxed-stall` | Required residency only | `min-stall` | Stop when executable relaxed pressure fits |

The four base `PrefetchRule`s choose one legal task-completion boundary at
which each already-required prefetch is enqueued:

| Rule | Intent |
|---|---|
| `packed-fifo` | Pack requests backward from deadlines while coordinating the inbound FIFO. |
| `packed-fit` | Apply FIFO packing, then delay triggers whose early destination occupancy would exceed analytic capacity. |
| `interval-entry` | Extend residency entries where capacity permits to expose additional transfer lead time. |
| `latest-safe` | Minimize early destination occupancy by choosing the latest individually feasible trigger. |

Each rule also has a `-coalesced` form that retains an object across a clean,
same-boundary release/reload pair. Dirty write-back/reload pairs are never
coalesced. All variants are still simulator-verified; no heuristic is assumed
to dominate the others.

The formal execution model, notation, paper algorithms, exact heuristic
definitions, architectural diagram, and source-module ownership live in
[PRESSUREFIT_ALGO.md](PRESSUREFIT_ALGO.md).

## Configuration

| Argument | Default | Meaning |
|---|---|---|
| `bare` | required | Unannotated ordered `TaskChain`. |
| `fast_memory_capacity` | `None` | Optional per-call override of the chain's fast-memory capacity. |
| `preplace` | `"greedy"` | `"greedy"` fills spare initial fast memory; `"task0"` initially places only task-0 inputs. |
| `prefetch_rules` | `None` | Optional tuple of exact rule names to evaluate. `None` evaluates the complete built-in portfolio. |

`prefetch_rules` is intended for controlled trials and diagnostics. Normal
planning should leave it unset so the simulator can choose among all current
rules. Unknown-only selections raise `ValueError`; a raw rule never silently
falls back to another rule.

## Diagnostics

`PressureFitDiagnostics` reports:

- total planning wall time and task/object counts;
- fast-memory capacity and evaluated/valid candidate counts;
- selected candidate ID and simulated makespan; and
- one row per evaluated candidate with status, wall time, makespan when valid,
  and the rejection error when invalid.

A candidate ID is `<residency-strategy>/<prefetch-rule>`, for example
`tight-stall/packed-fit`. Diagnostics are observational and do not influence
selection. The planner may evaluate fewer than the maximum forty candidates
when a residency strategy is infeasible or a verified candidate reaches the
serialized compute-only lower bound.

## Guarantees and limits

PressureFit provides these guarantees:

- **Schema-generic:** planning depends only on framework-neutral task/object
  facts.
- **Deterministic:** fixed inputs and configuration produce the same candidate
  ordering and selected chain.
- **Bounded:** at most five residency strategies times eight prefetch-rule
  variants are replayed, plus bounded physical repair.
- **Simulator-verified:** the selected plan is structurally valid and accepted
  by the same simulator used to score its makespan.
- **Standalone:** it does not silently delegate to another planning policy.

It remains a greedy bounded heuristic, not a global scheduling optimizer:

- local residency cuts can miss a faster globally coordinated movement plan;
- physical repair addresses capacity contradictions, not arbitrary makespan
  improvement;
- prefetch rules target ideal consumer deadlines and may reject a niche plan
  whose only feasible behavior deliberately stalls through the consumer's
  predecessor; and
- the boundary model assumes an ordered `TaskChain`; a general DAG requires a
  different planning formulation.

## Development and validation

Use the following references rather than duplicating their material here:

| Need | Reference |
|---|---|
| Formal simulator contract, algorithms, architecture, heuristics, and implementation map | [PRESSUREFIT_ALGO.md](PRESSUREFIT_ALGO.md) |
| Reproducible cross-version quality, feasibility, and planner-time comparison | [pressurefit-quality.md](pressurefit-quality.md) |
| Quality-harness CLI | [tools/bench/README.md](../../../tools/bench/README.md) |
| Policy unit tests | [test_pressurefit.py](../../../tests/dataflow_sim/planning/policies/test_pressurefit.py) |
| Tiny-chain exact oracle tests | [test_pressurefit_exact_oracle.py](../../../tests/dataflow_sim/planning/test_pressurefit_exact_oracle.py) |

Any planner change should pass the policy/simulator tests and the
cross-version quality gate. A refactor is acceptable only when feasibility and
makespan do not regress; small planner-wall changes should be weighed against
code clarity.
