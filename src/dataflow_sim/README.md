# Dataflow Sim

*`dataflow_sim` is the simulator/planning component of the `dataflow` runtime — its `DataflowProgram` schema is consumed by the `dataflow` engine (`dataflow.core.convert`) and `dataflow_training`'s lowering.*

A discrete-event simulator for memory-constrained dataflow workloads on a two-tier memory hierarchy. The simulator uses three parallel streams (compute, slow->fast memory, fast->slow memory). Our model was originally intended for CPU<-->GPU compute/communication overlap planning; however, it is also practical for HBM<-->SRAM hierarchy (just the units differ; same high-level problem). We assume workloads are constructed as a sequential list of abstract tasks where each task contains lists of input, output, and mutated object identifiers (we assume object sizes are specified, and task runtimes can be derived). The simulator enforces that all input and mutated objects are present in fast memory before starting the task and enforces that the task stalls until there is sufficient fast memory capacity to create all output objects. The simulator manages queues to track fast<-->slow transfer requests and only one transfer (per direction) can be in-flight at a time. ***The primary objective is to minimize overall runtime when there is a hard constraint on fast memory capacity. This means ensuring a combination of (a) avoiding idle time and (b) avoiding recomputation.***

We formulate this problem as annotating a *task-chain* with **release**, **offload**, and **prefetch** directives where each contains a list of 0 or more object identifiers. After a task completes, the runtime (or simulated runtime) triggers execution of such directives:

- **Release**: Free fast-memory storage associated with that object.
- **Offload**: Enqueue transfer request for object in the fast->slow transfer queue. Upon completion of transfer, the object is released.
- **Prefetch**: Enqueue transfer request for object in the slow->fast transfer queue. Fast memory is reserved for the object before initiating the transfer; transfer doesn't begin until there is sufficient fast memory capacity to reserve destination space.

Our main policy (methodology for deciding annotations) is called [PressureFit](../../docs/dataflow_sim/policy/pressurefit.md).

For DNN training workloads we further apply [recompute planning](../../docs/dataflow_sim/recompute.md) based on memory pressure and runtime results reported by the simulator; recomputation decisions add tasks to the original set.

## Visualizing Simulated Workloads with the Webapp

The default policy is quite effective and can be [visualized](https://dataflowsim.sunshein.net/) for carrying out model training in memory-constrained regimes.

The simulator ingests an abstract dataflow program; we take a model architecture specification and translate it to a task chain that mimics reality. In the webapp you will see an unannotated plan that contains all of the tasks with input/output/mutated objects along with associated task runtime. After you run a simulation you can see a summary of overall metrics, the annotated plan, a timeline of events on each of the streams, composition of fast memory over time, and replayable events. The `Throughput vs. Fast Memory Budget` sweep at the top will run simulations across different memory budgets; then choose a memory budget level that is interesting to see how events actually unfold. *The ideal case is to achieve a runtime close to that of the unlimited fast-memory-capacity regime using just a fraction of fast-memory...*

You can also [create your own dataflow program](../../docs/extending_programs.md) and export it to a `DataflowProgram v1` JSON file that the webapp can ingest and simulate.

<!-- 
> [!NOTE]
> The space of possible planning decisions is combinatorial and becomes more difficult when the number of tasks increases and/or when memory pressure increases. Model-training workloads can be decomposed at different granularities: whole layers, smaller module phases, individual ops, or chunks of the main activation tensor. These are optimization opportunities, but they come with the challenge of more difficult planning and recomputation. -->

## Setup

For creating custom workloads or accessing simulator API.

`dataflow_sim` is a first-party package of this repo — there is no separate
install and no extra. Installing the repo at its root installs the `dataflow`
distribution, which bundles `dataflow_sim` and its webapp deps
(fastapi/uvicorn/pydantic):

```bash
# -1. Activate any python environment you want to work from

# 0. Clone the repo:
git clone git@github.com:als244/dataflow.git

# 1. Install the repo (installs the `dataflow` distribution + bundled dataflow_sim):
cd dataflow && pip install -e .
```

To author your own dataflow programs, see
[extending_programs.md](../../docs/extending_programs.md).

<!--
## Repo Layout

- `src/dataflow_sim/core/` - task-chain schema, validation, and reference-stream utilities.
- `src/dataflow_sim/engine/` - workload-agnostic event simulator.
- `src/dataflow_sim/policies/` - policies that annotate bare workloads with release/offload/prefetch plans.
- `src/dataflow_sim/workloads/` - generic workload schema, workload builders, and shared workload concepts such as hardware specs.
- `src/dataflow_sim/app/` - FastAPI backend for the current webapp.
- `src/dataflow_sim/ui/` - React frontend.
- `docs/dataflow_sim/` - design + recipe docs.
-->

## TODOs

- [ ] Model distributed training in the *simulator itself*: add scale-up and scale-out network streams/queues to the simulator, and 'directives' to intiate P2P and collective communication transfers (or maybe these details should be 'baked' in to intra-layer efficiency...). (The `dataflow` engine already performs real distributed training; this item scopes only the simulator's own modeling of it.)
- [ ] Add customizable / finer-grained model-training task decomposition (both in terms of op granularity and batch/chunk granularity).
- [ ] Enable periodic planning to handle gradient accumulation and multi-step training efficiently.
- [ ] Expand custom dataflow examples beyond model training.
