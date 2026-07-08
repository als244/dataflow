# Dataflow — A CPU–GPU Dataflow Runtime

A runtime that realizes the execution model of
[dataflow_sim](https://dataflowsim.sunshein.net/) ([repo link](https://github.com/als244/dataflow_sim)) on real hardware:
programs are linear chains of tasks over named objects; each task declares
its inputs / mutations / outputs; annotated directives (release, offload,
prefetch) move objects between GPU ("fast") and pinned-host ("slow/backing")
memory.

## Installation

From your Python environment of choice (3.12+), with a sibling
`dataflow_sim` checkout next to this repo:

```bash
uv pip install -e ".[sim,cuda]"
```

(`pip install -e ".[sim,cuda]"` works identically. The `sim` extra
resolves the sibling simulator; `cuda` pulls the real device backend.)

### Quickstart: benchmark training throughput under tight GPU memory budgets

One command sweeps the throughput-vs-memory frontier — models ×
device-memory budgets at a given sequence length and batch size (in
sequences per optimizer step) — picking each model's best
batch/accumulation shape with a fresh profiling oracle, enforcing that
every measured device peak stays under its budget, and rendering the
results table with per-cell provenance:

```bash
python tools/bench_frontier.py \
    --presets dsv32-mini,glm52-mini --seq-tag s4k --seqs-per-step 16 \
    --device-gib 12,16,20,24,28 \
    --shapes oracle --run --no-legacy \
    --out-dir results/bench/quickstart
```

Swap in any builtin presets (`olmoe-7b --seq-tag s1k --seqs-per-step 64`
reproduces the OLMoE column). Output: `TABLES.md` (throughput / sim
prediction / measured peak / chosen shape / recompute fraction per
cell) plus, per cell, the exact dataflow program, its annotated plan,
and the measured row. Full protocol — legality contract, placement
modes, tool matrix: [docs/benchmarking.md](docs/benchmarking.md).

---

# High Level Components

## Dataflow Engine

At its core is a `Dataflow Engine`
([API reference](docs/engine_api.md)) that accepts a `Dataflow
Program` ([schema reference](docs/program_schema.md)): a linear chain
of tasks over named objects, where each task declares its input /
mutated / output objects (with sizes) and a compute key that a
resolver maps to an executable (`task -> executable.launch(ctx)`).
Tasks execute in chain order; a task is dispatched once all of its
input and mutated objects are resident in fast memory and space is
reserved for its outputs.

Additionally, each task may carry data-movement/placement directives
that fire on task completion:
- `release` — free the object's fast-memory allocation
- `offload` — enqueue a fast → slow transfer, then release the fast
  copy when the transfer completes
- `prefetch` — enqueue a slow → fast transfer

A transfer begins only once the engine has reserved sufficient
destination memory. Compute and the two transfer directions ride
separate CUDA streams, so movement overlaps execution; the engine is
deterministic and name-agnostic — everything model-specific lives in
the program and the resolver, not in the engine.


## Building a Dataflow Program for ML Training

The initial intended workload is high-throughput DNN training in low
GPU-memory regimes. A library of builtin model families (Llama 3,
OLMoE, Qwen 3, Qwen 3 MoE, Qwen 3.5, Qwen 3.5 MoE, DeepSeek V3,
DeepSeek V3.2, GLM 5.2) *lowers* a model + training configuration
(sequence length, batch, grad-accum rounds, dtype policy, optimizer
policy) into the Program format the engine expects.

Lowering decomposes each training step into tasks over named objects.
Notation: `i` is the layer index — the intra-round task id equals the
layer id, and every block task is named `{kind}_{step}_{round}_{layer}`
(e.g. `block_fwd_0_1_7`). The objects:

- `W_i` (parameters) and `O_i` (optimizer state) — persistent across
  the whole run; updated in place by optimizer tasks.
- `dW_i` (parameter gradients) — created fresh each step (round 0
  creates, later grad-accumulation rounds accumulate into it).
- `A_i` (saved activation context) — one per (step, round).
- `M_i` — computed metadata that shouldn't be recomputed (MoE router
  assignments, sparse-attention index selections); one per
  (step, round) where the layer makes discrete choices.

Task kinds, in chain order (heterogeneous models get distinct block
task kinds per distinct layer type):

- **Embed**
  - inputs: `tokens`, `W_embed`
  - outputs: the first hidden state `x_0`
- **Forward** (per layer)
  - inputs: layer input hidden state `x_i`, parameters `W_i`
  - outputs: next hidden state `x_{i+1}`, `A_i` (+ optionally `M_i`)
- **Head + Loss** — a SINGLE fused task: final norm + LM head forward
  + loss + head backward, rowwise token-chunked so no (tokens, vocab) tensor
  is ever materialized
  - inputs: last hidden state = `x_L`, `targets`, `W_head`
  - outputs: `loss`, the first upstream gradient = `dy_L`; `dW_head` on round 0
  - mutates: `dW_head` on later rounds (accumulation)
- **Recompute** (per layer; only where the plan dropped `A_i`)
  - inputs: `x_i` and `W_i` (+ optionally `M_i`)
  - outputs:`A_i`, repopulated with same activations as seen during forwards
- **Backward** (per layer)
  - inputs: upstream gradient `dy_{i+1}`, `A_i` (+ optionally `M_i`), `W_i`, `x_i`
  - outputs: downstream gradient `dy_i`; `dW_i` on round 0
  - mutates: `dW_i` on later rounds (accumulation)
- **Embed Backward**
  - inputs: the gradient reaching the embedding, `tokens`
  - outputs: `dW_embed` on round 0; mutates it on later rounds
- **Optimizer** (one task per layer, plus embed and head; composes
  every parameter field's update, with per-field optimizer choice,
  hyperparameters, and state sizing set by the config's optimizer
  policy)
  - inputs: `W_i`, `dW_i`, `O_i`
  - mutates: `W_i`, `O_i`

Correctness of every family is pinned against isolated plain-autograd
reference models at three levels (per op, per task, per model step);
`python tools/verify_family.py --family <name>` runs the whole
ladder. 


To add a family — builtin or from your own package — see
[docs/extending.md](docs/extending.md) and
[docs/extending_external.md](docs/extending_external.md); for programs
outside the standard training shape (e.g. RL training engine from inference-engine saved activations, with worked per-family examples under
[examples/rl_training](examples/rl_training/RL_TRAINING_EXAMPLE.md)),
see [docs/extending_programs.md](docs/extending_programs.md).


## Memory Planning: PressureFit

[PressureFit](https://github.com/als244/dataflow_sim/blob/master/docs/policy/pressurefit.md) is the general planning policy: given any bare task chain
and a fast-memory budget, it annotates every task's release / offload /
prefetch directives so the program executes within budget while keeping
transfers overlapped with compute. It reads only task order, object
sizes/lifetimes, and per-task costs — no model knowledge — so it
applies unchanged to hand-built programs
([docs/extending_programs.md](docs/extending_programs.md)).

Task costs come from a either a roofline estimate or a profiling pass (each unique task signature is
measured once and cached), and the simulator's makespan prediction for
the chosen plan is reported next to every real measurement — the
sim-vs-real gap is tracked as a first-class fidelity metric.

## Recompute Planning (for training workloads)

For DNN training chains specifically, a second planner runs BEFORE
PressureFit: it decides, per layer, whether to keep saved activations
or re-derive them in the backward pass, using the recompute rewrites
the training lowering declares. The selection is a simulator-verified
greedy search — each candidate assignment is re-lowered and priced in
`dataflow_sim` on measured task costs before it is accepted. Custom
(non-training) programs skip this stage and place recompute tasks
explicitly.
