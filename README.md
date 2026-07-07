# Dataflow — A CPU–GPU Dataflow Runtime

A runtime that realizes the execution model of
[dataflow_sim](https://dataflowsim.sunshein.net/) on real hardware:
programs are linear chains of tasks over named objects; each task declares
its inputs / mutations / outputs; annotated directives (release, offload,
prefetch) move objects between GPU ("fast") and pinned-host ("slow/backing")
memory.

## Installation

Uses [uv](https://docs.astral.sh/uv/). With a sibling `dataflow_sim`
checkout next to this repo:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
uv sync --extra sim --extra cuda                   # creates .venv, installs torch + deps
uv run pytest -q                                   # verify (GPU tests need a CUDA device)
```

`--extra sim` pulls the planner/verifier integration (the sibling
checkout, editable, via `[tool.uv.sources]`); `--extra cuda` pulls the
real device backend. Plain pip works too: `pip install -e .[sim,cuda]`.
Every command below runs as `uv run python ...` (or activate `.venv`).

### Quickstart: benchmark training throughput under tight GPU memory budgets

One command runs a full campaign — models × device-memory budgets at a
given sequence length and batch size (in sequences per optimizer step) —
picking each model's best batch/accumulation shape with a fresh
profiling oracle, enforcing that every measured device peak stays under
its budget, and rendering the results table with per-cell provenance:

```bash
uv run python tools/bench_campaign.py \
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

## Dataflow Engine

At its core is a `Dataflow Engine` that accepts a `Dataflow Program`: a
linear chain of tasks over named objects, where each task declares its
input / mutated / output objects (with sizes) and a compute key that a
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

## Memory Planning

Planning turns a bare task chain into an annotated one, in two separable
stages (each consumes and produces a plain `Program` — see
[docs/extending_programs.md](docs/extending_programs.md) for the full
pipeline contract):

1. **Recompute planner** (optional; standard training chains): decides,
   per layer, whether to keep saved activations or re-derive them in the
   backward pass, using the program's declared recompute rewrites. The
   selection is a simulator-verified greedy search — each candidate
   assignment is re-lowered and priced in `dataflow_sim` on measured task
   costs before it is accepted.
2. **PressureFit**: given the (possibly re-lowered) chain and a fast-
   memory budget, annotates every task's release / offload / prefetch
   directives so the program executes within budget while keeping
   transfers overlapped. PressureFit reads only task order, object
   sizes/lifetimes, and measured costs — it has no model knowledge and
   works on any Program.

Task costs come from a profiling pass (each unique task signature is
measured once and cached), and the simulator's makespan prediction for
the chosen plan is reported next to every real measurement — the
sim-vs-real gap is tracked as a first-class fidelity metric.

## Building a Dataflow Program for ML Training

The initial intended workload is high-throughput DNN training in low
GPU-memory regimes. A library of builtin model families (Llama 3,
OLMoE, Qwen 3, Qwen 3 MoE, Qwen 3.5, Qwen 3.5 MoE, DeepSeek V3,
DeepSeek V3.2, GLM 5.2) *lowers* a model + training configuration
(sequence length, batch, grad-accum rounds, dtype policy, optimizer
policy) into the Program format the engine expects.

Lowering decomposes each training step into tasks over named objects —
parameters `W_i`, optimizer state `O_i`, gradients `dW_i`, saved
activations `A_i`, and (for MoE / sparse-attention models) small
metadata objects `M_i` holding discrete decisions. Per layer kind the
task vocabulary is: forward, recompute, backward, plus embed, head/loss,
and optimizer tasks. Heterogeneous models get distinct task kinds per
distinct layer type.

- **Forward** tasks take the layer's input hidden state and parameters,
  and output the next hidden state plus the saved context `A_i` (and
  `M_i` where the layer makes discrete choices — MoE routing,
  sparse-attention index selection).
- **Recompute** tasks re-derive `A_i` from the same layer input and
  parameters when the plan chose not to keep it. `M_i` is NEVER
  recomputed — discrete selections are cheap to store and must be
  bit-identical in the backward, so recompute repopulates only the
  float context and consumes the saved decisions.
- **Backward** tasks take the upstream hidden-state gradient, `A_i`
  (and `M_i`), parameters, and the layer input, and produce the
  downstream gradient — creating `dW_i` on the first grad-accumulation
  round and mutating it on later rounds.
- **Optimizer** tasks take `dW_i` and mutate `W_i` and `O_i` — one task
  per layer composes every parameter field's update, with per-field
  optimizer choice, hyperparameters, and state sizing set by the
  config's optimizer policy.

Correctness of every family is pinned against isolated plain-autograd
reference models at three levels (per op, per task, per model step);
`uv run python tools/verify_family.py --family <name>` runs the whole
ladder. To add a family — builtin or from your own package — see
[docs/extending.md](docs/extending.md) and
[docs/extending_external.md](docs/extending_external.md); for programs
outside the standard training shape (e.g. RL post-training from saved
rollouts, with worked per-family examples under
[examples/rl_training](examples/rl_training/RL_TRAINING_EXAMPLE.md)),
see [docs/extending_programs.md](docs/extending_programs.md).
