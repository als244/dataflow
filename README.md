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

(`uv sync` also works. The `sim` extra resolves the SIBLING
`dataflow_sim` checkout through `[tool.uv.sources]` — plain pip does
not read that table, so under pip install the sibling editable first:
`pip install -e ../dataflow_sim -e ".[sim,cuda]"`. The `cuda` extra
pulls the real device backend.)

### Quickstart: train

`tools/train/train.py` is the one training tool at every world size —
zero-config single GPU, a data-parallel fleet when given a topology —
with checkpointing and resume built in (flag reference:
[tools/train/README.md](tools/train/README.md); fleet setup:
[docs/distributed_training.md](docs/distributed_training.md);
save/restore mechanics: [docs/checkpointing.md](docs/checkpointing.md)):

```bash
# single GPU, zero config, with periodic checkpoints. The preset fixes
# max sequence length; --batch x --ga-rounds x seq_len = tokens/step
# (here 8 x 8 x 1024 = 64K tokens per step from the fineweb shards)
python tools/train/train.py train --preset gpt2_124m --steps 1000 \
    --data datasets/fineweb10B --batch 8 --ga-rounds 8 \
    --checkpoint-every 100 --run-name demo

# resume from the newest complete checkpoint
python tools/train/train.py train --preset gpt2_124m --steps 1000 \
    --data datasets/fineweb10B --batch 8 --ga-rounds 8 \
    --checkpoint-every 100 --run-name demo --resume auto

# Nsight capture of exact warmed steps, report written under --prof-dir
python tools/train/train.py train --preset gpt2_124m --steps 10 \
    --profile --profile-start-before-step 5 --profile-stop-after-step 8 \
    --prof-dir results/pretrain/logs
```

### Quickstart: benchmark training throughput under tight GPU memory budgets

Three tools cover the throughput workflow in escalating cost — predict
(CPU, instant), measure (GPU, minutes), profile (GPU, one capture).
Full guide: [docs/benchmarking.md](docs/benchmarking.md); builtin model
families and preset configs: [builtin_models](docs/builtin_models.md).

```bash
# 1. Predict: simulated sweep over geometry x memory budgets — per cell
#    s/step, tok/s, effective/hardware TFLOPs/s, fast/backing memory
#    peaks, PCIe traffic + link %, recompute/idle %
python tools/bench/predict_step.py --preset gpt2_124m --hw 3090 \
    --t-round 8192,32768,65536 --tokens-step 524288 --budget 16,8,4,2 \
    --seq-len 1024,4096 --backing 16

# 2. Measure: the same grid, each cell RUN on the real engine — the
#    warmed measurement lands beside the prediction for that cell's plan
python tools/bench/measure_step.py --preset gpt2_124m \
    --t-round 8192,65536 --tokens-step 524288 --budget 16,4 \
    --seq-len 1024,4096 --backing-gib 16 --steps 12

# 3. Profile: the same run under Nsight Systems, capture bracketed to
#    exact warmed steps via the daemon's profiler_control verb
python tools/train/train.py train --preset gpt2_124m --steps 10 \
    --profile --profile-start-before-step 5 --profile-stop-after-step 8
```

Everything is inspectable in the
[webapp simulator](https://dataflowsim.sunshein.net/):
`tools/export/export_program.py` writes any preset's exact program, annotated
plan, and predicted timeline as uploadable files — the simulator's
expectations for that plan, priced from profiled/estimated task costs.
For the true timeline of a real run, `tools/export/trace_real_run.py` drives a
few real steps through the daemon and emits the measured event log next
to the sim's prediction; the webapp renders both in the same panels and
diffs them, which is exactly how the sim-vs-real fidelity gap is
inspected (full guide: [docs/exporting_runs.md](docs/exporting_runs.md)).

---

### Quickstart: distributed training

The same `train.py` drives a data-parallel fleet — the only addition
is a `topology.toml` describing the machines (copy
`topology.example.toml`; full guide:
[docs/distributed_training.md](docs/distributed_training.md)). Each
member needs passwordless ssh from the conductor box, the same repo
at the same git commit (launches refuse version skew), the same
Python environment, and the dataset shards on local disk. Per-host
entries carry the peer address, NIC/RDMA device, and default memory
sizes; groups name the participating members:

```toml
conductor = "boxA"
[hosts.boxA]
peer_listen = "192.168.50.23:29700"
[hosts.boxB]
ssh = "boxB"                     # ssh alias from the conductor
python = "/home/me/env/bin/python"
repo = "/home/me/dataflow"
peer_listen = "192.168.50.32:29700"
[groups.dp]
members = ["boxA", "boxB"]
backend = "auto"                 # nccl on real fabrics
```

```bash
# weighted data parallelism: the step's data is divided 3:1 —
# boxA's faster GPU takes six rounds' worth of tokens, boxB two, and
# each rank's LOCAL grad-accum count simply equals its data share.
# Per-rank fast/backing memory, comma per rank.
# Checkpoints and --resume auto work exactly as in the solo quickstart.
python tools/train/train.py train --preset gpt2_124m --steps 1000 \
    --data datasets/fineweb10B --ga-rounds 8 \
    --topology topology.toml --group dp --rounds 6,2 \
    --fast-budget 14,12 --backing-budget 60,30 \
    --checkpoint-every 100 --run-name fleet-demo
```

# High Level Components

## Dataflow Engine

At its core is a `Dataflow Engine`
([API reference](docs/engine_api.md)) that accepts a `Dataflow Program` ([schema reference](docs/program_schema.md)): a linear chain
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
GPU-memory regimes. A library of builtin model families (GPT-2, Llama 3,
OLMoE, Qwen 3, Qwen 3 MoE, Qwen 3.5, Qwen 3.5 MoE, DeepSeek V3,
DeepSeek V3.2, GLM 5.2 — full preset table with parameter counts:
[docs/builtin_models.md](docs/builtin_models.md); per-family deep
references with every task's objects, stages, and measured kernel
sequences: [docs/models/](docs/models/README.md)) *lowers* a model +
training configuration
(sequence length, batch, grad-accum rounds, dtype policy, optimizer
policy) into the Program format the engine expects.

Lowering decomposes each training step into tasks over named objects.
Notation: `i` is the layer index — the intra-round task id equals the
layer id, and every block task is named `{kind}_{step}_{round}_{layer}`
(e.g. `block_fwd_0_1_7`). Hidden states are written `x_i` below; their
chain object ids carry the same step/round suffixes (`y_embed_{s}_{r}`,
`y_{s}_{r}_{i}`, gradients `dy_{s}_{r}_{i}`). The objects:

- `W_i` (parameters) and `O_i` (optimizer state) — persistent across
  the whole run; updated in place by optimizer tasks.
- `dW_i` (parameter gradients) — created fresh each step (round 0
  creates, later grad-accumulation rounds accumulate into it).
- `A_i` (saved activation context) — one per (step, round).
- `AuxTemp_i` — computed metadata that must not be re-derived (MoE
  router assignments, sparse-attention index selections); one per
  (step, round) where the layer makes discrete choices; recompute and
  backward consume it VERBATIM.
- `Aux_i` — persistent per-layer auxiliary state (e.g. the expert-load
  counts driving MoE load balancing): zeroed at round 0, accumulated
  by every round's forward, read by the last round's backward.

Task kinds, in chain order (heterogeneous models get distinct block
task kinds per distinct layer type):

- **Embed**
  - inputs: `tokens`, `W_embed`
  - outputs: the first hidden state `x_0`
- **Forward** (per layer)
  - inputs: layer input hidden state `x_i`, parameters `W_i`
  - outputs: next hidden state `x_{i+1}`, `A_i` (+ optionally `AuxTemp_i`)
- **Head + Loss** — a SINGLE fused task: final norm + LM head forward
  + loss + head backward, rowwise token-chunked so no (tokens, vocab) tensor
  is ever materialized
  - inputs: last hidden state = `x_L`, `targets`, `W_head`
  - outputs: `loss`, the first upstream gradient = `dy_L`; `dW_head` on round 0
  - mutates: `dW_head` on later rounds (accumulation)
- **Recompute** (per layer; only where the plan dropped `A_i`)
  - inputs: `x_i` and `W_i` (+ optionally `AuxTemp_i`)
  - outputs: `A_i`, repopulated with same activations as seen during forwards
- **Backward** (per layer)
  - inputs: upstream gradient `dy_{i+1}`, `A_i` (+ optionally `AuxTemp_i`), `W_i`, `x_i`
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
`python tools/verify/verify_family.py --family <name>` runs the whole
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

Task costs come from either a roofline estimate or a profiling pass (each unique task signature is
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

---

# Documentation References

### Using the runtime
- [usage.md](docs/usage.md) — programmatic end-to-end walkthrough: config → plan → engine, in Python
- [builtin_models.md](docs/builtin_models.md) — the builtin model families and preset table
- [benchmarking.md](docs/benchmarking.md) — the throughput workflow: predict (simulated), measure (real), profile (Nsight)
- [throughput.md](docs/throughput.md) — throughput methodology and reference numbers
- [exporting_runs.md](docs/exporting_runs.md) — exporting measured runs and programs to the webapp simulator

### Training
- [distributed_training.md](docs/distributed_training.md) — fleets: topology, parallelism scheme, responsibility, resume
- [checkpointing.md](docs/checkpointing.md) — snapshot/restore engine API, lease protection, checkpoint record, resuming
- [data_feeds.md](docs/data_feeds.md) — data pipelines, packing, and the content-view contract
- [frozen_training.md](docs/frozen_training.md) — frozen-parameter training: the freeze API and warm-up phases
- [correctness_compare.md](docs/correctness_compare.md) — the parity methodology: weight-space instruments and envelopes

### Engine and program model
- [architecture.md](docs/architecture.md) — codebase map and dependency rules
- [engine_api.md](docs/engine_api.md) — the in-process engine API
- [engine_service.md](docs/engine_service.md) — the persistent daemon: verbs, store, peers
- [program_contract.md](docs/program_contract.md) — what a valid program promises
- [program_schema.md](docs/program_schema.md) — the serialized program format
- [task-contract.md](docs/task-contract.md) — the task execution contract
- [task_kinds.md](docs/task_kinds.md) — the registry of task kinds
- [kernel_registry.md](docs/kernel_registry.md) — registered kernels, variants, and determinism flags

### Extending
- [extending.md](docs/extending.md) — adding a model family inside the repo
- [extending_external.md](docs/extending_external.md) — external families via the program-resolver plugin contract
- [extending_programs.md](docs/extending_programs.md) — authoring custom (non-training) programs
