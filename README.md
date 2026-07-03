# dataflow — a CPU–GPU dataflow runtime

A runtime that realizes the execution model of
[dataflow_sim](https://dataflowsim.sunshein.net/) on real hardware:
programs are linear chains of tasks over named objects; each task declares
its inputs / mutations / outputs; annotated directives (release, offload,
prefetch) move objects between GPU ("fast") and pinned-host ("backing")
memory. A planning policy (PressureFit + a recompute planner, from the
simulator) chooses those annotations so that memory-constrained execution
approaches unconstrained throughput by overlapping transfers with compute.

The first workload is memory-constrained single-GPU DNN training. Headline
(llama3-8B, bf16 AdamW, 65,536 tok/step, RTX 5090): **3,501 wall tokens/s
under a 23.75 GiB budget** — above the strongest baseline we could measure
on the same machine (flextrain, 3,410–3,435, memory-insensitive) — and 97%
of that ceiling at a 12 GiB budget. Every number is measured against a
sim prediction built from measured task costs, and every configuration's
math is pinned to a plain-autograd golden model.

## Layout

```
src/dataflow/
├── core/        program IR + validation + JSON + sim converters
├── runtime/     generic engine: ledger, pool, placement, dispatcher,
│   └── device/  transfer engines, trace; DeviceBackend (fake + cuda)
├── tasks/       executables: ops → staged blocks; kernel registry; layouts
├── training/    lowering, planning (the only sim importer), profiling,
│                train loop, gradcheck harness
└── models/      golden autograd references
tools/           m4_train (sweeps), gap_analysis, window_plans, nsys_profile,
                 export_measured_run, m4_correctness, m4_tables
docs/            architecture, usage, extending + notes/ (measured post-mortems)
results/         promoted measurement tables (each dir has its README)
```

## Start here

- **What is this / how does it fit together** — [docs/architecture.md](docs/architecture.md)
- **Run training under a budget** (API + CLI) — [docs/usage.md](docs/usage.md)
- **Add an op / block / model family** — [docs/extending.md](docs/extending.md)
- **Why the numbers are what they are** — [docs/m4-report.md](docs/m4-report.md),
  [docs/notes/perf-headroom.md](docs/notes/perf-headroom.md),
  [docs/notes/step-boundary.md](docs/notes/step-boundary.md)
- **Project plan + decision log** — [PLAN_V4.md](PLAN_V4.md)

## Quick start

```bash
conda activate dataflow
python -m pytest tests -m "not gpu"     # CPU suite (sim parity, IR, planning)
python -m pytest tests -m gpu           # GPU suite (gradcheck ladder, gates)
python tools/m4_train.py --config 8b-s1k-bs8ga8 --budgets 16 --steps 3 --recompute
```

Principles the code holds throughout: sizes are exact (packed layouts),
costs are measured (profiled tasks, measured PCIe, disk-cached for
reproducibility), plans are proven (static placement validated against
physical VRAM at planning time), and results are honest (wall-clock
tokens/s reported next to makespan; escape/eviction counters in every row;
correctness gated by plan-invariance, poison-on-free, and
interleaving-stress).
