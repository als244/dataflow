# dataflow_training — the workload package

The WORKLOAD side of the engine/workload split: model families and
their block executables, the shared block/kernel libraries, lowering +
planning (the `dataflow_sim` dependency lives here), data streaming,
the run drivers, and the distributed conductor. It builds programs and
registers the resolvers that execute them; the engine (`dataflow`)
runs them without ever importing this package. The seam is specified
in [docs/program_contract.md](../../docs/program_contract.md).

## Public API surface

What each subpackage exposes and what tools/tests actually import:

- **`model_families`** — `families.py` is the registry:
  `ModelFamily` / `Model`, `family(name)`, `resolve_family(cfg)`,
  `register_family`, `load_plugins`, `validate_family`, and
  `build_init_program` (init-as-program). One package per family
  (`llama3/`, `qwen3/`, `qwen35/`, `olmoe/`, `qwen3moe/`,
  `qwen35moe/`, `dsv3/`, `dsv32/`, `glm52/`), each with `model.py`
  (config, dims, lowering, seeded init), `blocks.py` (executables +
  resolver), `bridge.py` (packed bytes -> the `reference_models` twin),
  `presets.py` (study/smoke shapes). `bridges.py` re-exports the
  uniform bridge pair (`build_reference_model`, `load_reference_init`);
  `init_policy.py` the init-rule vocabulary.
- **`register`** — the ONE hookup into the engine service:
  `register_all()` registers resolver kind `"model_family"`;
  `canonical_spec(family, cfg_dict, hyper)` builds the wire spec.
- **`blocks`** — shared executable library: `base_blocks.py`
  (family-neutral `EmbedFwd`/`HeadLoss`/`EmbedBwd`/`OptimizerStep` +
  `AdamWHyper`, `RoundPrologue`), `layouts.py` (`PackedLayout`,
  `DTypePolicy`, `grad_layout`, `opt_state_layout`, embed/head table
  layouts), `optim.py` (optimizer defs, `OptPolicy`,
  `register_optimizer`, `freeze`, `LRSchedule`), `ops.py`,
  `linear.py`, `modules/` (the pluggable MoE package, `dsa_forms`,
  `mla_forms`), `adamw/` (optimizer-step comm variants, one file
  each: `dp`, `shards`, `rs` over the shared `update` core).
- **`kernels`** — the registry op implementations
  (`resolve_kernels`, `registry.py` ABI; one file per op family).
- **`lowering`** — config -> executable program:
  `shaped_program.build_shaped_program` + `LayerKindSpec` (the
  family-generic chain grammar), `emit.py` (`FamilyLayouts`,
  `LayerLayout`, `apply_exact_sizes`, `object_size_factory`,
  `initial_values_from_layouts`), `planning.py` (`plan_program`,
  `simulate_program` — the dataflow_sim boundary), `freeze_plan.py` /
  `freeze_program.py`, `replay.py` (`replay_gap_pct`).
- **`data`** — `segments.py` (`Segments`, `resolve_segments`,
  `uniform_segments` — the varlen value object), `sources/`
  (deterministic token streams), `feed.py` + `packer.py` (the
  pipeline + packing), `pipeline.py`.
- **`run`** — drivers and study plumbing: `driver.py`
  (`engine_client`, `init_model`, `run_engine`, `run_reference`,
  `plan_at_budget`), `presets.py` (locked training config +
  cross-family preset re-exports, `resolve_preset`, `cfg_dict`,
  `resolver_family`), `bench_presets.py` (`register_bench_config`),
  `recipe.py` (`Recipe`), `profiling.py` (`profile_program`,
  `load_or_profile`, `apply_measured_costs`, `cached_pcie`),
  `parity.py`, `schedule.py`, `scaling.py`, `crosscheck.py`.
- **`distributed`** — the distribution vocabulary and machinery:
  `topology.py` (host/rank maps), `hosts.py` (run commands / move
  files on any host), `daemons.py` (dataflowd lifecycle on a host),
  `parallelism.py` (`ParallelismScheme` — THE contract), `ranks.py`,
  `sharding.py`, `responsibility.py`, `group_annotation.py`,
  `grouped_lowering.py`, `fleet.py` (re-export facade). Driven by
  the conductor in `run/` via `tools/train/train.py`.
- **`testing`** — `gradcheck.py` (`check_block_backward`,
  `check_model_step`), `block_forms.py`.

## Dependency arrows

- `dataflow_training` -> **`dataflow` public surfaces only** (rule R2,
  `tests/test_import_boundaries.py`): `dataflow.core.*`,
  `dataflow.runtime.*` (the ABIs — `TaskContext`, `torch_view`, device
  backends), `dataflow.service` (Server/EngineConfig/EngineClient for
  rigs), `dataflow.service.client`, `dataflow.service.registry`
  (`register_program_resolver`), `dataflow.service.wire`.
- `dataflow_training` -> **`dataflow_sim`**: only under `lowering`
  (planning/replay), and lazily in-function everywhere it appears
  (rule R4) — importing the workload without the simulator installed
  works until you plan.
- `dataflow_training` -> **torch** (blocks, kernels, data, run).
- Nothing imports this package from the engine side (rule R1), and the
  truth tree (`reference_models/`) imports nothing from anywhere —
  the bridges here import IT.

## Extending

Adding a family — builtin or external — is three files
(model/blocks/bridge) + one twin + one registry line:
[docs/extending.md](../../docs/extending.md),
[docs/extending_external.md](../../docs/extending_external.md).
Custom (non-family) programs: [docs/extending_programs.md](../../docs/extending_programs.md).
