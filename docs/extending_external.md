# Extending from OUTSIDE the package: external model families

`extending.md` is the walkthrough for adding a *builtin* family. This is
the same walkthrough for a family that lives in **your own repo/package**
— a new model drops, the dataflow package hasn't been updated, and you
want the core engine, planner, profiler, and bench tools to run it
unmodified.

This is supported. Everything a family is made of — ops, kernel
registrations, block classes, golden models, the lowering helpers — is
public, inheritance-based API; the only things that were package-private
were the two *registries* (the family table and the bench preset table),
and both now have registration functions.

## The plugin contract

The API surface a family implements is TYPED and VALIDATED:

- `dataflow.training.families.Family` — the registration record. Its
  fields are `typing.Protocol`s with documented signatures (`DimsOfFn`,
  `LowerFn`, `InitialValuesFn`, `BuildResolverFn`, `GoldenFn` — see
  their docstrings in `families.py` for the exact contracts, including
  the task-naming shape `lower` must keep and what the golden class
  must expose).
- Block executables subclass `BlockFwd` / `BlockRecompute` / `BlockBwd`
  (the stage grammar, MetaState, ProfileFill machinery is inherited).
- Fused kernels register through `dataflow.tasks.kernels.registry.register`.
- `validate_family("mymodel")` structurally checks the whole surface in
  seconds — config presets, lowering + task-naming shape, resolver
  coverage of every emitted task, golden class members — before any
  deep math runs (verify_family runs it as level 0).

Write one module that registers your family at import time:

```python
# mypkg/dataflow_plugin.py
from dataflow.training.families import Family, register_family
from dataflow.training.presets import register_bench_config

from .mymodel_training import (        # your code, structured like a
    ShapedMyModelConfig,               # builtin training/<family>.py
    dims_of_mymodel, lower_mymodel, initial_values_mymodel,
)
from .mymodel_blocks import build_mymodel_resolver   # your blocks


def _mymodel() -> Family:
    return Family(
        name="mymodel",
        config_type=ShapedMyModelConfig,
        dims_of=dims_of_mymodel,
        lower=lower_mymodel,
        initial_values=initial_values_mymodel,
        build_resolver=build_mymodel_resolver,
        golden=lambda: __import__("mypkg.mymodel_reference",
                                  fromlist=["GoldenMyModel"]).GoldenMyModel,
    )


register_family("mymodel", _mymodel)
register_bench_config("mymodel-mini-s4k-bs8ga2",
                      ShapedMyModelConfig.mini(batch=8, grad_accum_rounds=2))
```

Two discovery paths, both first-class:

1. **Packaging (the normal path)** — declare an entry point in YOUR
   package's `pyproject.toml`; every dataflow tool discovers it
   automatically once your package is installed, zero configuration:

   ```toml
   [project.entry-points."dataflow.families"]
   mymodel = "mypkg.dataflow_plugin"
   ```

2. **Dev loop (uninstalled code)** — every tool takes `--plugin`:

   ```bash
   python tools/verify_family.py --plugin mypkg.dataflow_plugin --family mymodel ...
   ```

Programmatic use (your own scripts) needs neither: import your plugin
module, then call the dataflow APIs directly.

## After it's built: validate -> verify -> benchmark -> use

```bash
# 0. structural contract (seconds, no GPU math)
python - <<'PY'
import mypkg.dataflow_plugin  # registers
from dataflow.training.families import validate_family
assert validate_family("mymodel") == [], validate_family("mymodel")
PY

# 1. correctness: per-op, per-task (fwd/recompute/bwd), per-model —
#    your test module follows the 11-gate canon (extending.md §7);
#    verify_family runs it + the contract check + the coverage audit
python tools/verify_family.py --plugin mypkg.dataflow_plugin \
    --family mymodel --module mypkg/tests/test_mymodel_math.py

# 2. throughput: single cells or a full campaign
python tools/bench_train.py --plugin mypkg.dataflow_plugin \
    --config mymodel-mini-s4k-bs8ga2 --device-gib 16 --steps 3 \
    --out artifacts/bench
python tools/bench_campaign.py --plugin mypkg.dataflow_plugin \
    --presets mymodel-mini --seq-tag s4k --device-gib 12,16,20 \
    --shapes oracle --run --no-legacy --out-dir results/bench/mymodel

# 3. programmatic training (no tools at all)
python - <<'PY'
import mypkg.dataflow_plugin
from dataflow.training.families import family
from dataflow.training.planning import plan_program
from dataflow.training.train_loop import train
from dataflow.runtime.device.cuda import CudaBackend

fam = family("mymodel")
cfg = mypkg.dataflow_plugin.ShapedMyModelConfig.mini()
planned = plan_program(fam.lower(cfg), fast_memory_capacity=16 << 30)
report = train(planned.program, cfg, CudaBackend(), steps=10)
PY
```

## What your package implements (all public imports)

The content is identical to `extending.md` — only the file locations
move into your package:

| builtin location | yours | imports you use |
|---|---|---|
| `tasks/ops.py` additions | `mypkg/mymodel_ops.py` | plain torch; reference forms for the golden |
| `tasks/kernels/<op>.py` | `mypkg/mymodel_kernels.py` | `from dataflow.tasks.kernels.registry import register, none, internal` — the decorator ABI is public; your fused impls join the same kernel-set stamp |
| `tasks/<family>_blocks.py` | `mypkg/mymodel_blocks.py` | subclass `BlockFwd`/`BlockRecompute`/`BlockBwd` from `dataflow.tasks.llama3_blocks` (or a closer builtin family); STAGES grammar, MetaState mixins, ProfileFill — all inherited machinery |
| `models/<family>_reference.py` | `mypkg/mymodel_reference.py` | subclass a builtin golden or compose `dataflow.tasks.ops.*_reference` |
| `training/<family>.py` | `mypkg/mymodel_training.py` | `build_shaped_program`, `LayerKindSpec`, `MetaShare`, `FamilyLayouts`, `size_of_factory`, `initial_values_from_layouts` — the whole lowering toolkit is importable |
| `training/families.py` entry | the plugin module above | `register_family` |
| `tools/bench_train.py` CONFIGS | the plugin module above | `register_bench_config` |
| `tests/tasks/test_<family>_math.py` | `mypkg/tests/test_mymodel_math.py` | copy the NEWEST builtin family's module as the template; `check_block_backward` / `check_model_step` import from `dataflow.training.testing.gradcheck` |

Config rule, relaxed for external families: `resolve_family` dispatches
EXACT type first, then isinstance — so your config MAY subclass a
builtin config (convenient when your arch is a variant of one). Distinct
dataclasses remain the cleaner default.

## Verification and benchmarking

- Correctness: your test module follows the same 11-gate canon
  (`extending.md` §7); `verify_family --family mymodel --module <path>`
  runs it and audits coverage. Pin your lowering tripwire hash inside
  your own module (the builtin `test_lowering_stability.py` stays
  builtin-only).
- Perf: once `register_bench_config` names exist, every bench tool works
  unchanged — including the oracle (`best_config` resolves presets via
  the registered family's config classmethods) and full campaigns.
- Profile cache: keyed by task signatures + kernel set, family-agnostic
  — your family profiles into the same cache. If you ship custom
  kernels, their registration names become part of the kernel-set stamp
  automatically.

## Current limitations (fork-only edges)

Small, cosmetic, and shrinking — none block a working external family:

- `train_loop._STEP0_ID` — the NVTX step-renamer knows builtin prefixes;
  external families get generic step labels (display only).
- `tools/window_plans.py` name regexes assert full task-name coverage;
  keep the `prefix_{step}_{round}_{layer}` naming shape (any prefix) and
  they hold.
- The builtin lowering-stability tripwire file doesn't import plugin
  families; pin your hash in your own test module (above).
- `verify_family`'s canon audit scans shared builtin op-suite modules
  for coverage credit; external ops' pins should live in your one test
  module (simpler anyway).
