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

Write one module (any importable path) that registers your family at
import time:

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

Activate it via the environment — every tool entrypoint calls
`load_plugins()` at startup:

```bash
export DATAFLOW_PLUGINS=mypkg.dataflow_plugin      # comma-separated list

python tools/verify_family.py --family mymodel \
    --module mypkg/tests/test_mymodel_math.py      # correctness
python tools/bench_train.py --config mymodel-mini-s4k-bs8ga2 --device-gib 16
python tools/bench_campaign.py --presets mymodel-mini --seq-tag s4k \
    --device-gib 12,16,20 --shapes oracle --run --out-dir results/bench/mymodel
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
  (`extending.md` §6); `verify_family --family mymodel --module <path>`
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
