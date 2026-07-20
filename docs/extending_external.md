# Extending from OUTSIDE the package: external model families

`extending.md` is the walkthrough for adding a *builtin* family. This is
the same walkthrough for a family that lives in **your own repo/package**
— a new model drops, the dataflow packages haven't been updated, and you
want the engine, planner, profiler, service daemon, and bench tools to
run it unmodified.

This is supported and CI-gated: the repo carries a complete external
family as a test fixture —
`tests/fixtures/external_family/toy_family.py`, exercised end to end by
`tests/test_external_family.py` (registration, lowering, structural
validation, and resolution through the SERVICE seam). When in doubt,
read the fixture; this page follows it.

## The plugin contract

Everything a family is made of — the lowering toolkit, block templates,
layouts, kernels registry — is public API; the registries have
registration functions:

- `dataflow_training.model_families.families.ModelFamily` — the
  registration record. Its callable fields are `typing.Protocol`s with
  documented signatures (`DeriveDimsFn`, `LowerFn`, `InitialValuesFn`,
  `BuildResolverFn` — see their docstrings in `families.py`, including
  the task-naming shape `lower` must keep). The gradcheck-bundle and
  twin/bridge fields are optional (extending.md §7).
- `register_family(name, thunk)` — registers a ZERO-ARG thunk returning
  the `ModelFamily` (lazy, so registration is import-cheap). Duplicate
  names raise.
- `load_plugins(explicit=[...])` — imports plugin modules so they
  self-register. Two discovery paths, both first-class (below).
- `validate_family("mymodel")` — structural contract check in seconds,
  no GPU math: presets exist, lowering runs and keeps the naming shape,
  the resolver covers every emitted task with a `.launch`
  (`tools/verify_family.py` runs it as level 0).
- Custom OPTIMIZERS register through
  `dataflow_training.blocks.optim.register_optimizer` (state slots +
  step rule; per-field assignment via the config's `opt_policy` —
  extending.md §6); your family's `derive_dims` must forward
  `opt_policy=cfg.opt_policy`.
- Fused kernels register through the kernels registry
  (`dataflow_training.kernels.registry`) — your impls join the same
  kernel-set stamp the profile cache keys on.
- Bench presets register through
  `dataflow_training.run.bench_presets.register_bench_config` — the
  named-config table the bench tools merge over their builtins.

Write one module that registers your family at import time — this is
exactly what the fixture does (abridged; `toy_family.py` is the full
version, CPU-safe: torch only inside `launch`/init bodies, never at
import):

```python
# mypkg/dataflow_plugin.py  (toy_family.py in the fixture)
from dataflow_training.model_families.families import ModelFamily, register_family

def toy_family() -> ModelFamily:
    return ModelFamily(
        name="toyfam",
        config_type=ToyConfig,          # frozen dataclass with tiny()
        derive_dims=toy_dims,
        lower=lower_toy,                # build_shaped_program + apply_exact_sizes
        initial_values=toy_initial_values,   # accepts into= (see KNOWN GAPS)
        build_resolver=build_toy_resolver,   # (dims, hyper=None) -> resolver
    )

register_family("toyfam", toy_family)
```

The family composes ONLY public surfaces: the standard chain grammar
(`lowering.shaped_program.build_shaped_program` + `LayerKindSpec`),
packed-byte truth + seeded init (`lowering.emit` — `FamilyLayouts`,
`LayerLayout`, `apply_exact_sizes`, `object_size_factory`,
`initial_values_from_layouts`), the family-neutral templates
(`blocks.base_blocks` — `EmbedFwd`/`HeadLoss`/`EmbedBwd`/
`OptimizerStep`), layouts (`blocks.layouts` — `PackedLayout`,
`DTypePolicy`, the embed/head table layouts), and the kernels registry
(`kernels.resolve_kernels`).

## Discovery: two paths, both first-class

1. **Packaging (the normal path)** — declare an entry point in YOUR
   package's `pyproject.toml`; `load_plugins()` discovers it
   automatically once your package is installed, zero configuration:

   ```toml
   [project.entry-points."dataflow.families"]
   mymodel = "mypkg.dataflow_plugin"
   ```

2. **Dev loop (uninstalled code)** — the family-aware tools take
   `--plugin` (repeatable): `verify_family.py`, `predict_step.py`,
   `measure_step.py` — and the DAEMON takes it too:

   ```bash
   python tools/verify_family.py --plugin mypkg.dataflow_plugin --family mymodel ...
   python tools/dataflowd.py start --plugin mypkg.dataflow_plugin ...
   ```

   A running daemon can also load one at runtime:
   `client.load_plugin({"module": "mypkg.dataflow_plugin"})` — the
   reply's `kinds_registered` lists any NEW resolver kinds the import
   registered ([engine_service.md](engine_service.md)).

Programmatic use (your own scripts) needs neither: import your plugin
module, then call the dataflow APIs directly.

Both paths are pinned by `tests/dataflow_training/training/test_plugins.py`
(explicit `load_plugins(explicit=[...])` + a stubbed
`dataflow.families` entry point) and by the toy-family gate.

## After it's built: validate -> verify -> serve

```bash
# 0. structural contract (seconds, no GPU math)
python - <<'PY'
import mypkg.dataflow_plugin  # registers
from dataflow_training.model_families.families import validate_family
assert validate_family("mymodel") == [], validate_family("mymodel")
PY

# 1. correctness: per-op, per-task (fwd/recompute/bwd), per-model —
#    your test module follows the ladder canon (extending.md §8);
#    verify_family runs it + the contract check + the coverage audit
python tools/verify_family.py --plugin mypkg.dataflow_plugin \
    --family mymodel --module mypkg/tests/test_mymodel.py

# 2. throughput sweeps (register_bench_config names work everywhere)
python tools/measure_step.py --plugin mypkg.dataflow_plugin \
    --presets mymodel-mini --device-gib 12,16,20 --shapes oracle --run \
    --out-dir results/bench/mymodel
```

Serving it through the daemon is the same seam every builtin uses
(`tests/test_external_family.py` is the executable version of this
snippet):

```python
import dataclasses
import mypkg.dataflow_plugin                       # registers the family
from dataflow_training.register import register_all
register_all()                                     # registers kind "model_family"

spec = {"kind": "model_family", "family": "mymodel",
        "cfg": dataclasses.asdict(cfg)}            # see KNOWN GAPS on cfg_dict
# register_program(program, resolver=spec) + init_model + run — the
# family resolves out of the box, family_init included.
```

## What your package implements (all public imports)

The content is identical to `extending.md` — only the file locations
move into your package:

| builtin location | yours | imports you use |
|---|---|---|
| `model_families/<family>/model.py` | `mypkg/mymodel_model.py` | `build_shaped_program`, `LayerKindSpec`, `AuxShare`, `FamilyLayouts`, `apply_exact_sizes`, `object_size_factory`, `initial_values_from_layouts` — the whole lowering toolkit is importable |
| `model_families/<family>/blocks.py` | `mypkg/mymodel_blocks.py` | your `launch(ctx)` executables + resolver; reuse `blocks.base_blocks` templates (`EmbedFwd`/`HeadLoss`/`EmbedBwd`/`OptimizerStep`), `blocks.layouts`, `kernels.resolve_kernels` |
| `model_families/<family>/bridge.py` + `reference_models/<family>.py` | `mypkg/mymodel_reference.py` (+ bridge) | your own isolated pure-torch twin + a bytes->state_dict bridge, if you want the parity treatment (docs/correctness_compare.md) |
| `model_families/<family>/presets.py` | `mypkg/mymodel_presets.py` | preset classmethods on your config; `register_bench_config` for named bench shapes |
| `kernels/<op>.py` | `mypkg/mymodel_kernels.py` | the registry decorator ABI (`dataflow_training.kernels.registry`) — your fused impls join the kernel-set stamp |
| `model_families/families.py` entry | the plugin module above | `register_family` |
| `tests/dataflow_training/models/test_<family>.py` | `mypkg/tests/test_mymodel.py` | copy the NEWEST builtin family's module as the template; `check_block_backward` / `check_model_step` import from `dataflow_training.testing.gradcheck` |

Config rule, relaxed for external families: `resolve_family` dispatches
EXACT type first, then isinstance — so your config MAY subclass a
builtin config (convenient when your arch is a variant of one; the
stub plugin in `test_plugins.py` does exactly this over
`ShapedLlamaConfig`). Distinct dataclasses remain the cleaner default.

## Known gaps

Real, current, and worked around by the fixture — check here before
debugging a mysterious plugin failure:

- **Closed wire-spec tables.** `dataflow_training/run/presets.py` keys
  `CFG_DICT_BY_TYPE` / `RESOLVER_FAMILY_BY_TYPE` by BUILTIN config-type
  name, and `ModelFamily.cfg_dict()` / `.resolver_spec()` route through
  them — for an external family they raise `KeyError`. Until the tables
  open up, build the wire spec yourself via
  `dataclasses.asdict(cfg)`:
  `{"kind": "model_family", "family": "mymodel", "cfg": asdict(cfg)}`
  (exactly what `tests/test_external_family.py` does). Every cfg field
  must be a constructor kwarg of your config type — the daemon rebuilds
  `config_type(**cfg)`.
- **`InitialValuesFn`'s service contract is wider than its Protocol.**
  The Protocol documents `(program, cfg, backend, seed)`, but the
  daemon's init-as-program executable
  (`dataflow_training/register.py`, `FamilyInitExecutable`) calls
  `initial_values(program, cfg, None, seed=seed, into=buffers)` — and
  with `tp_view=` when a tensor-parallel view rides the init task's
  `block_params`. Your `initial_values` MUST accept `into=` (write into
  the provided pinned buffers) to serve the service path; accept
  `tp_view=` only if you support TP fills. Routing through
  `initial_values_from_layouts` (which takes `into=`) gets you this for
  free — the fixture's `toy_initial_values` does.
- **`build_resolver`'s `hyper` is positional.** When the wire spec
  carries `"hyper"`, the service build calls
  `fam.build_resolver(dims, build_hyper(hyper))` — a SECOND POSITIONAL
  argument the `BuildResolverFn` Protocol doesn't declare. Sign your
  builder as `build_resolver(dims, hyper=None)` (the fixture does) or
  hyper-carrying specs will crash resolution.
- **No public executable base.** The family-neutral launch machinery
  (layer parsing, acc closures, context plumbing) lives on `_Base` in
  `blocks/base_blocks.py` — a private name with no stability promise.
  Public reuse is the CONCRETE templates (`EmbedFwd`/`HeadLoss`/
  `EmbedBwd`/`OptimizerStep`) plus writing your block `launch(ctx)`
  from scratch against the `TaskContext` ABI (the fixture's
  `ToyBlockFwd`/`ToyBlockBwd` show the full pattern, positional buffer
  binding included).
- **The shaped-program metadata stamp is unconditional.**
  `build_shaped_program` stamps `n_heads`/`n_kv_heads`/`d_ff` (with the
  rest of the config block) into `Program.metadata["config"]` for every
  family — so every config must CARRY those fields even when the
  architecture has no use for them (the fixture's attention-free
  `ToyConfig` carries vestigial `n_heads=1`/`n_kv_heads=1`).

## Current limitations (fork-only edges)

Small, cosmetic, and shrinking — none block a working external family:

- planner/tooling name regexes assert full task-name coverage;
  keep the `prefix_{step}_{round}_{layer}` naming shape (any prefix)
  and they hold.
- The builtin lowering-stability tripwire file
  (`tests/dataflow_training/training/test_lowering_stability.py`)
  doesn't import plugin families; pin your hash in your own test
  module.
- `verify_family`'s canon audit scans shared builtin op-suite modules
  for coverage credit; external ops' pins should live in your one test
  module (simpler anyway).

If your model's STRUCTURE doesn't fit the standard training chain at
all (no forward pass, custom schedules, arbitrary DAGs):
[extending_programs.md](extending_programs.md). The seam underneath
all of this: [program_contract.md](program_contract.md).

### The `acc` contract (frozen-safe weight gradients)

Every block backward writes weight gradients through the `acc(name,
value)` closure (`blocks/base_blocks.py`), and this is a FREEZE
contract, not just a convenience:

- `acc` SKIPS the write for any field absent from the (policy-filtered)
  dW layout, and is a no-op when the layer has no dW at all — frozen
  fields can never crash a backward or corrupt storage.
- wgrads with their OWN standalone cost (the `X.T @ dY` GEMMs) must be
  guarded at the call site: `if acc.wanted("wq"): acc("wq", h1.T @ dq)`
  — frozen fields then skip the COMPUTATION, not just the write.
- BYPRODUCT gradients (norm weights, biases, fla-kernel side outputs)
  call `acc` bare: they fall out of fused dgrad kernels at negligible
  cost, so there is nothing to skip — the write-skip is the whole
  story.

New block code MUST follow this split; the freeze gates
(`tests/dataflow_training/training/test_freeze_plan.py`) exercise both
paths.
