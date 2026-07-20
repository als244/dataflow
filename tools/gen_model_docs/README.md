# tools/gen_model_docs/ — generated documentation

Five separate CLIs, one per generated artifact. Everything derives
from the code — no hand-maintained content; regenerate after adding
a family, preset, kernel, or task kind.

## gen_model_docs.py — the per-preset deep references

Writes `docs/models/<family>/<preset>_16x4K.md` (every family ×
preset at the standard documentation run shape) plus the index:
dims, per-object and aggregate sizes, field-level W/A/AuxTemp
tables, every task kind's buffer contract + STAGES + traced kernel
sequence.

    python tools/gen_model_docs/gen_model_docs.py [--family X [--preset P]]

`--no-record` skips kernel tracing (CPU-only machines);
`--out-dir` overrides the destination.

## gen_model_page.py — one page at an arbitrary run shape

    python tools/gen_model_docs/gen_model_page.py --preset glm52_mini \
        --microbatch 8 --seq-len 2048

`--family` disambiguates shared preset names (e.g. `tiny`);
`--record/--no-record`, `--out-dir` as above.

## list_models.py / list_kernels.py / list_tasks.py — stdout tables

    python tools/gen_model_docs/list_models.py  > docs/builtin_models.md
    python tools/gen_model_docs/list_kernels.py > docs/kernel_registry.md
    python tools/gen_model_docs/list_tasks.py   > docs/task_kinds.md

No flags: the family/preset table (params computed from lowered
layouts), the kernel-registry inventory (impl priorities, resolved
set, determinism/workspace/alloc columns), and the task-kind table
(compute key → executable per family). Plugins load first, so
external families appear automatically.
