"""Enumerate every family's task kinds (compute keys -> executables)
as a markdown table — the generator behind docs/task_kinds.md.

    python tools/list_tasks.py > docs/task_kinds.md

For each builtin family: lower the tiny preset, collect the unique
compute_block_keys the chain emits (forward/recompute/backward per
layer kind, embed, head, optimizer), resolve each through the family's
resolver, and report the executable class with the first line of its
docstring. This is the task-side counterpart of
docs/kernel_registry.md — an INVENTORY, not a dispatch registry:
dispatch stays the resolver ABI (`task -> executable`), which is what
lets custom programs compose family executables with their own
(docs/extending_programs.md).
"""
from __future__ import annotations

import inspect

from dataflow.training import families as F

HEADER = """# Task kinds: compute keys and executables

GENERATED — regenerate with `python tools/list_tasks.py >
docs/task_kinds.md` after adding a family or layer kind. Dispatch is
the resolver ABI (`resolver(task) -> executable.launch(ctx)`, keyed on
`task.compute_block_key`) — family-scoped by design; this table is the
fleet inventory. Buffer contracts (positional input/output order per
key) are documented next to each executable class; recompute keys are
derived from the forward stage lists, never hand-written
(docs/extending.md §2).
"""


def _doc(obj) -> str:
    """First REAL docstring line up the MRO (dataclasses synthesize a
    signature-shaped __doc__ when none is written — skip those)."""
    import re as _re

    for cls in type(obj).__mro__:
        if cls is object:
            break
        d = cls.__doc__
        if not d:
            continue
        line = d.strip().splitlines()[0].rstrip(".")
        if _re.match(rf"^{cls.__name__}\(", line):
            continue          # synthesized dataclass signature
        return line if len(line) <= 96 else line[:93] + "..."
    return "—"


def main() -> None:
    F.load_plugins()
    print(HEADER)
    print("| family | compute key | group | executable | description |")
    print("|---|---|---|---|---|")
    for name in F._FAMILIES:
        fam = F.family(name)
        cfg = fam.config_type.tiny()
        dims = fam.dims_of(cfg)
        # force recompute levels so recompute keys appear? rc tasks are
        # planner-inserted; derive their keys from the fwd keys instead
        prog = fam.lower(cfg)
        resolver = fam.build_resolver(dims)
        seen: dict[tuple[str, str], object] = {}
        for task in prog.task_by_id().values():
            key = (task.compute_block_key, task.group or "—")
            if key not in seen:
                seen[key] = resolver(task)
        rows = []
        for (ck, group), ex in sorted(seen.items()):
            rows.append((ck, group, type(ex).__name__, _doc(ex)))
            if ck.endswith("_fwd") and not ck.startswith(("embed",)):
                rc = ck.replace("_fwd", "_recompute")
                if (rc, "recompute") not in seen and rc != ck:
                    import dataclasses

                    t0 = next(t for t in prog.task_by_id().values()
                              if t.compute_block_key == ck)
                    rt = dataclasses.replace(t0, compute_block_key=rc)
                    try:
                        rex = resolver(rt)
                        rows.append((rc, "recompute (planner/derived)",
                                     type(rex).__name__, _doc(rex)))
                    except Exception:
                        pass
        for ck, group, cls, doc in sorted(rows):
            print(f"| {name} | `{ck}` | {group} | `{cls}` | {doc} |")


if __name__ == "__main__":
    main()
