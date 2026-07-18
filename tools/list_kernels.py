"""Enumerate the kernel registry as a markdown table — the generator
behind docs/kernel_registry.md.

    python tools/list_kernels.py > docs/kernel_registry.md

Per op: every registered implementation (impl id, priority,
deterministic flag, workspace style, allocator discipline), which impl
RESOLVES on this machine, the call signature (taken from the
highest-priority impl whose Python signature is introspectable — the
eager fallback is the reference ABI), and a description pulled from
the implementation's docstring or its defining module's docstring.
"""
from __future__ import annotations

import inspect

from dataflow_training.kernels import registry as R
from dataflow_training import kernels as K   # noqa: F401  (imports register all)

HEADER = """# Kernel registry

GENERATED — regenerate with `python tools/list_kernels.py >
docs/kernel_registry.md` after registering ops or implementations.
The registry (`tasks/kernels/registry.py`) selects, per op, the
highest-priority implementation whose `requires(caps)` passes on this
machine; `DATAFLOW_KERNELS=eager` forces the priority-0 fallbacks for
bisection. The chosen set is stamped into profiles (measured costs are
measurements of a SPECIFIC kernel set). Contract for what
implementations may do: docs/task-contract.md; adding ops:
docs/extending.md §1.

Column notes: *resolved* = the impl selected on the machine that
generated this doc; *ws* = declared workspace (none / internal
estimate); *alloc* = allocator discipline (`none` = no allocations in
the launch path, `torch` = op-internal torch scratch, measured by
profiling).
"""


def _sig(fn) -> str:
    try:
        s = str(inspect.signature(fn))
        return s if len(s) <= 100 else s[:97] + "..."
    except (TypeError, ValueError):
        return "(…)"


def _doc(entry) -> str:
    d = inspect.getdoc(entry.fn)
    if not d:
        mod = inspect.getmodule(entry.fn)
        d = inspect.getdoc(mod) if mod else None
    if not d:
        return "—"
    line = d.strip().splitlines()[0].rstrip(".")
    return line if len(line) <= 90 else line[:87] + "..."


def main() -> None:
    resolved = {}
    try:
        from dataflow_training.kernels import resolve_kernels

        resolved = resolve_kernels().describe()
    except Exception:
        pass
    print(HEADER)
    print("| op | impls (priority) | resolved | det | ws | alloc | signature | description |")
    print("|---|---|---|---|---|---|---|---|")
    for op in sorted(R._REGISTRY):
        entries = R._REGISTRY[op]
        by_prio = sorted(entries.values(), key=lambda e: -e.priority)
        impls = ", ".join(f"{e.impl_id}({e.priority})" for e in by_prio)
        best = by_prio[0]
        eager = entries.get("eager", best)
        det = "yes" if all(e.deterministic for e in by_prio) else "MIXED"
        ws = best.workspace.style
        alloc = best.allocates
        print(f"| `{op}` | {impls} | {resolved.get(op, '—')} | {det} | "
              f"{ws} | {alloc} | `{_sig(eager.fn)}` | {_doc(eager)} |")


if __name__ == "__main__":
    main()
