"""Kernel registry: named ops, swappable implementations, pinned selection.

Each *op* (e.g. "swiglu_bwd") has a fixed call signature and one shared
autograd-able reference; each *implementation* is an opaque callable plus
metadata. How an implementation is built — eager torch, Triton, CuTe DSL,
TileLang, a ctypes binding to a .so, a cuda-python cubin launch — is
invisible here: if it honors the ABI below, it registers.

ABI (the contract every implementation signs):
- ``fn(kctx, *args)``: enqueue ALL work, return immediately. No syncs, no
  event waits, no launches on other streams.
- Stream: torch-based impls may rely on the ambient torch stream (executables
  run under ``torch.cuda.stream(...)``); foreign toolchains use
  ``kctx.stream_handle`` (raw vendor handle). Never the default stream.
- Tensors arrive as contiguous torch views over runtime-owned buffers; take
  ``.data_ptr()`` freely, retain nothing past the call.
- Workspace is *declared*, not improvised: ``none`` | ``arena(bytes_fn)``
  (impl receives ``kctx.workspace``) | ``internal(hint_fn)`` (impl allocates
  inside — legal, discouraged; the hint is validated against the profiled
  peak, and absent a hint the profiler's measurement becomes the declaration).
  ``allocates="vendor"`` flags impls that may implicit-sync (cudaMalloc).

Selection is PINNED once per resolve (override > requires(caps) > priority)
and never re-decided mid-run: measured task costs are measurements of a
specific implementation, so profiles and saved programs record
``KernelSet.describe()`` — re-running under a different resolution is loud,
not silent. ``deterministic`` gates the plan-invariance test mode.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

# --- workspace declarations --------------------------------------------------


@dataclass(frozen=True)
class Workspace:
    style: str                      # "none" | "arena" | "internal"
    bytes_fn: Callable | None = None  # (shapes...) -> bytes; hint for "internal"


def none() -> Workspace:
    return Workspace("none")


def arena(bytes_fn: Callable) -> Workspace:
    return Workspace("arena", bytes_fn)


def internal(hint_fn: Callable | None = None) -> Workspace:
    return Workspace("internal", hint_fn)


# --- ctx ----------------------------------------------------------------------


@dataclass
class KernelCtx:
    """Per-call context handed to every implementation."""

    stream_handle: int = 0          # raw vendor stream (foreign toolchains)
    torch_stream: object = None     # torch.cuda.Stream/ExternalStream or None
    workspace: object = None        # torch uint8 view when style == "arena"
    # packed-mode metadata for THIS launch: (seq_lens, cu_dev,
    # max_seqlen, positions_dev) — set by block launches from the
    # run_args prologue; None in static mode. Lives here because
    # KernelCtx is constructed fresh per launch (race-free by
    # construction) and already flows through every stage/backward
    # signature.
    pk: tuple | None = None


# --- registry ------------------------------------------------------------------


@dataclass(frozen=True)
class KernelEntry:
    op: str
    impl_id: str
    fn: Callable
    deterministic: bool
    workspace: Workspace
    requires: Callable              # (caps: dict) -> bool
    priority: int                   # higher wins among available impls
    allocates: str = "none"         # "none" | "torch" | "vendor" (may sync!)


_REGISTRY: dict[str, dict[str, KernelEntry]] = {}


def register(
    op: str,
    impl_id: str,
    *,
    deterministic: bool,
    workspace: Workspace,
    requires: Callable = lambda caps: True,
    priority: int = 0,
    allocates: str = "none",
    fn: Callable | None = None,
):
    """Register an implementation; usable as decorator or direct call."""

    def _add(f: Callable) -> Callable:
        entry = KernelEntry(
            op=op, impl_id=impl_id, fn=f, deterministic=deterministic,
            workspace=workspace, requires=requires, priority=priority,
            allocates=allocates,
        )
        impls = _REGISTRY.setdefault(op, {})
        if impl_id in impls:
            raise ValueError(f"duplicate registration: {op}:{impl_id}")
        impls[impl_id] = entry
        return f

    return _add(fn) if fn is not None else _add


def registered(op: str) -> dict[str, KernelEntry]:
    return dict(_REGISTRY.get(op, {}))


# --- resolution ------------------------------------------------------------------


class KernelSet:
    """A pinned op -> implementation mapping. Blocks call ops through this;
    the choice never changes after resolve (cost-measurement integrity)."""

    def __init__(self, entries: dict[str, KernelEntry]):
        self._entries = entries
        # bound fast-path: k.swiglu_bwd(kctx, ...) with zero dict lookups
        for op, entry in entries.items():
            setattr(self, op, entry.fn)

    def entry(self, op: str) -> KernelEntry:
        return self._entries[op]

    def describe(self) -> dict[str, str]:
        """op -> impl_id; stamped into profiles/reports for provenance."""
        return {op: e.impl_id for op, e in sorted(self._entries.items())}

    def all_deterministic(self) -> bool:
        return all(e.deterministic for e in self._entries.values())


def device_caps() -> dict:
    """Capabilities used by ``requires`` gates. Cheap, import-light."""
    caps: dict = {"cuda": False, "triton": False}
    try:
        import torch

        caps["cuda"] = torch.cuda.is_available()
        if caps["cuda"]:
            caps["cc"] = torch.cuda.get_device_capability()
    except Exception:
        return caps
    try:
        import triton  # noqa: F401

        caps["triton"] = True
    except Exception:
        pass
    caps["flash_mla"] = False
    if caps["cuda"] and caps.get("cc", (0, 0))[0] >= 9:
        try:
            import flash_mla  # noqa: F401

            caps["flash_mla"] = True
        except Exception:
            pass
    return caps


def resolve_kernels(
    caps: dict | None = None,
    overrides: dict[str, str] | None = None,
) -> KernelSet:
    """Pin one implementation per op: override > requires(caps) > priority.

    ``DATAFLOW_KERNELS=eager`` (or any impl_id) forces that impl_id for every
    op that has it — the numerics-bisection switch.
    """
    caps = device_caps() if caps is None else caps
    overrides = dict(overrides or {})
    forced = os.environ.get("DATAFLOW_KERNELS")

    chosen: dict[str, KernelEntry] = {}
    for op, impls in _REGISTRY.items():
        want = overrides.get(op) or (forced if forced in impls else None)
        if want is not None:
            if want not in impls:
                raise KeyError(f"no implementation {op}:{want} registered")
            entry = impls[want]
            if not entry.requires(caps):
                raise RuntimeError(f"{op}:{want} not available on this device")
            chosen[op] = entry
            continue
        avail = [e for e in impls.values() if e.requires(caps)]
        if not avail:
            raise RuntimeError(f"no available implementation for {op!r}")
        chosen[op] = max(avail, key=lambda e: (e.priority, e.impl_id))
    return KernelSet(chosen)
