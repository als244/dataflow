"""Profiler annotation abstraction: named ranges over the execution timeline.

Vendor-portable by the same rule as DeviceBackend: the runtime only calls
this ~3-method protocol, so a vendor implementation (NVTX here; AMD's roctx
exposes the identical push/pop shape) plugs in without touching engine code.

Semantics note: NVTX/roctx ranges are HOST-THREAD scoped, not stream
scoped. The engine opens a range exactly around one task's (or transfer's)
enqueue onto one stream, so the profiler's projection view attributes each
range to the stream that executed the work — which is the per-stream
timeline people actually read in nsys ("NVTX projected onto GPU").

Annotation is off by default (a push/pop pair costs ~1 us of host time and
strict-pacing dispatch is host-latency sensitive); the nsys wrapper enables
it via DATAFLOW_NVTX=1.
"""
from __future__ import annotations

import ctypes
import glob
import os


class NoopAnnotator:
    """Default: zero-cost stubs."""

    enabled = False

    def range_push(self, name: str) -> None:  # pragma: no cover - trivial
        pass

    def range_pop(self) -> None:  # pragma: no cover - trivial
        pass

    def mark(self, name: str) -> None:  # pragma: no cover - trivial
        pass


class RecordingAnnotator:
    """Test double: records the range stream and checks pairing."""

    enabled = True

    def __init__(self) -> None:
        self.events: list[tuple[str, str | None]] = []
        self.depth = 0
        self.max_depth = 0

    def range_push(self, name: str) -> None:
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        self.events.append(("push", name))

    def range_pop(self) -> None:
        assert self.depth > 0, "range_pop without matching push"
        self.depth -= 1
        self.events.append(("pop", None))

    def mark(self, name: str) -> None:
        self.events.append(("mark", name))


_NVTX_CANDIDATES = (
    os.environ.get("DATAFLOW_NVTX_LIB", ""),
    "libnvToolsExt.so.1",
    "libnvToolsExt.so",
)


def _find_nvtx() -> ctypes.CDLL | None:
    paths = [p for p in _NVTX_CANDIDATES if p]
    paths += sorted(glob.glob("/usr/local/cuda*/lib64/libnvToolsExt.so.1"))
    paths += sorted(glob.glob("/opt/cuda*/lib64/libnvToolsExt.so.1"))
    for path in paths:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


class NvtxAnnotator:
    """NVTX via ctypes (no torch, no extra deps). Raises if the library is
    unavailable — callers decide whether that is fatal (the nsys wrapper
    treats it as fatal; ad-hoc use falls back to Noop)."""

    enabled = True

    def __init__(self) -> None:
        lib = _find_nvtx()
        if lib is None:
            raise OSError(
                "libnvToolsExt not found (searched loader path and "
                "/usr/local/cuda*/lib64); set DATAFLOW_NVTX_LIB=/path/to/"
                "libnvToolsExt.so.1"
            )
        self._push = lib.nvtxRangePushA
        self._push.argtypes = [ctypes.c_char_p]
        self._pop = lib.nvtxRangePop
        self._mark = lib.nvtxMarkA
        self._mark.argtypes = [ctypes.c_char_p]

    def range_push(self, name: str) -> None:
        self._push(name.encode())

    def range_pop(self) -> None:
        self._pop()

    def mark(self, name: str) -> None:
        self._mark(name.encode())


def annotator_from_env():
    """DATAFLOW_NVTX=1 -> NVTX (fatal if lib missing: you asked for it);
    unset -> Noop."""
    if os.environ.get("DATAFLOW_NVTX") == "1":
        return NvtxAnnotator()
    return NoopAnnotator()
