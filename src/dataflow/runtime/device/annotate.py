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

    def start_capture(self) -> None:  # pragma: no cover - trivial
        pass

    def stop_capture(self) -> None:  # pragma: no cover - trivial
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

    def start_capture(self) -> None:
        self.events.append(("start_capture", None))

    def stop_capture(self) -> None:
        self.events.append(("stop_capture", None))

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

    def start_capture(self) -> None:
        """cudaProfilerStart: with nsys --capture-range=cudaProfilerApi
        recording begins HERE (the conductor brackets chosen steps).
        AMD impl slot: rocprofiler's start/stop pair behind these same
        two methods — engine and conductor code never see the vendor."""
        from cuda.bindings import runtime as cudart

        cudart.cudaProfilerStart()

    def stop_capture(self) -> None:
        from cuda.bindings import runtime as cudart

        cudart.cudaProfilerStop()

    def start_capture(self) -> None:
        """cudaProfilerStart: with nsys --capture-range=cudaProfilerApi
        recording begins HERE (the conductor brackets chosen steps).
        AMD impl slot: rocprofiler's start/stop pair behind the same
        two methods — engine and conductor code never know the vendor."""
        from cuda.bindings import runtime as cudart

        cudart.cudaProfilerStart()

    def stop_capture(self) -> None:
        from cuda.bindings import runtime as cudart

        cudart.cudaProfilerStop()


class SwitchableAnnotator:
    """The default annotator: OFF (pure no-ops) until a profiler
    capture activates it, ON only within the profiled window.
    start_capture() lazily builds the vendor annotator (NVTX here;
    an AMD rocprofiler/roctx impl slots in behind the same methods),
    begins the vendor capture (pairs with nsys
    --capture-range=cudaProfilerApi), and enables range annotations;
    stop_capture() ends the capture and returns every range call to
    a no-op. No environment variables involved."""

    def __init__(self) -> None:
        self.active = False
        self.vendor = None

    @property
    def enabled(self) -> bool:
        return self.active

    def range_push(self, name: str) -> None:
        if self.active:
            self.vendor.range_push(name)

    def range_pop(self) -> None:
        if self.active:
            self.vendor.range_pop()

    def mark(self, name: str) -> None:
        if self.active:
            self.vendor.mark(name)

    def start_capture(self) -> None:
        if self.vendor is None:
            self.vendor = NvtxAnnotator()
        self.active = True
        self.vendor.start_capture()

    def stop_capture(self) -> None:
        if self.vendor is not None:
            self.vendor.stop_capture()
        self.active = False


def annotator_from_env():
    """Kept for call-site compatibility: annotations are OFF by
    default and switch on only inside a profiled step window —
    the old DATAFLOW_NVTX env gate is gone."""
    return SwitchableAnnotator()
