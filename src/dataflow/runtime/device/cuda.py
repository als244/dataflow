"""Real CUDA DeviceBackend over cuda-python (cuda.bindings.runtime).

Completion tokens: the engine registers (event, token, priority) per stream;
`next_completion` resolves them in true completion order. Two delivery modes:

- ``poll`` (default): the control thread polls each stream's oldest pending
  event with cudaEventQuery (events on one stream complete in order, so only
  heads are polled). Measured on the RTX 5090: ~1-3 us per query, giving
  ~us-scale wake latency at the cost of spinning while blocked.
- ``hostfn``: cudaLaunchHostFunc pushes tokens from CUDA's callback thread
  (measured ~160-270 us delivery latency on this machine). Kept for
  comparison; the engine gate reports both.

Timebase: an origin event recorded+synced at first stream creation;
`event_time_us` = cudaEventElapsedTime(origin, ev) in microseconds. All
events are created with timing enabled.

Steady-state discipline: no cudaMalloc/cudaFree after warmup (the BufferPool
holds buffers; allocation happens during initial-object load), no
synchronizing calls on the hot path (cudaEventQuery/ElapsedTime only).
"""
from __future__ import annotations

import ctypes
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from cuda.bindings import runtime as cudart

from .base import Buffer, Event, Location, Stream, StreamKind


class CudaError(RuntimeError):
    pass


def _check(result: tuple) -> tuple:
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        name = cudart.cudaGetErrorName(err)[1] if isinstance(err, cudart.cudaError_t) else err
        raise CudaError(f"CUDA call failed: {err} ({name})")
    return result[1:]


_HOSTFN_CFUNC_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


@dataclass
class _Pending:
    event: Event
    token: Any
    priority: int


@dataclass(frozen=True)
class PcieBandwidth:
    """bytes/us in each direction, measured alone and under concurrent load."""

    uni_h2d: int
    uni_d2h: int
    bidi_h2d: int
    bidi_d2h: int


@dataclass
class CudaBackend:
    name: str = "cuda"
    physical: bool = True
    device: int = 0
    completion_mode: str = "poll"  # "poll" | "hostfn"
    poll_yield: bool = True        # sleep(0) between poll sweeps

    _streams: list[Stream] = field(default_factory=list)
    _pending: dict[str, deque[_Pending]] = field(default_factory=dict)
    _origin: Any = None            # cudaEvent_t reference point (t=0)
    _t0_host: float = 0.0
    _seq: int = 0
    # hostfn mode state
    _hostfn_queue: "queue.SimpleQueue[int]" = field(default_factory=queue.SimpleQueue)
    _hostfn_tokens: dict[int, _Pending] = field(default_factory=dict)
    _hostfn_outstanding: int = 0
    _hostfn_lock: threading.Lock = field(default_factory=threading.Lock)
    _hostfn_cfunc: Any = None
    _hostfn_ptr: Any = None
    # diagnostics
    events_created: int = 0

    def __post_init__(self) -> None:
        _check(cudart.cudaSetDevice(self.device))
        _check(cudart.cudaFree(0))  # establish context eagerly
        self._t0_host = time.perf_counter()
        if self.completion_mode == "hostfn":
            self._hostfn_cfunc = _HOSTFN_CFUNC_TYPE(self._on_hostfn)
            addr = ctypes.cast(self._hostfn_cfunc, ctypes.c_void_p).value
            self._hostfn_ptr = cudart.cudaHostFn_t(addr)

    # --- streams & events ---------------------------------------------------
    def create_stream(self, kind: StreamKind) -> Stream:
        (raw,) = _check(cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking))
        self._seq += 1
        stream = Stream(id=f"{kind}:{self._seq}", kind=kind, raw=raw)
        self._streams.append(stream)
        self._pending[stream.id] = deque()
        if self._origin is None:
            (origin,) = _check(cudart.cudaEventCreate())
            _check(cudart.cudaEventRecord(origin, raw))
            _check(cudart.cudaEventSynchronize(origin))
            self._origin = origin
        return stream

    def record_event(self, stream: Stream) -> Event:
        (raw,) = _check(cudart.cudaEventCreate())  # timing-enabled default
        _check(cudart.cudaEventRecord(raw, stream.raw))
        self._seq += 1
        self.events_created += 1
        return Event(id=f"ev{self._seq}", raw=raw)

    def stream_wait_event(self, stream: Stream, event: Event) -> None:
        _check(cudart.cudaStreamWaitEvent(stream.raw, event.raw, 0))

    def align_stream_to_host(self, stream: Stream) -> None:
        return  # physical time: enqueued work can't start before enqueue

    def event_time_us(self, event: Event) -> float:
        (ms,) = _check(cudart.cudaEventElapsedTime(self._origin, event.raw))
        return float(ms) * 1e3

    # --- memory ---------------------------------------------------------------
    pinned_bytes: int = 0
    pinned_peak: int = 0

    @property
    def annotator(self):
        # cached; DATAFLOW_NVTX=1 enables NVTX ranges (nsys wrapper sets it)
        a = getattr(self, "_annotator", None)
        if a is None:
            from .annotate import annotator_from_env

            a = annotator_from_env()
            self._annotator = a
        return a

    def alloc(self, location: Location, size_bytes: int) -> Buffer:
        if location == "fast":
            (ptr,) = _check(cudart.cudaMalloc(size_bytes))
            raw = None
        else:
            ptr, raw = self._alloc_pinned(size_bytes)
        self._seq += 1
        return Buffer(
            id=f"buf{self._seq}", location=location, size_bytes=size_bytes,
            ptr=int(ptr), raw=raw,
        )

    def _alloc_pinned(self, size_bytes: int):
        """cudaHostAlloc, page-granular (no power-of-2 rounding — that lore is
        torch's pinned CACHING allocator, which this path bypasses entirely).

        malloc + madvise(HUGEPAGE) + cudaHostRegister was measured as the
        alternative on this system (THP=madvise honored): pin throughput was
        IDENTICAL at 8 GB (~4.5 GB/s — the driver pins at its own granularity
        regardless of backing page size) and 10x slower for small buffers, so
        the simpler call stays. Revisit if a driver ever honors huge-page
        registration."""
        (ptr,) = _check(cudart.cudaHostAlloc(size_bytes, cudart.cudaHostAllocDefault))
        self.pinned_bytes += size_bytes
        self.pinned_peak = max(self.pinned_peak, self.pinned_bytes)
        return int(ptr), ("hostalloc", size_bytes)

    def free(self, buffer: Buffer) -> None:
        if buffer.location == "fast":
            _check(cudart.cudaFree(buffer.ptr))
        else:
            _check(cudart.cudaFreeHost(buffer.ptr))
            if isinstance(buffer.raw, tuple) and buffer.raw[0] == "hostalloc":
                self.pinned_bytes -= buffer.raw[1]

    # --- async work -----------------------------------------------------------
    def memcpy_async(
        self,
        dst: Buffer,
        src: Buffer,
        size_bytes: int,
        stream: Stream,
        *,
        duration_us: float | None = None,
    ) -> None:
        del duration_us  # physical copies take physical time
        if src.location == "fast" and dst.location == "backing":
            kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        elif src.location == "backing" and dst.location == "fast":
            kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        elif src.location == "fast" and dst.location == "fast":
            kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        else:
            kind = cudart.cudaMemcpyKind.cudaMemcpyHostToHost
        _check(cudart.cudaMemcpyAsync(dst.ptr, src.ptr, size_bytes, kind, stream.raw))

    def memset_async(self, buffer: Buffer, value: int, stream: Stream) -> None:
        _check(cudart.cudaMemsetAsync(buffer.ptr, value, buffer.size_bytes, stream.raw))

    def event_complete(self, event: Event) -> bool:
        err = cudart.cudaEventQuery(event.raw)[0]
        if err == cudart.cudaError_t.cudaSuccess:
            return True
        if err == cudart.cudaError_t.cudaErrorNotReady:
            return False
        raise CudaError(f"cudaEventQuery failed: {err}")

    def advance_stream(self, stream: Stream, duration_us: float) -> tuple[float, float]:
        raise CudaError(
            "advance_stream models virtual time and is fake-backend-only; real "
            "runs need real executables (e.g. the calibrated spin kernel)"
        )

    # --- completion tokens ------------------------------------------------------
    def notify_after(self, stream: Stream, event: Event, token: Any, *, priority: int) -> None:
        pending = _Pending(event=event, token=token, priority=priority)
        if self.completion_mode == "hostfn":
            with self._hostfn_lock:
                self._seq += 1
                key = self._seq
                self._hostfn_tokens[key] = pending
                self._hostfn_outstanding += 1
            _check(cudart.cudaLaunchHostFunc(stream.raw, self._hostfn_ptr, key))
        else:
            self._pending[stream.id].append(pending)

    def _on_hostfn(self, user_data: int) -> None:
        # Runs on CUDA's callback thread: queue push only, no CUDA calls.
        self._hostfn_queue.put(int(user_data) if user_data is not None else 0)

    def next_completion(self) -> Any | None:
        if self.completion_mode == "hostfn":
            with self._hostfn_lock:
                outstanding = self._hostfn_outstanding
            if outstanding == 0:
                return None
            key = self._hostfn_queue.get()
            with self._hostfn_lock:
                pending = self._hostfn_tokens.pop(key)
                self._hostfn_outstanding -= 1
            return pending.token

        # poll mode: only stream heads can complete next (in-order per stream)
        if all(not dq for dq in self._pending.values()):
            return None
        while True:
            best: tuple[float, int, str] | None = None
            for stream_id, dq in self._pending.items():
                if not dq:
                    continue
                head = dq[0]
                err = cudart.cudaEventQuery(head.event.raw)[0]
                if err == cudart.cudaError_t.cudaSuccess:
                    t = self.event_time_us(head.event)
                    cand = (t, head.priority, stream_id)
                    if best is None or cand < best:
                        best = cand
                elif err != cudart.cudaError_t.cudaErrorNotReady:
                    raise CudaError(f"cudaEventQuery failed: {err}")
            if best is not None:
                return self._pending[best[2]].popleft().token
            if self.poll_yield:
                time.sleep(0)

    def measure_pcie(self, nbytes: int = 512 * 1024 * 1024) -> "PcieBandwidth":
        """Measure pinned-copy bandwidth in both regimes (bytes/us ints).

        On this class of platform the two directions are NOT independent:
        concurrent h2d+d2h can collapse to ~half each (shared host-memory /
        link budget), and h2d may be slower than d2h even alone. Plans should
        normally use the bidirectional numbers — under memory pressure both
        directions run concurrently most of the time.
        """
        s1 = self.create_stream("h2d")
        s2 = self.create_stream("d2h")
        dev1, dev2 = self.alloc("fast", nbytes), self.alloc("fast", nbytes)
        host1, host2 = self.alloc("backing", nbytes), self.alloc("backing", nbytes)
        ctypes.memset(host1.ptr, 0x5A, nbytes)  # touch pages before DMA reads
        ctypes.memset(host2.ptr, 0xA5, nbytes)

        def one(direction_pairs: list[tuple[Buffer, Buffer, Stream]]) -> list[float]:
            events = []
            for dst, src, stream in direction_pairs:
                a = self.record_event(stream)
                self.memcpy_async(dst, src, nbytes, stream)
                b = self.record_event(stream)
                events.append((a, b))
            _check(cudart.cudaDeviceSynchronize())
            return [
                nbytes / (float(_check(cudart.cudaEventElapsedTime(a.raw, b.raw))[0]) * 1e3)
                for a, b in events
            ]

        # warmup + median of 3
        uni_h2d, uni_d2h, bidi_h2d, bidi_d2h = [], [], [], []
        for i in range(4):
            (h,) = one([(dev1, host1, s1)])
            (d,) = one([(host2, dev2, s2)])
            b_h, b_d = one([(dev1, host1, s1), (host2, dev2, s2)])
            if i > 0:
                uni_h2d.append(h); uni_d2h.append(d); bidi_h2d.append(b_h); bidi_d2h.append(b_d)
        import statistics as _st
        result = PcieBandwidth(
            uni_h2d=int(_st.median(uni_h2d)), uni_d2h=int(_st.median(uni_d2h)),
            bidi_h2d=int(_st.median(bidi_h2d)), bidi_d2h=int(_st.median(bidi_d2h)),
        )
        for buf in (dev1, dev2, host1, host2):
            self.free(buf)
        return result

    def host_now_us(self) -> float:
        return (time.perf_counter() - self._t0_host) * 1e6

    def mark_origin(self) -> None:
        if not self._streams:
            return
        (origin,) = _check(cudart.cudaEventCreate())
        _check(cudart.cudaEventRecord(origin, self._streams[0].raw))
        _check(cudart.cudaEventSynchronize(origin))
        self._origin = origin
        self._t0_host = time.perf_counter()

    def sync_all(self) -> None:
        _check(cudart.cudaDeviceSynchronize())
