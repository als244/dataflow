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
    """A CUDA API call that returned an error."""


class KernelCompileError(RuntimeError):
    """A kernel that would not compile. The device is fine and the API call
    succeeded; the source did not build, so the compiler log is the payload."""


class DeviceMemoryError(MemoryError):
    """The device would not grant an allocation."""


class HostMemoryError(MemoryError):
    """The host would not PIN an allocation.

    Separate from CudaError because nothing is wrong with the device and the
    remedy is different: ask for a smaller slab, or free host memory. The
    driver reports both refusals with the same code, cudaErrorMemoryAllocation,
    since pinned host memory is allocated through CUDA — so a caller told only
    that code goes looking at the GPU, which is where this was going wrong."""


def host_available_bytes() -> int:
    """Host memory that could still be pinned, as the kernel accounts for it."""
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


# Host allocations go through mmap rather than cudaHostAlloc so a slab costs
# the bytes it asks for; see CudaBackend._alloc_pinned for what the rounding
# did. Flag values are the Linux x86-64 ABI's.
_PROT_READ, _PROT_WRITE = 0x1, 0x2
_MAP_PRIVATE, _MAP_ANONYMOUS = 0x02, 0x20
_MAP_FAILED = ctypes.c_void_p(-1).value
_LIBC = None


def libc():
    """ctypes handle on libc, with mmap/munmap typed for 64-bit pointers.

    Without the restype the return value is truncated to a C int, which turns
    a valid high address into a negative number and a working mapping into a
    phantom failure."""
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
        _LIBC.mmap.restype = ctypes.c_void_p
        _LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_long]
        _LIBC.munmap.restype = ctypes.c_int
        _LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return _LIBC


def mmap_anon(size_bytes: int):
    """A private anonymous mapping of exactly ``size_bytes``, or None."""
    ptr = libc().mmap(None, size_bytes, _PROT_READ | _PROT_WRITE,
                      _MAP_PRIVATE | _MAP_ANONYMOUS, -1, 0)
    if not ptr or ptr == _MAP_FAILED:
        return None
    return ptr


def munmap(ptr: int, size_bytes: int) -> None:
    libc().munmap(ctypes.c_void_p(ptr), size_bytes)


def pin_region(size_bytes: int) -> int:
    """Map and pin exactly ``size_bytes`` of host memory; returns the address.

    The one place host memory gets pinned. Both the runtime's buffer backend
    and the service's boot slab come through here, because when they each had
    their own copy the fix for the rounding landed in one of them and a real
    run went on paying for the other.

    Raises HostMemoryError on either half's refusal, naming which half."""
    ptr = mmap_anon(size_bytes)
    if ptr is None:
        raise HostMemoryError(
            f"could not map {size_bytes / 1024 ** 3:.1f} GiB of host memory "
            f"(MemAvailable {host_available_bytes() / 1024 ** 3:.1f} GiB)")
    err = cudart.cudaHostRegister(ptr, size_bytes, cudart.cudaHostRegisterDefault)
    if isinstance(err, tuple):
        err = err[0]
    if int(err) != 0:
        munmap(ptr, size_bytes)
        name = cudart.cudaGetErrorName(err)[1]
        if isinstance(name, bytes):
            name = name.decode()
        raise HostMemoryError(
            f"could not pin {size_bytes / 1024 ** 3:.1f} GiB of host memory: "
            f"cudaHostRegister returned {name} (MemAvailable "
            f"{host_available_bytes() / 1024 ** 3:.1f} GiB)")
    return int(ptr)


def unpin_region(ptr: int, size_bytes: int) -> None:
    """Unregister then unmap. In that order: handing the mapping back while
    the driver still holds the pages pinned leaves it pinning memory this
    process no longer owns."""
    _check(cudart.cudaHostUnregister(ptr))
    munmap(ptr, size_bytes)


import weakref

# every constructed backend, for test-teardown drains (weak: a backend
# dies with its last reference; drain_all only touches live ones)
_BACKENDS: "weakref.WeakSet" = weakref.WeakSet()


def drain_all_backends() -> int:
    """Release every outstanding allocation on every live CudaBackend —
    the raw-cudaMalloc counterpart of torch.cuda.empty_cache for test
    teardown. Returns total bytes released."""
    total = 0
    for be in list(_BACKENDS):
        total += be.drain()
    return total


def _check(result: tuple) -> tuple:
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        name = cudart.cudaGetErrorName(err)[1] if isinstance(err, cudart.cudaError_t) else err
        if isinstance(name, bytes):
            name = name.decode()
        raise CudaError(f"CUDA call failed: {name}")
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


@dataclass(eq=False)
class CudaBackend:
    name: str = "cuda"
    physical: bool = True
    device: int = 0
    completion_mode: str = "poll"  # "poll" | "hostfn"
    poll_yield: bool = True        # GIL yield during poll waits
    # Yield cadence: a bare per-iteration sleep(0) hands the GIL to the
    # daemon's other threads on EVERY sweep, and each re-acquisition can
    # wait up to the interpreter switch interval — measured ~54 us per
    # yield, ~115 yields per task boundary = milliseconds of dispatch
    # latency. Yielding at most once per this many seconds keeps service
    # threads responsive at ms granularity without the per-sweep GIL
    # roulette (the event-query C call also drops the GIL briefly each
    # sweep, so other threads are never fully starved between yields).
    poll_yield_interval_s: float = 0.001
    last_poll_yield_s: float = 0.0

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
    # live-allocation ledger: every alloc() registers here, free()
    # deregisters. drain() releases whatever is still outstanding — the
    # lifecycle backstop for callers that exit without freeing (test
    # teardown; raw cudaMalloc is invisible to torch's empty_cache)
    _live: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check(cudart.cudaSetDevice(self.device))
        _check(cudart.cudaFree(0))  # establish context eagerly
        self._t0_host = time.perf_counter()
        _BACKENDS.add(self)
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
        # cached; capture windows (profiler_control) switch annotations on
        a = getattr(self, "_annotator", None)
        if a is None:
            from .annotate import annotator_from_env

            a = annotator_from_env()
            self._annotator = a
        return a

    def alloc(self, location: Location, size_bytes: int) -> Buffer:
        if location == "fast":
            err, ptr = cudart.cudaMalloc(size_bytes)
            if err != cudart.cudaError_t.cudaSuccess:
                free, total = _check(cudart.cudaMemGetInfo())
                raise DeviceMemoryError(
                    f"could not allocate {size_bytes / 1024 ** 3:.1f} GiB of "
                    f"device memory (free {free / 1024 ** 3:.1f} GiB of "
                    f"{total / 1024 ** 3:.1f} GiB)")
            raw = None
        else:
            ptr, raw = self._alloc_pinned(size_bytes)
        self._seq += 1
        buf = Buffer(
            id=f"buf{self._seq}", location=location, size_bytes=size_bytes,
            ptr=int(ptr), raw=raw,
        )
        self._live[buf.id] = buf
        return buf

    def _alloc_pinned(self, size_bytes: int):
        """Anonymous mapping, then cudaHostRegister to pin it.

        NOT cudaHostAlloc, which rounds the request up to a power of two. That
        is invisible as waste until the request crosses a boundary, where it
        stops being waste and becomes a hard ceiling: measured on a 125.7 GiB
        host, requests of 33, 40 and 63 GiB each consumed 65 GiB of free
        memory, and anything above 64 GiB tried to reserve 128 GiB and was
        refused outright -- with 113 GiB free. The largest slab such a host
        could hold was the largest power of two beneath its RAM, and the error
        it raised named device memory rather than the rounding.

        mmap hands back exactly the pages asked for, so a slab costs what it
        says. The driver pins at its own granularity either way, so throughput
        is unchanged for the large allocations this path exists to serve."""
        ptr = pin_region(size_bytes)
        self.pinned_bytes += size_bytes
        self.pinned_peak = max(self.pinned_peak, self.pinned_bytes)
        return int(ptr), ("hostregister", size_bytes)

    def free(self, buffer: Buffer) -> None:
        if self._live.pop(buffer.id, None) is None:
            return                      # already freed (drain idempotence)
        # evict any torch views over this memory and mark it freed BEFORE the
        # unmap, so no cached/fresh view can read the released range
        from ..interop import invalidate_views

        invalidate_views(buffer.ptr, buffer.size_bytes)
        if buffer.location == "fast":
            _check(cudart.cudaFree(buffer.ptr))
        elif isinstance(buffer.raw, tuple) and buffer.raw[0] == "hostregister":
            unpin_region(buffer.ptr, buffer.raw[1])
            self.pinned_bytes -= buffer.raw[1]
        else:
            _check(cudart.cudaFreeHost(buffer.ptr))

    def drain(self) -> int:
        """Free every allocation still outstanding on this backend;
        returns the byte total released. Safe only when no work is in
        flight (synchronize first)."""
        released = 0
        for buf in list(self._live.values()):
            released += buf.size_bytes
            self.free(buf)
        return released

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
    def drain_aborted(self) -> int:
        """Abort-path cleanup (engine service session reuse): wait for
        ALL enqueued device work, then discard every pending completion
        token. A cancelled/failed run otherwise leaves its _TaskDone /
        transfer completions queued, and the NEXT run on the same
        Session pops them ("completion for a job that is not in
        flight"). Returns the number of tokens discarded."""
        _check(cudart.cudaDeviceSynchronize())
        n = 0
        for dq in self._pending.values():
            n += len(dq)
            dq.clear()
        if self.completion_mode == "hostfn":
            with self._hostfn_lock:
                n += len(self._hostfn_tokens)
                self._hostfn_tokens.clear()
                self._hostfn_outstanding = 0
        return n

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
                now = time.perf_counter()
                if now - self.last_poll_yield_s >= self.poll_yield_interval_s:
                    self.last_poll_yield_s = now
                    time.sleep(0)

    def measure_pcie(self, nbytes: int = 512 * 1024 * 1024) -> "PcieBandwidth":
        """Measure pinned-copy bandwidth in both regimes (bytes/us ints).

        On this class of platform the two directions are NOT independent:
        concurrent h2d+d2h can collapse to ~half each (shared host-memory /
        link budget), and h2d may be slower than d2h even alone. Plans
        use the BIDI numbers by doctrine — conservative pricing makes
        predictions floors rather than promises. Chain-ordered plans
        alternate offload-heavy and reload-heavy phases, so lanes often
        achieve their uni rates and reality beats the bidi-priced
        prediction (traced: up to ~20% at the tightest budgets).
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
