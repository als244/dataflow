"""Calibrated spin-kernel executables for real-GPU synthetic runs.

A clock64 busy-loop kernel occupies the compute stream for a target duration,
so an annotated program's *planned* runtimes become physically true on the
device — isolating the scheduler/transfer behavior from task math. Calibration
measures the SM clock via event timing at startup.
"""
from __future__ import annotations

import ctypes
import statistics
from dataclasses import dataclass

from cuda.bindings import driver, nvrtc
from cuda.bindings import runtime as cudart

from dataflow.core import TaskSpec
from .base import Stream
from .cuda import CudaBackend, CudaError, _check

# Spin on %globaltimer (nanosecond wall clock) rather than clock64 cycles:
# SM clocks ramp with load (idle ~0.4GHz -> boost ~2.9GHz), so cycle-counted
# spins launched after idle gaps run several times longer than calibrated.
# The global timer is clock-invariant, making durations wall-true with no
# calibration dependence.
_SPIN_SRC = rb"""
__device__ __forceinline__ unsigned long long __gt_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
extern "C" __global__ void spin_kernel(long long ns) {
    unsigned long long start = __gt_ns();
    while (__gt_ns() - start < (unsigned long long)ns) { }
}
"""


def _check_drv(result: tuple) -> tuple:
    err = result[0]
    if err != driver.CUresult.CUDA_SUCCESS:
        raise CudaError(f"CUDA driver call failed: {err}")
    return result[1:]


def _check_nvrtc(result: tuple) -> tuple:
    err = result[0]
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise CudaError(f"NVRTC call failed: {err}")
    return result[1:]


class SpinKernel:
    """Compiled clock64 spin kernel + SM-clock calibration."""

    def __init__(self, device: int = 0):
        (props,) = _check(cudart.cudaGetDeviceProperties(device))
        arch = f"--gpu-architecture=compute_{props.major}{props.minor}".encode()
        (prog,) = _check_nvrtc(nvrtc.nvrtcCreateProgram(_SPIN_SRC, b"spin.cu", 0, [], []))
        res = nvrtc.nvrtcCompileProgram(prog, 1, [arch])
        if res[0] != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            (log_size,) = _check_nvrtc(nvrtc.nvrtcGetProgramLogSize(prog))
            log = b" " * log_size
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise CudaError(f"spin kernel compile failed:\n{log.decode()}")
        (ptx_size,) = _check_nvrtc(nvrtc.nvrtcGetPTXSize(prog))
        ptx = b" " * ptx_size
        _check_nvrtc(nvrtc.nvrtcGetPTX(prog, ptx))
        _check_drv(driver.cuInit(0))
        (self._module,) = _check_drv(driver.cuModuleLoadData(ptx))
        (self._fn,) = _check_drv(driver.cuModuleGetFunction(self._module, b"spin_kernel"))

    def launch_us(self, stream: Stream, duration_us: float) -> None:
        arg = ctypes.c_longlong(max(int(duration_us * 1e3), 1))  # ns
        arg_ptrs = (ctypes.c_void_p * 1)(ctypes.addressof(arg))
        _check_drv(driver.cuLaunchKernel(
            self._fn, 1, 1, 1, 32, 1, 1, 0, stream.raw, arg_ptrs, 0,
        ))

    def verify(self, backend: CudaBackend, *, target_us: float = 2_000.0, repeats: int = 3) -> float:
        """Median measured/target ratio (should be ~1.0; wall-true spin)."""
        stream = backend.create_stream("compute")
        ratios: list[float] = []
        for i in range(repeats + 1):
            start = backend.record_event(stream)
            self.launch_us(stream, target_us)
            end = backend.record_event(stream)
            _check(cudart.cudaEventSynchronize(end.raw))
            (ms,) = _check(cudart.cudaEventElapsedTime(start.raw, end.raw))
            if i > 0:  # first iteration is warmup
                ratios.append((ms * 1e3) / target_us)
        return statistics.median(ratios)


@dataclass(frozen=True)
class SpinExecutable:
    """Occupies ctx.stream for `runtime_us` of wall-true device time."""

    runtime_us: float
    kernel: SpinKernel

    def launch(self, ctx) -> None:
        self.kernel.launch_us(ctx.stream, self.runtime_us)


def make_spin_resolver(backend: CudaBackend):
    """Every task becomes a wall-true spin of its planned runtime."""
    kernel = SpinKernel(device=backend.device)
    accuracy = kernel.verify(backend)

    def resolver(task: TaskSpec) -> SpinExecutable:
        return SpinExecutable(runtime_us=task.runtime_us, kernel=kernel)

    resolver.spin_accuracy_ratio = accuracy  # type: ignore[attr-defined]
    return resolver
