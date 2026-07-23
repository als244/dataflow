"""Zero-copy torch interop over runtime-owned buffers.

- `torch_view(buffer_or_ptr, shape, dtype)` wraps a raw device pointer as a
  torch tensor via a ctypes-built DLPack capsule: no torch allocator
  involvement, no ownership transfer (the runtime owns the bytes; keep the
  view's lifetime inside the task launch).
- `external_stream(stream)` returns a `torch.cuda.ExternalStream` for a
  runtime Stream so torch/Triton kernels enqueue on runtime-owned streams.

This module is the ONLY place torch touches raw runtime pointers.
"""
from __future__ import annotations

import ctypes
from typing import Union

import torch

from dataflow.runtime.device.base import Buffer, Stream

# torch dtype -> (DLPack type code, bits). Codes: int=0, uint=1, float=2, bfloat=4.
_DLPACK_DTYPE: dict[torch.dtype, tuple[int, int]] = {
    torch.float64: (2, 64),
    torch.float32: (2, 32),
    torch.float16: (2, 16),
    torch.bfloat16: (4, 16),
    torch.int64: (0, 64),
    torch.int32: (0, 32),
    torch.int16: (0, 16),
    torch.int8: (0, 8),
    torch.uint8: (1, 8),
}

TORCH_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
}


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_KDL_CUDA = 2
# Pinned host memory is presented as a plain CPU tensor (torch rejects
# kDLCUDAHost=3 on import). Host-side fill/readback is all these views do;
# device access to pinned bytes goes through cudaMemcpyAsync on raw pointers.
_KDL_CPU = 1

_capsule_new = ctypes.pythonapi.PyCapsule_New
_capsule_new.restype = ctypes.py_object
_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]


# --- view cache -----------------------------------------------------------------
# Building a DLPack view costs ~2.2 us; a block task fans out to ~30 of them
# (~50-70 us of the exposed time-to-first-kernel). Under static placement and
# pool reuse, buffers recur at IDENTICAL addresses every step, so views are
# cached by (address, offset, dtype, shape, device-kind). The cached tensor
# does not own memory; the pool retains allocations for the session, and
# long-lived callers clear it at teardown (clear_view_cache).
_VIEW_CACHE: dict = {}
_VIEW_CACHE_MAX = 8192


def clear_view_cache() -> None:
    _VIEW_CACHE.clear()
    _STREAM_CACHE.clear()


def invalidate_views(ptr: int, size_bytes: int) -> None:
    """Evict every cached view over [ptr, ptr+size). Call when the backing
    memory is really freed/unmapped (cudaFree/cudaFreeHost) so no later lookup
    returns a view of memory that no longer belongs to the buffer — the exact
    cross-session hazard when a re-registered program hashes to the same prog_id
    and the pool hands its address to a different buffer. Evicting is always
    safe: a miss simply rebuilds a fresh view over whatever the address now
    holds, so this cannot false-positive across allocators.

    A view a caller still HOLDS across a real free is the ownership rule's
    domain, not this cache's: the run boundary stops it escaping into an error,
    the client-only workload-test rule removes the workload cases, and pool
    reclaim keeps memory mapped so reclaimed-buffer views stay valid."""
    lo, hi = int(ptr), int(ptr) + int(size_bytes)
    dead = [key for key in _VIEW_CACHE if lo <= key[0] < hi]
    for key in dead:
        _VIEW_CACHE.pop(key, None)


def torch_view(
    src: Union[Buffer, int],
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    device_id: int = 0,
    offset_bytes: int = 0,
    pinned_host: bool = False,
) -> torch.Tensor:
    """Dense zero-copy view of runtime-owned memory as a torch tensor.

    The returned tensor does NOT own the memory (no deleter); it must not
    outlive the runtime buffer. Size is validated against the buffer when a
    Buffer is passed.
    """
    if isinstance(src, Buffer):
        ptr = src.ptr + offset_bytes
        want = _numel(shape) * _element_bytes(dtype)
        if offset_bytes + want > src.size_bytes:
            raise ValueError(
                f"view of {want} bytes at offset {offset_bytes} exceeds buffer "
                f"{src.id} ({src.size_bytes} bytes)"
            )
        pinned_host = pinned_host or src.location == "backing"
    else:
        ptr = int(src) + offset_bytes

    cache_key = (ptr, dtype, shape, pinned_host, device_id)
    cached = _VIEW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    code, bits = _DLPACK_DTYPE[dtype]
    ndim = len(shape)
    shape_arr = (ctypes.c_int64 * max(ndim, 1))(*shape)
    holder = _DLManagedTensor()
    holder.dl_tensor = _DLTensor(
        ctypes.c_void_p(ptr),
        _DLDevice(_KDL_CPU, 0) if pinned_host else _DLDevice(_KDL_CUDA, device_id),
        ndim,
        _DLDataType(code, bits, 1),
        shape_arr,
        None,  # dense row-major
        0,
    )
    holder.manager_ctx = None
    holder.deleter = None
    capsule = _capsule_new(ctypes.byref(holder), b"dltensor", None)
    tensor = torch.from_dlpack(capsule)
    # keep the ctypes structures alive as long as the tensor view exists
    tensor._dataflow_dlpack_holder = (holder, shape_arr)  # type: ignore[attr-defined]
    if len(_VIEW_CACHE) >= _VIEW_CACHE_MAX:
        _VIEW_CACHE.clear()
    _VIEW_CACHE[cache_key] = tensor
    return tensor


_STREAM_CACHE: dict = {}


def external_stream(stream: Stream, *, device_id: int = 0) -> torch.cuda.ExternalStream:
    key = (int(stream.raw), device_id)
    es = _STREAM_CACHE.get(key)
    if es is None:
        es = _STREAM_CACHE[key] = torch.cuda.ExternalStream(int(stream.raw), device=device_id)
    return es


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _element_bytes(dtype: torch.dtype) -> int:
    return _DLPACK_DTYPE[dtype][1] // 8
