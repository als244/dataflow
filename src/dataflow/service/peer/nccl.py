"""ctypes binding for libnccl — the group backend's production lane.

Deliberately tiny: the dozen calls the NcclComm needs, with argtypes
declared for EVERY function (64-bit pointer truncation without them
segfaults — the ctypes lesson from the rdma preflight). The library
comes from the nvidia-nccl wheel both boxes install with torch, so
versions match by construction.

Send/Recv are bound now even though no v1 consumer exists: composed
grouped Send/Recv is how NCCL expresses all-to-all, which the future
expert-parallel routing track needs.
"""
from __future__ import annotations

import ctypes
import os

NCCL_UNIQUE_ID_BYTES = 128

# ncclDataType_t
DTYPE_BY_NAME = {
    "int8": 0, "uint8": 1, "int32": 2, "uint32": 3,
    "int64": 4, "uint64": 5, "fp16": 6, "f16": 6,
    "fp32": 7, "f32": 7, "fp64": 8, "f64": 8, "bf16": 9,
}
NCCL_SUM = 0


class NcclUniqueId(ctypes.Structure):
    # c_uint8, NOT c_char: char arrays read back as NUL-terminated
    # strings and silently truncate the 128-byte id
    _fields_ = [("internal", ctypes.c_uint8 * NCCL_UNIQUE_ID_BYTES)]


class NcclError(RuntimeError):
    pass


def find_libnccl() -> str:
    """The nvidia-nccl wheel's library, falling back to torch/lib and
    the system loader."""
    try:
        import nvidia.nccl

        cand = os.path.join(nvidia.nccl.__path__[0], "lib", "libnccl.so.2")
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    try:
        import torch

        cand = os.path.join(os.path.dirname(torch.__file__), "lib",
                            "libnccl.so.2")
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    return "libnccl.so.2"


class Lib:
    """Loaded libnccl with argtypes on every bound call."""

    def __init__(self, path: str | None = None):
        self.cdll = ctypes.CDLL(path or find_libnccl())
        c = self.cdll
        p = ctypes.c_void_p
        sz = ctypes.c_size_t
        i = ctypes.c_int
        c.ncclGetErrorString.restype = ctypes.c_char_p
        c.ncclGetErrorString.argtypes = [i]
        c.ncclGetVersion.argtypes = [ctypes.POINTER(i)]
        c.ncclGetUniqueId.argtypes = [ctypes.POINTER(NcclUniqueId)]
        c.ncclCommInitRank.argtypes = [ctypes.POINTER(p), i,
                                       NcclUniqueId, i]
        c.ncclCommDestroy.argtypes = [p]
        c.ncclCommAbort.argtypes = [p]
        c.ncclCommGetAsyncError.argtypes = [p, ctypes.POINTER(i)]
        c.ncclAllReduce.argtypes = [p, p, sz, i, i, p, p]
        c.ncclBroadcast.argtypes = [p, p, sz, i, i, p, p]
        c.ncclReduce.argtypes = [p, p, sz, i, i, i, p, p]
        c.ncclReduceScatter.argtypes = [p, p, sz, i, i, p, p]
        c.ncclAllGather.argtypes = [p, p, sz, i, p, p]
        c.ncclSend.argtypes = [p, sz, i, i, p, p]
        c.ncclRecv.argtypes = [p, sz, i, i, p, p]
        c.ncclGroupStart.argtypes = []
        c.ncclGroupEnd.argtypes = []

    def check(self, rc: int, what: str) -> None:
        if rc != 0:
            msg = self.cdll.ncclGetErrorString(rc).decode()
            raise NcclError(f"{what}: {msg} (nccl rc={rc})")

    def version(self) -> int:
        v = ctypes.c_int()
        self.check(self.cdll.ncclGetVersion(ctypes.byref(v)),
                   "ncclGetVersion")
        return v.value

    def unique_id(self) -> bytes:
        uid = NcclUniqueId()
        self.check(self.cdll.ncclGetUniqueId(ctypes.byref(uid)),
                   "ncclGetUniqueId")
        return bytes(uid.internal)

    def comm_init_rank(self, world: int, uid: bytes, rank: int) -> int:
        raw = NcclUniqueId()
        ctypes.memmove(raw.internal, uid, NCCL_UNIQUE_ID_BYTES)
        comm = ctypes.c_void_p()
        self.check(self.cdll.ncclCommInitRank(ctypes.byref(comm), world,
                                              raw, rank),
                   "ncclCommInitRank")
        return comm.value

    def comm_destroy(self, comm: int) -> None:
        self.cdll.ncclCommDestroy(ctypes.c_void_p(comm))

    def comm_abort(self, comm: int) -> None:
        self.cdll.ncclCommAbort(ctypes.c_void_p(comm))

    def async_error(self, comm: int) -> int:
        e = ctypes.c_int()
        self.check(self.cdll.ncclCommGetAsyncError(
            ctypes.c_void_p(comm), ctypes.byref(e)),
            "ncclCommGetAsyncError")
        return e.value

    def all_reduce(self, send_ptr: int, recv_ptr: int, count: int,
                   dtype: int, comm: int, stream: int) -> None:
        self.check(self.cdll.ncclAllReduce(
            send_ptr, recv_ptr, count, dtype, NCCL_SUM,
            ctypes.c_void_p(comm), ctypes.c_void_p(stream)),
            "ncclAllReduce")

    def broadcast(self, send_ptr: int, recv_ptr: int, count: int,
                  dtype: int, root: int, comm: int, stream: int) -> None:
        self.check(self.cdll.ncclBroadcast(
            send_ptr, recv_ptr, count, dtype, root,
            ctypes.c_void_p(comm), ctypes.c_void_p(stream)),
            "ncclBroadcast")

    def reduce(self, send_ptr: int, recv_ptr: int, count: int,
               dtype: int, root: int, comm: int, stream: int) -> None:
        self.check(self.cdll.ncclReduce(
            send_ptr, recv_ptr, count, dtype, NCCL_SUM, root,
            ctypes.c_void_p(comm), ctypes.c_void_p(stream)),
            "ncclReduce")

    def reduce_scatter(self, send_ptr: int, recv_ptr: int,
                       recv_count: int, dtype: int, comm: int,
                       stream: int) -> None:
        self.check(self.cdll.ncclReduceScatter(
            send_ptr, recv_ptr, recv_count, dtype, NCCL_SUM,
            ctypes.c_void_p(comm), ctypes.c_void_p(stream)),
            "ncclReduceScatter")

    def all_gather(self, send_ptr: int, recv_ptr: int, send_count: int,
                   dtype: int, comm: int, stream: int) -> None:
        self.check(self.cdll.ncclAllGather(
            send_ptr, recv_ptr, send_count, dtype,
            ctypes.c_void_p(comm), ctypes.c_void_p(stream)),
            "ncclAllGather")

    def send(self, ptr: int, count: int, dtype: int, peer: int,
             comm: int, stream: int) -> None:
        self.check(self.cdll.ncclSend(
            ptr, count, dtype, peer, ctypes.c_void_p(comm),
            ctypes.c_void_p(stream)), "ncclSend")

    def recv(self, ptr: int, count: int, dtype: int, peer: int,
             comm: int, stream: int) -> None:
        self.check(self.cdll.ncclRecv(
            ptr, count, dtype, peer, ctypes.c_void_p(comm),
            ctypes.c_void_p(stream)), "ncclRecv")

    def group_start(self) -> None:
        self.check(self.cdll.ncclGroupStart(), "ncclGroupStart")

    def group_end(self) -> None:
        self.check(self.cdll.ncclGroupEnd(), "ncclGroupEnd")


LIB: Lib | None = None
LIB_ERROR: str | None = None


def get_lib() -> Lib:
    """Load-once accessor; raises NcclError with the original cause if
    the library is unavailable."""
    global LIB, LIB_ERROR
    if LIB is not None:
        return LIB
    if LIB_ERROR is not None:
        raise NcclError(LIB_ERROR)
    try:
        LIB = Lib()
        return LIB
    except OSError as ex:
        LIB_ERROR = f"libnccl unavailable: {ex}"
        raise NcclError(LIB_ERROR)


def available() -> bool:
    try:
        get_lib()
        return True
    except NcclError:
        return False
