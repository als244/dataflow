"""RDMA preflight (peer plane P2): proves the load-bearing assumptions
on THIS box before any transport code exists —

1. RLIMIT_MEMLOCK admits slab-scale registration (the slab is pinned
   TWICE: cudaHostAlloc once, ibv_reg_mr's kernel pinning again);
2. ibv_reg_mr ACCEPTS cudaHostAlloc'd memory (ctypes libibverbs on a
   real pinned buffer) and reports registration throughput.

Usage: python tools/rdma_preflight.py [--device mlx5_0] [--mib 512]
"""
import argparse
import ctypes
import ctypes.util
import resource
import sys
import time


class IbvDeviceAttr(ctypes.Structure):
    _fields_ = [("fw_ver", ctypes.c_char * 64), ("node_guid", ctypes.c_uint64),
                ("sys_image_guid", ctypes.c_uint64),
                ("max_mr_size", ctypes.c_uint64), ("page_size_cap", ctypes.c_uint64),
                ("vendor_id", ctypes.c_uint32), ("vendor_part_id", ctypes.c_uint32),
                ("hw_ver", ctypes.c_uint32), ("max_qp", ctypes.c_int),
                ("max_qp_wr", ctypes.c_int), ("device_cap_flags", ctypes.c_uint),
                ("max_sge", ctypes.c_int)]     # truncated: prefix-compatible


IBV_ACCESS_LOCAL_WRITE = 1
IBV_ACCESS_REMOTE_WRITE = 2
IBV_ACCESS_REMOTE_READ = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--mib", type=int, default=512)
    args = ap.parse_args()

    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    def fmt(v):
        return "unlimited" if v == resource.RLIM_INFINITY else f"{v >> 20} MiB"
    print(f"RLIMIT_MEMLOCK: soft={fmt(soft)} hard={fmt(hard)}")
    need = args.mib << 20
    if soft != resource.RLIM_INFINITY and soft < need:
        print(f"FAIL: memlock soft limit < test size ({args.mib} MiB).")
        print("fix: /etc/security/limits.conf ->")
        print("  * soft memlock unlimited\n  * hard memlock unlimited")
        print("(or systemd: LimitMEMLOCK=infinity), then re-login.")
        return 2

    libname = ctypes.util.find_library("ibverbs")
    if not libname:
        print("FAIL: libibverbs not found")
        return 2
    ibv = ctypes.CDLL(libname, use_errno=True)
    # EVERY call needs argtypes: ctypes' default int conversion
    # truncates 64-bit pointers (the first preflight run segfaulted
    # on exactly this — findings ledger, P2)
    ibv.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
    ibv.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
    ibv.ibv_get_device_name.restype = ctypes.c_char_p
    ibv.ibv_get_device_name.argtypes = [ctypes.c_void_p]
    ibv.ibv_open_device.restype = ctypes.c_void_p
    ibv.ibv_open_device.argtypes = [ctypes.c_void_p]
    ibv.ibv_alloc_pd.restype = ctypes.c_void_p
    ibv.ibv_alloc_pd.argtypes = [ctypes.c_void_p]
    ibv.ibv_reg_mr.restype = ctypes.c_void_p
    ibv.ibv_reg_mr.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_size_t, ctypes.c_int]
    ibv.ibv_dereg_mr.argtypes = [ctypes.c_void_p]

    n = ctypes.c_int(0)
    devs = ibv.ibv_get_device_list(ctypes.byref(n))
    names = [ibv.ibv_get_device_name(devs[i]).decode() for i in range(n.value)]
    print(f"devices: {names}")
    pick = args.device or (names[0] if names else None)
    if pick is None or pick not in names:
        print(f"FAIL: device {args.device!r} not in {names}")
        return 2
    ctx = ibv.ibv_open_device(devs[names.index(pick)])
    pd = ibv.ibv_alloc_pd(ctx)
    if not pd:
        print("FAIL: ibv_alloc_pd")
        return 2

    import torch
    buf = torch.empty(need, dtype=torch.uint8, pin_memory=True)
    ptr = buf.data_ptr()
    t0 = time.perf_counter()
    mr = ibv.ibv_reg_mr(pd, ctypes.c_void_p(ptr),
                        ctypes.c_size_t(need),
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                        | IBV_ACCESS_REMOTE_READ)
    dt = time.perf_counter() - t0
    if not mr:
        err = ctypes.get_errno()
        print(f"FAIL: ibv_reg_mr on cudaHostAlloc memory errno={err}")
        return 2
    rate = need / dt / (1 << 30)
    print(f"OK: registered {args.mib} MiB of cudaHostAlloc memory on "
          f"{pick} in {dt * 1000:.1f} ms ({rate:.1f} GiB/s) — "
          f"lkey/rkey valid, whole-slab MR is viable")
    ibv.ibv_dereg_mr(mr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
