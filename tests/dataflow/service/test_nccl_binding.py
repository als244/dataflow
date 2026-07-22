"""The ctypes libnccl binding: load, version, world-1 comm init/destroy,
and a world-1 allreduce identity.

Tests:
- test_binding_world1_roundtrip: a world-size-1 NCCL comm built from a unique id all-reduces a bf16 tensor as an identity (values unchanged), reports no async error, and destroys cleanly.
- test_dtype_map_covers_grad_dtypes: the NCCL dtype-name map includes bf16, f16, and f32.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from dataflow.service.peer import nccl

if not nccl.available():
    pytest.skip("libnccl unavailable", allow_module_level=True)

nccl_version = nccl.get_lib().version()
if nccl_version < 22000:
    pytest.skip(f"libnccl {nccl_version} < 22000 (binding needs >= 2.20)",
                allow_module_level=True)

pytestmark = [pytest.mark.gpu, pytest.mark.ncclbind]


def test_binding_world1_roundtrip():
    lib = nccl.get_lib()
    uid = lib.unique_id()
    assert len(uid) == nccl.NCCL_UNIQUE_ID_BYTES
    comm = lib.comm_init_rank(1, uid, 0)
    assert comm
    try:
        t = torch.full((1 << 16,), 3.0, device="cuda",
                       dtype=torch.bfloat16)
        stream = torch.cuda.current_stream()
        lib.all_reduce(t.data_ptr(), t.data_ptr(), t.numel(),
                       nccl.DTYPE_BY_NAME["bf16"], comm,
                       stream.cuda_stream)
        stream.synchronize()
        assert float(t[0]) == 3.0 and float(t[-1]) == 3.0
        assert lib.async_error(comm) == 0
    finally:
        lib.comm_destroy(comm)


def test_dtype_map_covers_grad_dtypes():
    for name in ("bf16", "f16", "f32"):
        assert name in nccl.DTYPE_BY_NAME
