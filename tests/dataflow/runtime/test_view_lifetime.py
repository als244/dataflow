"""View-lifetime hardening.

A torch view over runtime memory must never outlive the buffer. When memory is
really freed/unmapped, the backend evicts the view cache over that range, so no
later lookup can hand back a view of memory that no longer belongs to the
buffer — the cross-session hazard where a re-registered program hashes to the
same prog_id and the pool hands its address to a different buffer. Reclaiming a
buffer to the pool keeps its memory mapped and leaves its views valid (the whole
point of the cache).

Eviction is deliberately the ONLY mechanism: it is always safe (a cache miss
rebuilds a fresh view over whatever the address now holds), so it never
false-positives across the several allocators that hand out device memory. A
view a caller still holds across a real free is the ownership rule's domain (the
run boundary stops it escaping into an error; the client-only workload-test rule
removes the workload cases).

Tests:
- test_invalidate_evicts_cached_views: invalidate_views drops every cached view
  over the freed range and keeps views outside it.
- test_free_evicts_cache_no_stale_view: end to end on the real backend — a view
  is cached, the cache entry is evicted on free, and a fresh allocation at the
  address yields a freshly-built view (never the stale one).
"""
import pytest

torch = pytest.importorskip("torch")

from dataflow.runtime import interop


@pytest.fixture(autouse=True)
def clean_view_cache():
    interop.clear_view_cache()
    yield
    interop.clear_view_cache()


def test_invalidate_evicts_cached_views():
    interop._VIEW_CACHE[(0x1000, torch.float32, (4,), False, 0)] = "v1"
    interop._VIEW_CACHE[(0x1040, torch.float32, (4,), False, 0)] = "v2"
    interop._VIEW_CACHE[(0x9000, torch.float32, (4,), False, 0)] = "other"
    interop.invalidate_views(0x1000, 0x100)          # covers 0x1000..0x1100
    addresses = {key[0] for key in interop._VIEW_CACHE}
    assert 0x1000 not in addresses
    assert 0x1040 not in addresses
    assert 0x9000 in addresses                       # outside the range, kept


@pytest.mark.gpu
def test_free_evicts_cache_no_stale_view():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from dataflow.runtime.device.cuda import CudaBackend

    backend = CudaBackend()
    buf = backend.alloc("fast", 256)
    key = (buf.ptr, torch.float32, (64,), False, 0)
    interop.torch_view(buf, (64,), torch.float32).fill_(1.5)   # cached, live
    assert key in interop._VIEW_CACHE

    backend.free(buf)                                          # unmaps + evicts
    assert key not in interop._VIEW_CACHE

    # a fresh allocation that reuses the address gets a freshly-built view,
    # never the evicted stale one
    buf2 = backend.alloc("fast", 256)
    view2 = interop.torch_view(buf2, (64,), torch.float32)
    if buf2.ptr == buf.ptr:
        assert interop._VIEW_CACHE[key] is view2              # rebuilt, not stale
    view2.fill_(2.0)
    assert float(view2[0]) == 2.0
    backend.free(buf2)
