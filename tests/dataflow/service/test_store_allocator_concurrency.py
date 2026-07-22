"""The locked-allocator audit gate (peer plane): the extent allocator
is a mutex-guarded structure with TWO writers — the dispatcher (put/
release/transients) and the NetworkManager (inbound reservations).
This battery hammers both paths from concurrent threads against a fake
slab and asserts the allocator's invariants hold: no two live extents
ever overlap, bookkeeping balances, and everything coalesces back to
one free extent at quiesce. Allocation is SIDE-EFFECT-FREE by design
(no catalog/ledger mutation inside an alloc/free) — reserve_inbound
does its collision/lease checks and its alloc ATOMICALLY under
catalog_lock, because put()'s unlocked pre-checks are a
single-writer luxury the NM does not get.

Tests:
- test_two_writer_allocator_invariants: concurrent inbound reservations and dispatcher put/release keep live extents non-overlapping, keep bookkeeping balanced against the free list, and coalesce to one full free extent at quiesce.
"""
import threading

from dataflow.service.store import Store


def worker_inbound(store, idx, cycles, errors):
    for i in range(cycles):
        ext, code = store.reserve_inbound(f"peer_obj_{idx}_{i}",
                                          (idx + 1) * 4096)
        if ext is None:
            if code != "CAPACITY":
                errors.append(f"unexpected refusal {code}")
            continue
        view = store.view_extent(ext, (idx + 1) * 4096)
        view[:8] = bytes([idx] * 8)            # touch the landing
        if i % 3 == 0:
            store.release_inbound(ext)
        else:
            store.adopt_inbound(f"peer_obj_{idx}_{i}", ext,
                                (idx + 1) * 4096, from_peer="t")


def worker_dispatcher(store, idx, cycles, errors):
    for i in range(cycles):
        oid = f"disp_obj_{idx}_{i}"
        try:
            store.put(oid, bytes(2048))
            if i % 2 == 0:
                store.release(oid)
        except Exception as e:                 # capacity under pressure: ok
            if "CAPACITY" not in str(e):
                errors.append(repr(e))


def test_two_writer_allocator_invariants():
    store = Store(capacity_bytes=8 << 20)      # fake slab (bytearray)
    errors: list = []
    threads = []
    for idx in range(4):
        threads.append(threading.Thread(
            target=worker_inbound, args=(store, idx, 200, errors)))
        threads.append(threading.Thread(
            target=worker_dispatcher, args=(store, idx, 200, errors)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors[:5]
    # invariant: live catalog extents never overlap
    spans = sorted((r.extent.offset, r.extent.offset + r.extent.size)
                   for r in store.objects.values())
    for (alo, ahi), (blo, bhi) in zip(spans, spans[1:]):
        assert ahi <= blo, "overlapping live extents"
    # bookkeeping balances against the free list
    free_total = sum(e.size for e in store.allocator.free)
    assert free_total + store.allocator.used_bytes \
        == store.allocator.capacity
    # quiesce: releasing everything coalesces back to ONE free extent
    for oid in list(store.objects):
        store.release(oid)
    assert len(store.allocator.free) == 1
    assert store.allocator.free[0].size == store.allocator.capacity
    assert store.allocator.used_bytes == 0
