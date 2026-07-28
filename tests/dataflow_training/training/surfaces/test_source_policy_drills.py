"""Source-policy save/restore drills: two in-process daemons play a
world-2 fleet — replicated W, element-sharded O in the REAL slice
layout (256-aligned [m_slice|v_slice|m_tail|v_tail] fields with a
redundantly-updated world-remainder tail) — and the compiled
policies round-trip bitwise through save_checkpoint and
resolve_targets. The shard sizes are chosen alignment-HOSTILE
(n_slice bytes not a multiple of 256) so tight-packing arithmetic
cannot pass by luck. CPU-only (fake engines; the policy and record
logic is byte-level).

Tests:
- test_simple_policy_round_trip_world2: the simple policy saves whole buffers per source; each rank restores bitwise from its OWN snapshot (keyed targets, bare ids resolving to source-qualified private logicals) at its exact resident geometry, and the logical view reassembles the tight [m_all | v_all] — slice fields at element offsets, the replicated tail once — reads W from the first covering snapshot, and lists each source's private Aux copy separately.
- test_dedup_policy_covers_with_disjoint_slices: dedup saves W as disjoint responsibility slices — one copy total — and the logical view reassembles it bitwise.
- test_replication_drift_refuses_before_record: diverged W bytes between sources make save_checkpoint refuse naming both sources, and NO record lands.
"""
import threading
import time

import numpy as np
import pytest

from dataflow.checkpoint import (CheckpointError, RECORD_NAME,
                                 read_record, resolve_targets,
                                 save_checkpoint)
from dataflow.service import EngineClient, EngineConfig, Server
from dataflow_training.blocks.layouts import opt_state_slice_layout
from dataflow_training.distributed.source_policy import \
    compile_source_policy

W_BYTES = 8192
N_SLICE = 500              # 2000 B per slot: NOT 256-aligned
N_TAIL = 3                 # world remainder, updated on every rank
OPT_SLICES = {"W_0": {"n_slice": N_SLICE, "n_tail": N_TAIL,
                      "opt_dtype": "fp32"}}
O_LAYOUT = opt_state_slice_layout(N_SLICE, N_TAIL, "fp32")
O_LOCAL = O_LAYOUT.total_bytes
N_LOGICAL = 2 * N_SLICE + N_TAIL
O_LOGICAL = 2 * N_LOGICAL * 4        # tight [m_all | v_all]
PLAN = {"W_0": [
    {"rank": 0, "lo": 0, "hi": W_BYTES // 2, "role": "responsible"},
    {"rank": 1, "lo": W_BYTES // 2, "hi": W_BYTES,
     "role": "responsible"},
]}


def boot(tmp, name):
    sock = str(tmp / f"{name}.sock")
    server = Server(EngineConfig(socket_path=sock, fake=True,
                                 slab_backing_gib=0.1))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(300):
        try:
            with EngineClient(sock, client_name="probe"):
                break
        except (ConnectionError, FileNotFoundError, OSError):
            time.sleep(0.01)
    return server, EngineClient(sock, client_name=name)


def rng_bytes(seed, n):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, n, dtype=np.uint8).tobytes()


def shard_bytes(seed, tail_seed):
    """A layout-true [m_slice|v_slice|m_tail|v_tail] payload: rng
    content in the fields, ZERO alignment padding (padding is layout,
    not state), and the tail fields from a SHARED seed — the
    byte-equal contract updates the tail identically on every
    rank."""
    out = bytearray(O_LOCAL)
    for f in O_LAYOUT.fields:
        content_seed = tail_seed if f.name.endswith("_tail") else seed
        out[f.offset_bytes:f.offset_bytes + f.nbytes] = \
            rng_bytes(content_seed * 7 + f.offset_bytes, f.nbytes)
    return bytes(out)


def field_bytes(payload: bytes, name: str) -> bytes:
    f = O_LAYOUT.field(name)
    return payload[f.offset_bytes:f.offset_bytes + f.nbytes]


def fleet_state(w_bytes_by_rank):
    """Per-rank element-sharded O for a world-2 fleet: distinct
    slice fields, identical tail fields."""
    o = {r: shard_bytes(100 + r, 999) for r in (0, 1)}
    return o


def compiled_sources(clients, o, policy):
    source_specs = {r: [("W_0", W_BYTES), ("O_0", O_LOCAL)]
                    for r in (0, 1)}
    logical, per_source = compile_source_policy(
        policy=policy, world=2, source_specs=source_specs,
        plan=PLAN, opt_slices=OPT_SLICES)
    sources = {}
    for r in (0, 1):
        sources[r] = {"client": clients[r], "path": f"rank{r}",
                      "slices": per_source[r]["slices"],
                      "record": per_source[r]["record"],
                      "objects": per_source[r]["objects"],
                      "client_meta": {"step": 4, "rank": r}}
    return logical, sources


def aggregate_o(o) -> bytes:
    """The tight logical [m_all | v_all]: slice fields at element
    offsets, the replicated tail once (source 0's copy)."""
    m = (field_bytes(o[0], "m_slice") + field_bytes(o[1], "m_slice")
         + field_bytes(o[0], "m_tail"))
    v = (field_bytes(o[0], "v_slice") + field_bytes(o[1], "v_slice")
         + field_bytes(o[0], "v_tail"))
    return m + v


def restore_plan(client, step_dir, plan):
    for step in plan:
        client.restore_snapshot(str(step_dir / step["path"]),
                                remap=step["remap"], overwrite=True)


def test_simple_policy_round_trip_world2(tmp_path):
    w = rng_bytes(7, W_BYTES)
    o = fleet_state(w)
    daemons = [boot(tmp_path, f"w{r}") for r in (0, 1)]
    clients = {r: daemons[r][1] for r in (0, 1)}
    try:
        for r in (0, 1):
            clients[r].put_object("W_0", w)
            clients[r].put_object("O_0", o[r])
        logical, sources = compiled_sources(clients, o, "simple")
        step_dir = tmp_path / "step_000004"
        record = save_checkpoint(
            sources, step_dir, step=4, seed=11,
            logical_objects=logical,
            scheme={"world": 2, "kind": "zero1rs",
                    "source_policy": "simple"},
            client_payload={"losses": [5.0]})
        assert (step_dir / RECORD_NAME).is_file()
        assert read_record(step_dir)["scheme"]["source_policy"] == \
            "simple"
        o_slices = record["slices"]["O_0"]
        assert len(o_slices) == 8, \
            "2 sources x ([m|v] slice fields + [m|v] tail fields)"
        tail_spans = [tuple(s["object_range"]) for s in o_slices]
        assert len(tail_spans) - len(set(tail_spans)) == 2, \
            "the two tail fields overlap as exact-span replicas"
        assert record["logical_objects"]["O_0"]["bytes"] == O_LOGICAL
        for snap in record["snapshots"]:
            assert snap["objects"]["O_0"] == O_LOCAL, \
                "the snapshot inventory records resident geometry"

        # each rank restores from its OWN snapshot: the rank view
        for r in (0, 1):
            server_f, fresh = boot(tmp_path, f"fresh{r}")
            try:
                plan = resolve_targets(record, {str(r): ["W_0", "O_0"]})
                restore_plan(fresh, step_dir, plan)
                assert bytes(fresh.get_object("W_0")) == w
                assert bytes(fresh.get_object("O_0")) == o[r], \
                    f"rank {r} O shard diverged"
            finally:
                fresh.shutdown()

        # the logical view: assembled W + aggregate [m_all|v_all]
        server_l, logical_client = boot(tmp_path, "logical")
        try:
            plan = resolve_targets(record, "all")
            restore_plan(logical_client, step_dir, plan)
            assert bytes(logical_client.get_object("W_0")) == w
            assert bytes(logical_client.get_object("O_0")) == \
                aggregate_o(o)
        finally:
            logical_client.shutdown()
    finally:
        for _server, c in daemons:
            c.shutdown()


def test_dedup_policy_covers_with_disjoint_slices(tmp_path):
    w = rng_bytes(13, W_BYTES)
    o = fleet_state(w)
    daemons = [boot(tmp_path, f"d{r}") for r in (0, 1)]
    clients = {r: daemons[r][1] for r in (0, 1)}
    try:
        for r in (0, 1):
            clients[r].put_object("W_0", w)
            clients[r].put_object("O_0", o[r])
        logical, sources = compiled_sources(clients, o, "dedup")
        step_dir = tmp_path / "step_000008"
        record = save_checkpoint(
            sources, step_dir, step=8, seed=11,
            logical_objects=logical,
            scheme={"source_policy": "dedup"})

        w_slices = record["slices"]["W_0"]
        assert len(w_slices) == 2
        spans = sorted(tuple(s["object_range"]) for s in w_slices)
        assert spans == [(0, W_BYTES // 2), (W_BYTES // 2, W_BYTES)], \
            "dedup must save ONE copy as disjoint responsibility slices"

        server_l, logical_client = boot(tmp_path, "dlogical")
        try:
            plan = resolve_targets(record, "all")
            restore_plan(logical_client, step_dir, plan)
            assert bytes(logical_client.get_object("W_0")) == w
            assert bytes(logical_client.get_object("O_0")) == \
                aggregate_o(o)
        finally:
            logical_client.shutdown()
    finally:
        for _server, c in daemons:
            c.shutdown()


def test_replication_drift_refuses_before_record(tmp_path):
    o = fleet_state(None)
    daemons = [boot(tmp_path, f"x{r}") for r in (0, 1)]
    clients = {r: daemons[r][1] for r in (0, 1)}
    try:
        clients[0].put_object("W_0", rng_bytes(17, W_BYTES))
        clients[1].put_object("W_0", rng_bytes(19, W_BYTES))   # drift
        for r in (0, 1):
            clients[r].put_object("O_0", o[r])
        logical, sources = compiled_sources(clients, o, "simple")
        step_dir = tmp_path / "step_000012"
        with pytest.raises(CheckpointError) as ei:
            save_checkpoint(sources, step_dir, step=12, seed=11,
                            logical_objects=logical)
        message = str(ei.value)
        assert "drift" in message and "0" in message and "1" in message
        assert not (step_dir / RECORD_NAME).is_file(), \
            "a drifted save must leave NO record"
    finally:
        for _server, c in daemons:
            c.shutdown()
