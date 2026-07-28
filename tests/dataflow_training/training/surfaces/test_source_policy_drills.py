"""Source-policy save/restore drills: two in-process daemons play a
world-2 fleet — replicated W, zero1-sharded O ([m|v] slots) — and the
compiled policies round-trip bitwise through save_checkpoint and
resolve_targets. CPU-only (fake engines; the policy and record logic
is byte-level).

Tests:
- test_simple_policy_round_trip_world2: the simple policy saves whole buffers per writer; each rank restores bitwise from its OWN snapshot (keyed targets), and the logical view reassembles the aggregate O ([m_all | v_all]) and picks the authoritative W.
- test_dedup_policy_covers_with_disjoint_slices: dedup saves W as disjoint authoritative responsibility slices — one copy total — and the logical view reassembles it bitwise.
- test_replication_drift_refuses_before_record: diverged W bytes between writers make save_checkpoint refuse naming both writers, and NO record lands.
"""
import threading
import time

import numpy as np
import pytest

from dataflow.checkpoint import (CheckpointError, RECORD_NAME,
                                 read_record, resolve_targets,
                                 save_checkpoint)
from dataflow.service import EngineClient, EngineConfig, Server
from dataflow_training.distributed.source_policy import \
    compile_source_policy

W_BYTES = 8192
N_ELEMS = 512                        # zero1 elements per rank (fp32)
O_LOCAL = 2 * N_ELEMS * 4            # [m_r | v_r]
OPT_SLICES = {"W_0": {"n_slice": N_ELEMS, "n_tail": 0,
                      "opt_dtype": "fp32"}}
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


def fleet_state(w_bytes_by_rank):
    """(clients, o_bytes) for a world-2 fleet with the given per-rank
    W payloads and distinct zero1 O shards."""
    o = {r: rng_bytes(100 + r, O_LOCAL) for r in (0, 1)}
    return o


def compiled_writers(clients, o, policy):
    writer_specs = {r: [("W_0", W_BYTES), ("O_0", O_LOCAL)]
                    for r in (0, 1)}
    logical, per_writer = compile_source_policy(
        policy=policy, world=2, writer_specs=writer_specs,
        plan=PLAN, opt_slices=OPT_SLICES)
    writers = {}
    for r in (0, 1):
        writers[r] = {"client": clients[r], "path": f"rank{r}",
                      "slices": per_writer[r]["slices"],
                      "record": per_writer[r]["record"],
                      "client_meta": {"step": 4, "rank": r}}
    return logical, writers


def aggregate_o(o) -> bytes:
    half = O_LOCAL // 2
    return (o[0][:half] + o[1][:half]
            + o[0][half:] + o[1][half:])


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
        logical, writers = compiled_writers(clients, o, "simple")
        step_dir = tmp_path / "step_000004"
        record = save_checkpoint(
            writers, step_dir, step=4, seed=11,
            logical_objects=logical,
            scheme={"world": 2, "kind": "zero1rs",
                    "source_policy": "simple"},
            client_payload={"losses": [5.0]})
        assert (step_dir / RECORD_NAME).is_file()
        assert read_record(step_dir)["scheme"]["source_policy"] == \
            "simple"

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

        # the logical view: authoritative W + aggregate [m_all|v_all]
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
        logical, writers = compiled_writers(clients, o, "dedup")
        step_dir = tmp_path / "step_000008"
        record = save_checkpoint(
            writers, step_dir, step=8, seed=11,
            logical_objects=logical,
            scheme={"source_policy": "dedup"})

        w_slices = record["slices"]["W_0"]
        assert len(w_slices) == 2
        assert all(s["authoritative"] for s in w_slices)
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
        logical, writers = compiled_writers(clients, o, "simple")
        step_dir = tmp_path / "step_000012"
        with pytest.raises(CheckpointError) as ei:
            save_checkpoint(writers, step_dir, step=12, seed=11,
                            logical_objects=logical)
        message = str(ei.value)
        assert "drift" in message and "0" in message and "1" in message
        assert not (step_dir / RECORD_NAME).is_file(), \
            "a drifted save must leave NO record"
    finally:
        for _server, c in daemons:
            c.shutdown()
