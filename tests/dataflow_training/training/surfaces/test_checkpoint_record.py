"""Conductor-shaped checkpoint flow over the record: the training
save path compiles its policy and lands checkpoint_record.json LAST;
resume locates the newest COMPLETE step and each rank restores its
own self-sufficient snapshot; the loader resolves targets — including
the weights-only load, where optimizer bytes never enter the store.
CPU-only (fake engines).

Tests:
- test_conductor_save_resume_round_trip: the conductor save path writes a v1 record whose engine_spec carries each source's capability (backing/device/kernel_set/fake) and whose inventories sum to the rank state; resolve_resume(auto) picks the newest complete step and skips a step dir without a record; each rank's own-snapshot restore is bitwise and carries NO client_meta (the record's client_payload owns run state).
- test_load_checkpoint_targets: weights-only targets leave ZERO optimizer bytes resident in a scratch engine sized from the plan; "all" reassembles the aggregate optimizer object; a source key restores that rank's shard view.
- test_load_checkpoint_refuses_small_engine: a supplied engine whose backing cannot hold the targets refuses loudly BEFORE any restore call (capability, never placement).
- test_load_checkpoint_engines_mapping: an engines mapping restores every source's FULL rank view bitwise into the caller's engines; a mapping missing a source refuses, and a too-small engine in the mapping refuses on capability before any restore.
"""
import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest

from dataflow.checkpoint import RECORD_NAME
from dataflow.core.jsonio import program_to_dict
from dataflow.core.program import ObjectSpec, Program
from dataflow.service import EngineClient, EngineConfig, Server
from dataflow_training.blocks.layouts import opt_state_slice_layout
from dataflow_training.run.checkpointing import (load_checkpoint,
                                                 resolve_resume,
                                                 save_checkpoint)

W_BYTES = 8192
N_SLICE = 500              # 2000 B per slot: NOT 256-aligned
OPT_SLICES = {"W_0": {"n_slice": N_SLICE, "n_tail": 0,
                      "opt_dtype": "fp32"}}
O_LAYOUT = opt_state_slice_layout(N_SLICE, 0, "fp32")
O_LOCAL = O_LAYOUT.total_bytes       # aligned fields + padded total
PLAN = {"W_0": [
    {"rank": 0, "lo": 0, "hi": W_BYTES // 2, "role": "responsible"},
    {"rank": 1, "lo": W_BYTES // 2, "hi": W_BYTES,
     "role": "responsible"},
]}


@dataclass
class StubHost:
    name: str
    device: int = 0

    def is_local(self) -> bool:
        return True


@dataclass
class StubRank:
    name: str
    client: object
    prog_dict: dict


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


def shard_bytes(seed):
    """Layout-true [m_slice | v_slice] payload: rng content in the
    fields, ZERO alignment padding (padding is layout, not state)."""
    out = bytearray(O_LOCAL)
    for f in O_LAYOUT.fields:
        out[f.offset_bytes:f.offset_bytes + f.nbytes] = \
            rng_bytes(seed * 7 + f.offset_bytes, f.nbytes)
    return bytes(out)


def field_bytes(payload: bytes, name: str) -> bytes:
    f = O_LAYOUT.field(name)
    return payload[f.offset_bytes:f.offset_bytes + f.nbytes]


def rank_program() -> dict:
    return program_to_dict(Program(
        name="drill",
        initial_objects=(
            ObjectSpec("W_0", W_BYTES, persistent=True),
            ObjectSpec("O_0", O_LOCAL, persistent=True),
            ObjectSpec("tokens_0_0", 128, role="input"),
        )))


def fleet(tmp_path, w, o):
    daemons = []
    ranks = []
    for r in (0, 1):
        server, client = boot(tmp_path, f"r{r}")
        daemons.append((server, client))
        client.put_object("W_0", w)
        client.put_object("O_0", o[r])
        ranks.append(StubRank(name=f"host{r}", client=client,
                              prog_dict=rank_program()))
    ck = {"dir": tmp_path / "run", "run": "drill",
          "responsibility": PLAN, "opt_slices": OPT_SLICES,
          "source_policy": "simple", "keep_last": 0,
          "argv": ["train.py"], "resolved": {"preset": "drill"},
          "data_meta": {},
          "hosts_by_name": {f"host{r}": StubHost(f"host{r}")
                            for r in (0, 1)}}
    return daemons, ranks, ck


def quiet(_msg):
    return None


class TinyEngineStub:
    """query_backing-only stand-in whose engine is too small for any
    restore — the capability refusal must fire BEFORE a single
    restore call reaches it."""

    def query_backing(self):
        return {"capacity_bytes": 1024}

    def restore_snapshot(self, *args, **kwargs):
        raise AssertionError(
            "the capability refusal must precede restores")


def test_conductor_save_resume_round_trip(tmp_path):
    w = rng_bytes(7, W_BYTES)
    o = {r: shard_bytes(100 + r) for r in (0, 1)}
    daemons, ranks, ck = fleet(tmp_path, w, o)
    ck["dir"].mkdir(parents=True, exist_ok=True)
    try:
        meta = {"seed": 11, "rank_rounds": [1, 1], "backend": "fake",
                "hosts": ["host0", "host1"], "data_cursor": {"doc": 9}}
        save_checkpoint(ranks, ck, 4, meta, [5.0, 4.0], quiet)
        save_checkpoint(ranks, ck, 8, meta, [5.0, 4.0, 3.0], quiet)
        (ck["dir"] / "step_000012" / "rank0").mkdir(parents=True)

        record = resolve_resume(ck["dir"], "auto", quiet)
        assert record["step"] == 8, \
            "auto must pick the newest COMPLETE step (12 has no record)"
        assert record["client_payload"]["losses"] == [5.0, 4.0, 3.0]
        assert record["client_payload"]["data_cursor"] == {"doc": 9}
        assert record["scheme"]["source_policy"] == "simple"
        assert record["launch"]["resolved"]["world"] == 2

        # the record carries each source's CAPABILITY (what room the
        # state needs and what engine wrote it), never placement
        from dataflow.checkpoint import source_resident_bytes

        spec = record["engine_spec"]["0"]
        assert spec["fake"] is True and "kernel_set" in spec
        assert spec["backing_gib"] > 0
        assert source_resident_bytes(record, "0") == W_BYTES + O_LOCAL

        # each rank restores its OWN snapshot through the keyed plan
        # (the rank view: logical slices remapped to local geometry)
        from dataflow.checkpoint import resolve_targets

        step_dir = ck["dir"] / "step_000008"
        for r in (0, 1):
            server_f, fresh = boot(tmp_path, f"fresh{r}")
            try:
                plan = resolve_targets(record,
                                       {str(r): ["W_0", "O_0"]})
                assert len(plan) == 1, \
                    "the simple policy resumes from ONE own snapshot"
                res = fresh.restore_snapshot(
                    str(step_dir / plan[0]["path"]),
                    remap=plan[0]["remap"])
                assert not res.get("client_meta"), \
                    "training snapshots carry NO client_meta — the " \
                    "record's client_payload owns run state"
                assert bytes(fresh.get_object("W_0")) == w
                assert bytes(fresh.get_object("O_0")) == o[r]
            finally:
                fresh.shutdown()
        assert (step_dir / RECORD_NAME).is_file()
        assert (step_dir / "programs" / "rank1.json").is_file()
    finally:
        for _server, c in daemons:
            c.shutdown()


def test_load_checkpoint_targets(tmp_path):
    w = rng_bytes(13, W_BYTES)
    o = {r: shard_bytes(200 + r) for r in (0, 1)}
    daemons, ranks, ck = fleet(tmp_path, w, o)
    ck["dir"].mkdir(parents=True, exist_ok=True)
    try:
        meta = {"seed": 11, "rank_rounds": [1, 1], "backend": "fake",
                "hosts": ["host0", "host1"], "data_cursor": None}
        save_checkpoint(ranks, ck, 4, meta, [5.0], quiet)
        step_dir = ck["dir"] / "step_000004"

        # weights-only: optimizer bytes never enter the store, and
        # the scratch engine sizes itself from the plan (the tiny
        # targets land on the floor, not the old fixed default)
        record, client = load_checkpoint(step_dir, targets=["W_0"])
        try:
            assert bytes(client.get_object("W_0")) == w
            assert client.query_object("O_0") is None, \
                "weights-only targets must leave ZERO optimizer bytes"
            capacity = client.query_backing()["capacity_bytes"]
            assert capacity < 512 * 1024 ** 2, \
                "scratch engine must be sized from the resolved plan"
        finally:
            client.shutdown()

        # the logical view reassembles the tight [m_all | v_all]
        # from the sources' aligned slice fields
        expect = (field_bytes(o[0], "m_slice")
                  + field_bytes(o[1], "m_slice")
                  + field_bytes(o[0], "v_slice")
                  + field_bytes(o[1], "v_slice"))
        record, client = load_checkpoint(step_dir, targets="all")
        try:
            assert bytes(client.get_object("O_0")) == expect
        finally:
            client.shutdown()

        # a source key restores that rank's shard view
        record, client = load_checkpoint(step_dir,
                                         targets={"1": ["O_0"]})
        try:
            assert bytes(client.get_object("O_0")) == o[1]
        finally:
            client.shutdown()
    finally:
        for _server, c in daemons:
            c.shutdown()


def test_load_checkpoint_refuses_small_engine(tmp_path):
    w = rng_bytes(31, W_BYTES)
    o = {r: shard_bytes(300 + r) for r in (0, 1)}
    daemons, ranks, ck = fleet(tmp_path, w, o)
    ck["dir"].mkdir(parents=True, exist_ok=True)
    try:
        meta = {"seed": 11, "rank_rounds": [1, 1], "backend": "fake",
                "hosts": ["host0", "host1"], "data_cursor": None}
        save_checkpoint(ranks, ck, 4, meta, [5.0], quiet)
        from dataflow.checkpoint import CheckpointError

        with pytest.raises(CheckpointError, match="cannot hold"):
            load_checkpoint(ck["dir"] / "step_000004", targets="all",
                            client=TinyEngineStub())
    finally:
        for _server, c in daemons:
            c.shutdown()


def test_load_checkpoint_engines_mapping(tmp_path):
    w = rng_bytes(41, W_BYTES)
    o = {r: shard_bytes(400 + r) for r in (0, 1)}
    daemons, ranks, ck = fleet(tmp_path, w, o)
    ck["dir"].mkdir(parents=True, exist_ok=True)
    fresh = []
    try:
        meta = {"seed": 11, "rank_rounds": [1, 1], "backend": "fake",
                "hosts": ["host0", "host1"], "data_cursor": None}
        save_checkpoint(ranks, ck, 4, meta, [5.0], quiet)
        step_dir = ck["dir"] / "step_000004"
        pair = {}
        for r in (0, 1):
            server_f, c = boot(tmp_path, f"eng{r}")
            fresh.append(c)
            pair[str(r)] = c

        record, clients = load_checkpoint(step_dir, engines=pair)
        for r in (0, 1):
            assert bytes(clients[str(r)].get_object("W_0")) == w
            assert bytes(clients[str(r)].get_object("O_0")) == o[r], \
                f"source {r}'s rank view must restore ITS shard"

        from dataflow.checkpoint import CheckpointError

        with pytest.raises(CheckpointError, match="lacks sources"):
            load_checkpoint(step_dir, engines={"0": pair["0"]})
        with pytest.raises(CheckpointError, match="cannot hold"):
            load_checkpoint(step_dir,
                            engines={"0": pair["0"],
                                     "1": TinyEngineStub()})
    finally:
        for c in fresh:
            c.shutdown()
        for _server, c in daemons:
            c.shutdown()
