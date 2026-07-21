"""Checkpoint-record (v2) gates (CPU, fake engines): the end-to-end checkpoint
shape without the conductor — two daemons save per a zero1rs-shaped
responsibility plan (ranged params + whole own shards), the record
writes LAST, and restore-by-artifact-order reassembles every object
BITWISE on a fresh daemon pair. Plus: format guard (loud on v1/absent),
completeness marker (no checkpoint_record.json = no checkpoint), launch record
round-trip, programs saved beside artifacts."""
import threading
import time

import numpy as np
import pytest

from dataflow.service import EngineClient, EngineConfig, Server
from dataflow_training.run.checkpoint_record import (
    artifacts_for_restore,
    launch_record,
    read_record,
    save_programs,
    write_record,
)
from dataflow_training.distributed.responsibility import rank_save_args

N = 1 << 18                      # 256 KiB params
O_N = N // 2                     # per-rank opt shard


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


def wait_snap(client, out):
    for _ in range(600):
        s = client.snapshot_status(out["snap_id"])
        if s["state"] == "done":
            return
        if s["state"] == "error":
            raise AssertionError(s)
        time.sleep(0.01)
    raise AssertionError("snapshot timeout")


def test_manifest_v2_roundtrip_two_ranks(tmp_path):
    rng = np.random.default_rng(3)
    w = rng.integers(0, 256, N, dtype=np.uint8).tobytes()
    o0 = rng.integers(0, 256, O_N, dtype=np.uint8).tobytes()
    o1 = rng.integers(0, 256, O_N, dtype=np.uint8).tobytes()

    # zero1rs-shaped plan: W partitioned at the halfway boundary,
    # each rank's O shard its own (same object id, different bytes —
    # per-rank stores, exactly the zero1rs situation)
    plan = {"W_0": [
        {"rank": 0, "lo": 0, "hi": N // 2, "role": "responsible"},
        {"rank": 1, "lo": N // 2, "hi": N, "role": "responsible"},
    ]}
    step_dir = tmp_path / "step_000004"
    step_dir.mkdir()

    daemons = []
    for r, o_bytes in ((0, o0), (1, o1)):
        server, c = boot(tmp_path, f"m2-r{r}")
        daemons.append((server, c))
        c.put_object("W_0", w)
        c.put_object("O_0", o_bytes)
        ids, ranges = rank_save_args(plan, r, own_objects=["O_0"])
        assert ids == ["O_0", "W_0"]
        assert ("W_0" in ranges) and ("O_0" not in ranges)
        wait_snap(c, c.snapshot("all", str(step_dir / f"rank{r}"),
                                ids=ids, ranges=ranges,
                                client_meta={"rank": r, "step": 4}))

    progs = save_programs(step_dir, [{"tasks": ["t0"]}, {"tasks": ["t1"]}])
    launch = launch_record(argv=["train.py", "train"],
                           resolved={"preset": "unit", "world": 2},
                           data={"scheme": "unit"},
                           ranks=[{"host": "local", "device": 0},
                                  {"host": "local", "device": 0}],
                           repo=tmp_path, programs=progs)
    # completeness marker: BEFORE checkpoint_record.json, read must refuse
    with pytest.raises(RuntimeError):
        read_record(step_dir)
    write_record(step_dir, step=4, seed=11, world=2,
                   data_cursor={"doc": 9}, losses=[5.0, 4.0],
                   save_plan=plan, artifacts=["rank0", "rank1"],
                   launch=launch)
    m = read_record(step_dir)
    assert m["format"] == 2
    assert m["data_cursor"] == {"doc": 9}
    assert m["launch"]["argv"] == ["train.py", "train"]
    assert m["launch"]["programs"] == ["programs/rank0.json",
                                       "programs/rank1.json"]
    assert (step_dir / "programs" / "rank1.json").is_file()

    # fresh pair restores by artifact order -> bitwise reassembly
    for r, want_o in ((0, o0), (1, o1)):
        server, c = boot(tmp_path, f"m2-fresh{r}")
        daemons.append((server, c))
        for art in artifacts_for_restore(m, r):
            c.restore_snapshot(str(step_dir / art), overwrite=True)
        assert bytes(c.get_object("W_0")) == w, f"rank {r} W diverged"
        assert bytes(c.get_object("O_0")) == want_o, \
            f"rank {r} O shard clobbered"

    for server, c in daemons:
        try:
            c.shutdown()
        except Exception:
            pass


def test_format_guard_is_loud(tmp_path):
    import json

    step_dir = tmp_path / "step_000002"
    step_dir.mkdir()
    (step_dir / "checkpoint_record.json").write_text(json.dumps({"step": 2}))
    with pytest.raises(RuntimeError, match="format"):
        read_record(step_dir)
