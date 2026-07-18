"""S1.1 gates: allocator, object CRUD over the wire (binary frames both
directions), groups/hierarchy/reserved names, duplicate+lineage,
protect/wipe, queries — CPU (fake slab). GPU: real pinned boot +
init-program byte-identity vs initial_values.
"""
from __future__ import annotations

import threading
import time

import pytest

from dataflow.service import EngineClient, EngineConfig, Server, ServiceError
from dataflow.service.store import ALIGN, SlabAllocator, Store


def _boot(tmp_path, name, **cfg):
    sock = str(tmp_path / f"{name}.sock")
    server = Server(EngineConfig(socket_path=sock, **cfg))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    for _ in range(300):
        try:
            with EngineClient(sock, client_name="probe"):
                break
        except (ConnectionError, FileNotFoundError, OSError):
            time.sleep(0.01)
    else:
        raise RuntimeError("daemon did not come up")
    return sock, server, t


@pytest.fixture()
def daemon(tmp_path):
    sock, server, _ = _boot(tmp_path, "store", fake=True,
                            slab_backing_gib=0.0625)   # 64 MiB
    yield sock, server
    server.state.shutdown_requested.set()
    server.dispatcher.stop()


# ------------------------------------------------------------- allocator

def test_allocator_coalesce_and_reuse():
    a = SlabAllocator(1 << 20)
    x = a.alloc(100_000)
    y = a.alloc(200_000)
    z = a.alloc(50_000)
    assert x.offset % ALIGN == y.offset % ALIGN == 0
    a.release(y)
    assert a.stats()["free_extents"] == 2      # gap + tail
    y2 = a.alloc(150_000)                       # fits the gap (first fit)
    assert y2.offset == y.offset
    a.release(x)
    a.release(y2)
    a.release(z)
    st = a.stats()
    assert st["free_extents"] == 1 and st["used_bytes"] == 0
    assert st["largest_free_extent"] == 1 << 20


def test_allocator_capacity_error_detail():
    a = SlabAllocator(1 << 16)
    a.alloc(1 << 15)
    with pytest.raises(ServiceError) as ei:
        a.alloc(1 << 16)
    assert ei.value.code == "CAPACITY"
    assert ei.value.detail["largest_free"] < (1 << 16)


# ---------------------------------------------------------- CRUD (wire)

def test_put_get_roundtrip_bytes(daemon):
    sock, _ = daemon
    blob = bytes(range(256)) * 1000
    with EngineClient(sock, client_name="crud") as c:
        r = c.put_object("blob/a", blob, meta={"role": "test"})
        assert r["object"]["size_bytes"] == len(blob)
        back = c.get_object("blob/a")
        assert back == blob
        info = c.query_object("blob/a")
        assert info["meta"]["role"] == "test"
        assert info["lineage"] == {"parent": None, "dirty": True}


def test_put_get_file_forms(daemon, tmp_path):
    sock, _ = daemon
    src = tmp_path / "src.bin"
    src.write_bytes(b"\xab" * 123_457)
    with EngineClient(sock, client_name="files") as c:
        c.put_object("blob/f", path=src)
        out = tmp_path / "out" / "dst.bin"
        r = c.get_object("blob/f", dest=out)
        assert r["bytes"] == 123_457
        assert out.read_bytes() == src.read_bytes()


def test_overwrite_same_size_ok_mismatch_rejected(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="ow") as c:
        c.put_object("x", b"a" * 1000)
        c.put_object("x", b"b" * 1000)          # same size: fine
        assert c.get_object("x") == b"b" * 1000
        with pytest.raises(ServiceError) as ei:
            c.put_object("x", b"c" * 999)
        assert ei.value.code == "BINDING_MISMATCH"


def test_unknown_object(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="uo") as c:
        assert c.query_object("nope") is None
        with pytest.raises(ServiceError) as ei:
            c.get_object("nope")
        assert ei.value.code == "UNKNOWN_OBJECT"


# ------------------------------------------------------------ materialize

def test_materialize_zeros_and_tokens(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="mat") as c:
        c.materialize_object("z", {"kind": "zeros", "size_bytes": 4096})
        assert c.get_object("z") == b"\x00" * 4096
        c.materialize_object("tok", {"kind": "tokens", "vocab": 97,
                                     "n": 1024, "seed": 3})
        import numpy as np

        ids = np.frombuffer(c.get_object("tok"), dtype=np.int32)
        assert ids.shape == (1024,) and ids.min() >= 0 and ids.max() < 97
        # deterministic per seed
        c.materialize_object("tok2", {"kind": "tokens", "vocab": 97,
                                      "n": 1024, "seed": 3})
        assert c.get_object("tok2") == c.get_object("tok")


# ---------------------------------------------------- groups + duplicate

def test_object_groups_hierarchy_and_reserved(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="grp") as c:
        for i in range(3):
            c.put_object(f"W_{i}", b"w" * 1000)
            c.put_object(f"O_{i}", b"o" * 2000)
        c.create_object_group("weights", pattern="W_*")
        c.create_object_group("opt_state", pattern="O_*")
        g = c.create_object_group("model_state",
                                  object_groups=["weights", "opt_state"])
        assert g["object_group"]["n_members"] == 6
        q = c.query_object_group("model_state")
        assert q["bytes"] == 3 * 1000 + 3 * 2000
        with pytest.raises(ServiceError):
            c.create_object_group("backing")     # reserved
        with pytest.raises(ServiceError) as ei:
            c.query_object_group("nope")
        assert ei.value.code == "UNKNOWN_GROUP"


def test_duplicate_lineage_and_group_dup(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="dup") as c:
        c.put_object("W_0", b"w0" * 500)
        c.put_object("W_1", b"w1" * 500)
        c.create_object_group("weights", pattern="W_*")
        r = c.duplicate_object("W_0", "W_0@init")
        assert r["object"]["lineage"] == {"parent": "W_0", "dirty": False}
        assert c.get_object("W_0@init") == b"w0" * 500

        d = c.duplicate_object_group("weights", tag="ck1")
        assert d["mapping"] == {"W_0": "W_0@ck1", "W_1": "W_1@ck1"}
        assert c.query_object_group("weights@ck1")["n_members"] == 2

        # overwriting the parent dirties its lineage record
        c.put_object("W_0", b"XX" * 500)
        assert c.query_object("W_0")["lineage"]["dirty"] is True


# ------------------------------------------------------- protect + wipe

def test_protect_release_wipe(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="pw") as c:
        c.put_object("keep", b"k" * 100)
        c.put_object("drop", b"d" * 100)
        c.protect_object("keep")
        with pytest.raises(ServiceError) as ei:
            c.release_object("keep")
        assert ei.value.code == "PROTECTED"

        r = c.wipe("all")
        assert r["skipped"] == ["keep"] and r["n_objects"] >= 1
        assert c.query_object("drop") is None
        assert c.query_object("keep") is not None

        r = c.wipe("all", force=True)
        assert c.query_object("keep") is None
        u = c.query_backing()
        assert u["used_bytes"] == 0 and u["n_objects"] == 0


def test_query_backing_usage(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="q") as c:
        c.put_object("big", b"B" * 100_000)
        c.put_object("small", b"s" * 10)
        u = c.query_backing()
        assert u["n_objects"] == 2
        assert u["largest"][0][0] == "big"
        assert u["used_bytes"] >= 100_000
        assert c.engine_status()["pools"]["backing"]["n_objects"] == 2


# ------------------------------------------------------------- GPU gates

@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_real_boot_family_init_byte_identity(tmp_path):
    """Real pinned slab: the init PROGRAM persists initial objects
    byte-identical to in-process initial_values (init-as-program
    replaced the retired materialize_group verb)."""
    from dataflow_training.register import register_all

    register_all()      # in-process Server shares this registry
    sock, server, _ = _boot(tmp_path, "real", fake=False,
                            slab_backing_gib=2.0)
    try:
        from dataflow.runtime.device.cuda import CudaBackend
        from dataflow.runtime.interop import torch_view
        from dataflow_training.model_families.families import family
        from dataflow_training.run.driver import init_model

        fam = family("llama3")
        cfg = fam.config_type.tiny()
        cfg_dict = {"n_layers": cfg.n_layers, "d_model": cfg.d_model,
                    "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
                    "d_ff": cfg.d_ff, "vocab_size": cfg.vocab_size,
                    "seq_len": cfg.seq_len, "batch": cfg.batch}
        with EngineClient(sock, client_name="gpu") as c:
            created = init_model(c, "llama3", cfg_dict, seed=7)
            assert "W_0" in created and "O_0" in created

            ref_cfg = fam.config_type(**cfg_dict)
            prog = fam.lower(ref_cfg)
            ref = fam.initial_values(prog, ref_cfg, CudaBackend(), seed=7)
            import torch

            for oid in ("W_0", "O_0", "W_embed", "tokens_0_0"):
                got = c.get_object(oid)
                want = torch_view(ref[oid], (ref[oid].size_bytes,),
                                  torch.uint8).cpu().numpy().tobytes()
                assert got == want, f"{oid}: bytes differ"
            u = c.query_backing()
            assert u["used_bytes"] >= sum(
                s.size_bytes for s in prog.initial_objects)
    finally:
        server.state.shutdown_requested.set()
        server.dispatcher.stop()
        if server.store.slab is not None:
            server.store.slab.free()
