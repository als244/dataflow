"""Store gates: allocator, object CRUD over the wire (binary frames both
directions), duplicate, protect/wipe, queries — CPU (fake slab).
GPU: real pinned boot + init-program byte-identity vs initial_values.

Tests:
- test_allocator_coalesce_and_reuse: the slab allocator aligns offsets, first-fit-reuses a freed gap, and coalesces back to one full-capacity extent once all is released.
- test_allocator_capacity_error_detail: an over-capacity alloc raises CAPACITY carrying the largest free extent in its detail.
- test_put_get_roundtrip_bytes: put/get round-trips bytes and query reports the stored meta.
- test_put_get_file_forms: the file-path put/get forms write and read byte-identical content on disk.
- test_overwrite_same_size_ok_mismatch_rejected: a same-size overwrite succeeds but a different-size one raises BINDING_MISMATCH.
- test_unknown_object: querying a missing object returns None and getting it raises UNKNOWN_OBJECT.
- test_materialize_zeros_and_tokens: materialize produces zero bytes and per-seed-deterministic in-range token ids.
- test_duplicate_copies_bytes_independently: duplicate copies bytes and meta into an independent object — mutating the parent afterwards leaves the copy untouched, and duplicating onto an existing id refuses.
- test_protected_object_survives_wipe_unless_forced: a protected object refuses release and survives wipe until force=True.
- test_query_backing_usage: query_backing reports object count, largest object, and used bytes consistent with engine_status; peak_bytes is the slab high-water — it survives a release and resets to the surviving level on wipe.
- test_create_object_semantics: create_object allocates a payload-less catalogued extent (query shows the size), same-size re-create is idempotent, and a size change raises BINDING_MISMATCH.
- test_host_run_fills_in_place: an all-host program binds pre-created store extents and its task's writes land directly in the catalogued objects (fake boot: plain bytearray memory) — no engine, no placement, no transient copies.
- test_host_program_registration_guards: mixed host+device programs, host tasks with outputs, and host tasks touching non-initial ids are each rejected at registration as BAD_REQUEST.
- test_program_content_id_matches_registration: the client-side program_content_id hash equals the prog_id registration returns, and survives a program dict -> Program -> dict round-trip (the plan-artifact gate rests on both).
- test_real_boot_family_init_byte_identity: on a real pinned slab the init program persists weight and opt-state objects byte-identical to in-process initial_values and materializes no input-role object.
- test_real_boot_init_fits_tight_slab: on a real pinned slab sized ~1.3x the model state, init_model succeeds and leaves used bytes ~= exactly the model — the in-place host fill needs no transient double (the old output-task shape needed ~2x and failed this slab).
"""
from __future__ import annotations

import time

import pytest

from dataflow.service import EngineClient, EngineConfig, Server, ServiceError
from dataflow.service.store import ALIGN, SlabAllocator, Store
from tests.support.service import start_server_thread, stop_server_thread


def _boot(tmp_path, name, **cfg):
    sock = str(tmp_path / f"{name}.sock")
    server = Server(EngineConfig(socket_path=sock, **cfg))
    thread = start_server_thread(server)
    try:
        for _ in range(300):
            try:
                with EngineClient(sock, client_name="probe"):
                    break
            except (ConnectionError, FileNotFoundError, OSError):
                time.sleep(0.01)
        else:
            raise RuntimeError("daemon did not come up")
    except BaseException:
        stop_server_thread(server, thread)
        raise
    return sock, server, thread


@pytest.fixture()
def daemon(tmp_path):
    sock, server, thread = _boot(tmp_path, "store", fake=True,
                                 slab_backing_gib=0.0625)   # 64 MiB
    try:
        yield sock, server
    finally:
        stop_server_thread(server, thread)


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


# ---------------------------------------------------------- duplicate

def test_duplicate_copies_bytes_independently(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="dup") as c:
        c.put_object("W_0", b"w0" * 500, meta={"role": "weights"})
        r = c.duplicate_object("W_0", "W_0@init")
        assert r["object"]["size_bytes"] == 1000
        assert r["object"]["meta"] == {"role": "weights"}
        assert c.get_object("W_0@init") == b"w0" * 500

        # the copy is independent: mutating the parent changes nothing
        c.put_object("W_0", b"XX" * 500)
        assert c.get_object("W_0@init") == b"w0" * 500

        with pytest.raises(ServiceError) as ei:
            c.duplicate_object("W_0", "W_0@init")     # exists
        assert ei.value.code == "BAD_REQUEST"


# ------------------------------------------------------- protect + wipe

def test_protected_object_survives_wipe_unless_forced(daemon):
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

        # peak is the high-water: releasing does not lower it, wiping
        # restarts it from whatever survived
        assert u["peak_bytes"] >= u["used_bytes"]
        high = u["peak_bytes"]
        c.release_object("big")
        after = c.query_backing()
        assert after["used_bytes"] < u["used_bytes"]
        assert after["peak_bytes"] == high
        c.wipe("all", force=True)
        wiped = c.query_backing()
        assert wiped["peak_bytes"] == wiped["used_bytes"]


# --------------------------------------------------- create + host runs

def test_create_object_semantics(daemon):
    sock, _ = daemon
    with EngineClient(sock, client_name="co") as c:
        r = c.create_object("pre/a", 4096)
        assert r["object"]["size_bytes"] == 4096
        assert c.query_object("pre/a")["size_bytes"] == 4096
        c.create_object("pre/a", 4096)            # same size: idempotent
        with pytest.raises(ServiceError) as ei:
            c.create_object("pre/a", 8192)
        assert ei.value.code == "BINDING_MISMATCH"
        # content is unspecified until first written; a same-size put
        # defines it in place
        c.put_object("pre/a", b"\x07" * 4096)
        assert c.get_object("pre/a") == b"\x07" * 4096


class HostFillExecutable:
    """Toy HOST task: memsets each mutated object's extent with a byte
    derived from its id — the writes land in the store's catalogued
    memory directly (that in-place landing is what the test pins)."""

    def launch(self, ctx) -> None:
        import ctypes

        for oid, buf in ctx.mutates.items():
            ctypes.memset(buf.ptr, sum(oid.encode()) % 251, buf.size_bytes)


class HostFillResolver:
    def __call__(self, task):
        return HostFillExecutable()


def build_host_fill_resolver(spec: dict):
    return HostFillResolver()


def register_host_fill():
    from dataflow.service.registry import register_program_resolver

    register_program_resolver("host_fill", build_host_fill_resolver)
    return {"kind": "host_fill"}


def host_program_dict(objects: dict[str, int]) -> dict:
    from dataflow.core.jsonio import program_to_dict
    from dataflow.core.program import ObjectSpec, Program, TaskSpec

    ids = tuple(sorted(objects))
    return program_to_dict(Program(
        name="host-fill",
        initial_objects=tuple(ObjectSpec(id=o, size_bytes=objects[o])
                              for o in ids),
        tasks=(TaskSpec(id="fill_0", inputs=ids, mutates=ids, host=True,
                        compute_block_key="host_fill"),),
        final_locations={o: "backing" for o in ids},
    ))


def test_host_run_fills_in_place(daemon):
    sock, _ = daemon
    spec = register_host_fill()
    sizes = {"hf/a": 1000, "hf/b": 2000}
    with EngineClient(sock, client_name="host") as c:
        for oid, n in sizes.items():
            c.create_object(oid, n)
        reg = c.register_program(host_program_dict(sizes), resolver=spec)
        assert reg["bindings"]["missing_inputs"] == []
        out = c.run(reg["prog_id"], args={})
        assert out["state"] == "done"
        for oid, n in sizes.items():
            want = bytes([sum(oid.encode()) % 251]) * n
            assert c.get_object(oid) == want, oid
        c.unregister_program(reg["prog_id"])

        # an object the program declares but nobody created: the report
        # names it at registration, and the run refuses loudly
        more = dict(sizes)
        more["hf/missing"] = 512
        reg2 = c.register_program(host_program_dict(more), resolver=spec)
        assert reg2["bindings"]["missing_inputs"] == ["hf/missing"]
        with pytest.raises(ServiceError) as ei:
            c.run(reg2["prog_id"], args={})
        assert ei.value.code == "MISSING_INPUTS"


def test_host_program_registration_guards(daemon):
    from dataflow.core.jsonio import program_to_dict
    from dataflow.core.program import (ObjectSpec, OutputSpec, Program,
                                       TaskSpec)

    sock, _ = daemon
    spec = register_host_fill()
    obj = ObjectSpec(id="hg/a", size_bytes=64)

    mixed = program_to_dict(Program(
        name="mixed",
        initial_objects=(obj,),
        tasks=(TaskSpec(id="h", inputs=("hg/a",), mutates=("hg/a",),
                        host=True, compute_block_key="host_fill"),
               TaskSpec(id="d", compute_block_key="host_fill")),
    ))
    with_outputs = program_to_dict(Program(
        name="houts",
        initial_objects=(obj,),
        tasks=(TaskSpec(id="h", inputs=("hg/a",), mutates=("hg/a",),
                        host=True, compute_block_key="host_fill",
                        outputs=(OutputSpec(id="hg/out", size_bytes=32,
                                            location="backing"),)),),
    ))
    stray = program_to_dict(Program(
        name="hstray",
        initial_objects=(obj,),
        tasks=(TaskSpec(id="h", inputs=("hg/a", "hg/ghost"),
                        mutates=("hg/ghost",), host=True,
                        compute_block_key="host_fill"),),
    ))
    with EngineClient(sock, client_name="guards") as c:
        for pd, why in ((mixed, "mixes"), (with_outputs, "outputs"),
                        (stray, "initial_objects")):
            with pytest.raises(ServiceError) as ei:
                c.register_program(pd, resolver=spec)
            assert ei.value.code == "BAD_REQUEST"
            assert why in str(ei.value), why


def test_program_content_id_matches_registration(daemon):
    from dataflow.core.jsonio import program_from_dict, program_to_dict
    from dataflow.service.wire import program_content_id

    sock, _ = daemon
    spec = register_host_fill()
    pd = host_program_dict({"pi/a": 256})
    # round-trip stability: an artifact stores the dict, measure re-parses
    # and re-serializes it — the hash must survive that
    assert program_content_id(program_to_dict(program_from_dict(pd))) == program_content_id(pd)
    with EngineClient(sock, client_name="pid") as c:
        c.create_object("pi/a", 256)
        reg = c.register_program(pd, resolver=spec)
        assert reg["prog_id"] == program_content_id(pd)
        c.unregister_program(reg["prog_id"])


# ------------------------------------------------------------- GPU gates

@pytest.mark.gpu
def test_real_boot_family_init_byte_identity(tmp_path):
    """Real pinned slab: the init PROGRAM persists initial objects
    byte-identical to in-process initial_values (init-as-program
    replaced the retired materialize_group verb)."""
    pytest.importorskip("cuda.bindings.runtime")  # real pinned slab + CudaBackend
    from dataflow_training.register import register_all

    register_all()      # in-process Server shares this registry
    sock, server, thread = _boot(tmp_path, "real", fake=False,
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

            for oid in ("W_0", "O_0", "W_embed"):
                got = c.get_object(oid)
                want = torch_view(ref[oid], (ref[oid].size_bytes,),
                                  torch.uint8).cpu().numpy().tobytes()
                assert got == want, f"{oid}: bytes differ"
            # data is external: init must NOT have materialized any
            # input-role object — absence is the contract
            with pytest.raises(ServiceError, match="tokens_0_0"):
                c.get_object("tokens_0_0")
            u = c.query_backing()
            assert u["used_bytes"] >= sum(
                s.size_bytes for s in prog.initial_objects
                if s.role != "input")
    finally:
        stop_server_thread(server, thread)


@pytest.mark.gpu
def test_real_boot_init_fits_tight_slab(tmp_path):
    """Init's slab footprint is exactly the model state: in a slab with
    only ~30% headroom over the initial objects, init_model succeeds
    and used_bytes lands at ~the model. The old output-task shape held
    every object twice at once (task transient + final-capture copy)
    and exhausted this slab."""
    pytest.importorskip("cuda.bindings.runtime")
    from dataflow_training.model_families.families import family
    from dataflow_training.register import register_all
    from dataflow_training.run.presets import cfg_dict as preset_cfg_dict

    register_all()
    fam = family("llama3")
    cfg = fam.config_type.tiny()
    cd = preset_cfg_dict(cfg)
    prog = fam.lower(cfg)
    need = sum(s.size_bytes for s in prog.initial_objects
               if s.role != "input")
    slab_gib = need * 1.3 / 2 ** 30      # NO floor: the tightness is the test
    sock, server, thread = _boot(tmp_path, "tight", fake=False,
                                 slab_backing_gib=slab_gib)
    try:
        from dataflow_training.run.driver import init_model

        with EngineClient(sock, client_name="tight") as c:
            created = init_model(c, "llama3", cd, seed=7)
            assert "W_0" in created
            u = c.query_backing()
            # ALIGN rounding pads each extent a little; 10% covers it
            assert need <= u["used_bytes"] <= need * 1.1
    finally:
        stop_server_thread(server, thread)
