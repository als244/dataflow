"""Family-generic fleet gate: the REAL grouped-lowering path
(fleet.lower_with_group = blind family lower -> annotate_groups ->
exact sizes) driving a two-daemon DP step for a NON-llama3 family.

Same shape as the single-family DP-step gate, but everything
family-parameterized through the registry — lowering, resolver, init,
layouts. Asserts per family: (1) the two replicas' weights land BITWISE
IDENTICAL, (2) the result matches a single-engine combined-batch run
within bf16 accumulation noise. llama3 runs as the control; gpt2 was
the first non-llama3 passenger; qwen35 (hybrid linear attention) and
qwen3moe (routed experts + per-step load-balancing aux) put the two
structurally different architectures through the same machinery.

Tests:
- test_two_daemon_dp_step_replicas_bitwise_and_match_single_engine: per family (llama3, gpt2, qwen35, qwen3moe), a two-daemon DP step through the family-generic grouped lowering ends with bitwise-identical replicas that match the single-engine combined-batch run within rel_l2 2e-2.
"""
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)
pytest.importorskip("cuda.bindings")

from dataclasses import replace  # noqa: E402

from dataflow.core.jsonio import program_to_dict  # noqa: E402
from dataflow.runtime import Engine  # noqa: E402
from dataflow.runtime.device.cuda import CudaBackend  # noqa: E402
from dataflow.runtime.device.fake import FakeBackend  # noqa: E402
from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view  # noqa: E402
from dataflow.service import EngineClient, EngineConfig, Server  # noqa: E402
from dataflow_training.data.segments import uniform_segments  # noqa: E402
from dataflow_training.distributed.fleet import lower_with_group  # noqa: E402
from dataflow_training.lowering.planning import plan_program  # noqa: E402
from dataflow_training.model_families.families import resolve_family  # noqa: E402
from dataflow_training.register import register_all  # noqa: E402
from dataflow_training.run.driver import init_model  # noqa: E402
from dataflow_training.run.presets import cfg_dict, resolver_family  # noqa: E402
from dataflow_training.testing.gradcheck import rel_l2  # noqa: E402

pytestmark = pytest.mark.fleet

SEED = 11
# per-family port pairs: back-to-back params must not fight over
# lingering peer listeners (Address-already-in-use, suite-order flake)
PORTS = {"llama3": (29531, 29532), "gpt2": (29541, 29542),
         "qwen35": (29551, 29552), "qwen3moe": (29561, 29562)}


def tiny_pair_cfg(family_tiny):
    """The family's tiny config as one DP rank's step (ga=1); the
    master batch is two of these."""
    return replace(family_tiny, grad_accum_rounds=1)


def master_tokens(cfg, t_step):
    g = torch.Generator().manual_seed(99)
    tok = torch.randint(0, cfg.vocab_size, (t_step,), generator=g,
                        dtype=torch.int32)
    tgt = torch.randint(0, cfg.vocab_size, (t_step,), generator=g,
                        dtype=torch.int32)
    return tok, tgt


def single_box_reference(cfg_rank, t_step):
    """Direct-engine ga=2 over the SAME tokens (halves = the DP
    split), same global denominator: {W_root: {field: cpu tensor}}."""
    cfg = replace(cfg_rank, grad_accum_rounds=2)
    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    planned = plan_program(fam.lower(cfg),
                           fast_memory_capacity=96 * 1024 * 1024)
    backend = CudaBackend()
    values = fam.initial_values(planned.program, cfg, backend, seed=SEED)
    tok, tgt = master_tokens(cfg, t_step)
    per = t_step // 2
    for r in range(2):
        torch_view(values[f"tokens_0_{r}"], (per,),
                   torch.int32).copy_(tok[r * per:(r + 1) * per])
        torch_view(values[f"targets_0_{r}"], (per,),
                   torch.int32).copy_(tgt[r * per:(r + 1) * per])
    dry = Engine(FakeBackend()).execute(planned.program,
                                        initial_buffers=values)
    result = Engine(backend).execute(
        planned.program, resolver=fam.build_resolver(dims),
        initial_buffers=values, pool_prewarm=dry.pool_demand,
        run_args={"segments": uniform_segments(dims, planned.program),
                  "step": 0, "valid_rows": t_step})
    _, fl = fam.family_layouts(cfg)
    out = {}
    for i, layer in enumerate(fl.layers):
        rec = result.objects.get(f"W_{i}")
        slot = rec.backing or rec.fast
        out[f"W_{i}"] = {
            f.name: torch_view(slot.buffer, f.shape,
                               TORCH_DTYPE_BY_NAME[f.dtype],
                               offset_bytes=f.offset_bytes).clone().cpu()
            for f in layer.weights.fields}
    result.close()
    return out


def boot(tmp, name, peer_port):
    sock = str(tmp / f"{name}.sock")
    server = Server(EngineConfig(
        socket_path=sock, fake=False, slab_backing_gib=0.4,
        peer_name=name, peer_listen=f"127.0.0.1:{peer_port}"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for _ in range(600):
        try:
            EngineClient(sock, client_name="probe").close()
            break
        except OSError:
            time.sleep(0.01)
    return server, EngineClient(sock, client_name=name)


class RankRun:
    def __init__(self, client, prog_id, valid):
        self.client = client
        self.prog_id = prog_id
        self.valid = valid
        self.out = None
        self.err = None

    def __call__(self):
        try:
            self.out = self.client.run(
                self.prog_id, args={"step": 0, "valid_rows": self.valid},
                fetch=["loss_0_0"])
        except Exception as e:
            self.err = e


@pytest.mark.gpu
@pytest.mark.parametrize("family_name", ["llama3", "gpt2",
                                         "qwen35", "qwen3moe"])
def test_two_daemon_dp_step_replicas_bitwise_and_match_single_engine(
        tmp_path, family_name):
    register_all()
    from dataflow_training.model_families.families import _FAMILIES

    fam = _FAMILIES[family_name]()
    cfg = tiny_pair_cfg(fam.config_type.tiny())
    t_step = 2 * cfg.max_tokens
    planned = plan_program(lower_with_group(cfg, "dp"),
                           fast_memory_capacity=96 * 1024 * 1024)
    prog_dict = program_to_dict(planned.program)
    resolver = {"kind": "model_family", "family": resolver_family(cfg),
                "cfg": cfg_dict(cfg)}
    tok, tgt = master_tokens(cfg, t_step)

    pa, pb = PORTS[family_name]
    sa, ca = boot(tmp_path, f"{family_name}-a", pa)
    sb, cb = boot(tmp_path, f"{family_name}-b", pb)
    try:
        ca.peer_connect(f"{family_name}-b", f"127.0.0.1:{pb}")
        per = t_step // 2

        def seed_rank(rank, client):
            init_model(client, resolver_family(cfg), cfg_dict(cfg),
                       seed=SEED)
            lo = rank * per
            client.put_object("tokens_0_0",
                              tok[lo:lo + per].numpy().tobytes())
            client.put_object("targets_0_0",
                              tgt[lo:lo + per].numpy().tobytes())

        for rank, client in ((0, ca), (1, cb)):
            seed_rank(rank, client)
            reg = client.register_program(prog_dict, resolver=resolver)
            assert not reg["bindings"]["missing_inputs"]
        prog_id = reg["prog_id"]
        # warm-up before the group exists, then re-seed
        for rank, client in ((0, ca), (1, cb)):
            warm = client.run(prog_id,
                              args={"step": 0, "valid_rows": t_step})
            assert warm.get("state") == "done", warm
            seed_rank(rank, client)
        ca._call("create_peer_group",
                 {"name": "dp",
                  "members": [f"{family_name}-a", f"{family_name}-b"],
                  "backend": "hostmem"})

        runs = [RankRun(ca, prog_id, t_step), RankRun(cb, prog_id, t_step)]
        threads = [threading.Thread(target=r) for r in runs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(180)
        for r in runs:
            assert r.err is None, r.err
            assert r.out.get("state") == "done", r.out

        # replicas bitwise identical
        for i in range(cfg.n_layers):
            assert bytes(ca.get_object(f"W_{i}")) == \
                bytes(cb.get_object(f"W_{i}")), f"divergence at W_{i}"
        assert bytes(ca.get_object("W_embed")) == \
            bytes(cb.get_object("W_embed"))

        # equivalence vs the single-engine combined-batch run
        ref = single_box_reference(cfg, t_step)
        _, fl = fam.family_layouts(cfg)
        worst = 0.0
        for i, layer in enumerate(fl.layers):
            raw = bytearray(bytes(ca.get_object(f"W_{i}")))
            for f in layer.weights.fields:
                numel = 1
                for d in f.shape:
                    numel *= d
                got = torch.frombuffer(
                    raw, dtype=TORCH_DTYPE_BY_NAME[f.dtype], count=numel,
                    offset=f.offset_bytes).reshape(f.shape)
                worst = max(worst,
                            rel_l2(got.float(), ref[f"W_{i}"][f.name].float()))
        assert worst < 2e-2, worst
    finally:
        for c in (ca, cb):
            try:
                c.shutdown()
            except Exception:
                pass
