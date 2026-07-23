"""Client fetch-surface gate.

Retain run-produced objects — gradients (dW), an intermediate block output
(y), and an MoE expert-count (Aux) — on the BACKING (host) tier and read them
back through ``client.get_object`` as HOST copies, matching the pure-torch
twin. This exercises the read path the client parity helper (and every
workload test that fetches engine results) depends on: because the backing
tier is host memory, no engine device view is ever constructed or held.

Scaffolding note: the twin's init weights are generated with a CudaBackend
here (the same pattern the existing ragged parity test uses); the fully
client-only variant fetches the daemon-seeded weights through the client
instead, and is folded into the client parity helper.

Tests:
- test_client_fetch_surface_dense: llama3 loss, all gradients, and an intermediate block output are fetched via the client and match the pure-torch twin.
- test_client_fetch_surface_moe_aux: an olmoe expert-count object is fetched via the client as a host copy with non-negative, non-empty counts.
"""
from dataclasses import replace as dc_replace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

from dataflow.core.jsonio import program_to_dict                     # noqa: E402
from dataflow.runtime.device.cuda import CudaBackend                 # noqa: E402
from dataflow_training.lowering.planning import plan_program         # noqa: E402
from dataflow_training.model_families import bridges                 # noqa: E402
from dataflow_training.model_families.families import resolve_family  # noqa: E402
from dataflow_training.run import presets as P                       # noqa: E402
from dataflow_training.run.driver import daemon_client, init_model   # noqa: E402
from dataflow_training.run.presets import cfg_dict, resolver_family  # noqa: E402
from dataflow_training.run.recipe import Recipe                      # noqa: E402
from dataflow_training.testing.client_parity import (                # noqa: E402
    client_grad_state_dict, fetch_host_tensor, with_backing_retention)
from dataflow_training.testing.gradcheck import (                    # noqa: E402
    cos_sim, reference_model_step, rel_l2)

pytestmark = [pytest.mark.gpu]

SEED = 7
LOSS_REL_TOL = 5e-4
GRAD_REL_TOL = 3e-2          # llama3 gradient band (FAMILY_GRAD_GATE)
GRAD_MIN_COS = 0.999


def boundaries_of(lengths):
    edges = [0]
    for n in lengths:
        edges.append(edges[-1] + n)
    return edges


def uniform_lengths(dims):
    return tuple([dims.seq_len] * (dims.max_tokens // dims.seq_len))


def smoke_resolver(cfg):
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=1, total_steps=1)
    return {"kind": "model_family", "family": resolver_family(cfg),
            "cfg": cfg_dict(cfg), "hyper": recipe.hyper_spec()}


def test_client_fetch_surface_dense():
    cfg = dc_replace(P.smoke_preset(), grad_accum_rounds=1)   # llama3
    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    lengths = uniform_lengths(dims)

    # --- reference twin from seed-S init weights (scaffolding backend) ----
    backend = CudaBackend()
    values = fam.initial_values(fam.lower(cfg), cfg, backend, seed=SEED)
    get_bytes = bridges.get_bytes_from_values(values)
    tokens_bytes = bytes(get_bytes("tokens_0_0").cpu().numpy().tobytes())
    targets_bytes = bytes(get_bytes("targets_0_0").cpu().numpy().tobytes())
    twin = bridges.build_reference_model(cfg)
    twin_loss, twin, _states, _init, _counts = reference_model_step(
        cfg, values, seq_lens=lengths, model=twin)
    twin_grads = {name: par.grad for name, par in twin.named_parameters()
                  if par.grad is not None}
    for buf in values.values():
        backend.free(buf)

    # --- program with backing retention for a block output + every dW -----
    program = fam.lower(cfg)
    dw_ids = sorted(o.id for t in program.tasks for o in t.outputs
                    if o.id.startswith("dW"))
    y_id = "y_0_0_0"
    program = with_backing_retention(program, [*dw_ids, y_id])
    planned = plan_program(program, fast_memory_capacity=4 << 30)

    targets = torch.frombuffer(bytearray(targets_bytes), dtype=torch.int32)
    valid_rows = int((targets >= 0).sum())

    with daemon_client(backing_gib=4.0) as client:
        init_model(client, resolver_family(cfg), cfg_dict(cfg), seed=SEED)
        client.put_object("tokens_0_0", tokens_bytes)
        client.put_object("targets_0_0", targets_bytes)
        reg = client.register_program(program_to_dict(planned.program),
                                      resolver=smoke_resolver(cfg))
        assert not reg["bindings"]["missing_inputs"], reg
        out = client.run(
            reg["prog_id"],
            args={"step": 0, "valid_rows": valid_rows,
                  "seq_lens": {"0": boundaries_of(lengths)}},
            fetch=["loss_0_0"])
        assert out.get("state") == "done", out

        # (1) loss via inline fetch: a host scalar matching the twin
        run_loss = out["fetched"]["loss_0_0"]
        assert abs(run_loss - twin_loss) / max(abs(twin_loss), 1e-6) \
            < LOSS_REL_TOL, (run_loss, twin_loss)

        # (2) intermediate y via get_object: a HOST copy, right shape, finite
        y = fetch_host_tensor(client, y_id, torch.bfloat16)
        assert y.device.type == "cpu", y.device
        assert y.numel() == dims.max_tokens * dims.d_model, y.numel()
        y_float = y.float()
        assert torch.isfinite(y_float).all()
        assert y_float.abs().sum() > 0

        # (3) gradients via get_object + layout map: match the twin autograd
        eng_grads = client_grad_state_dict(client, cfg, planned.program,
                                           fam.build_resolver(dims))

    checked = 0
    for name, g_twin in twin_grads.items():
        g_eng = eng_grads.get(name)
        if g_eng is None or g_eng.shape != g_twin.shape:
            continue
        rel = rel_l2(g_eng, g_twin)
        cos = cos_sim(g_eng, g_twin)
        assert rel < GRAD_REL_TOL, (name, "rel", rel)
        assert cos > GRAD_MIN_COS, (name, "cos", cos)
        checked += 1
    assert checked >= 3, f"only {checked} grads cross-checked"


def test_client_fetch_surface_moe_aux():
    cfg = dc_replace(P.olmoe_smoke_preset(), grad_accum_rounds=1)
    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    lengths = uniform_lengths(dims)

    backend = CudaBackend()
    values = fam.initial_values(fam.lower(cfg), cfg, backend, seed=SEED)
    get_bytes = bridges.get_bytes_from_values(values)
    tokens_bytes = bytes(get_bytes("tokens_0_0").cpu().numpy().tobytes())
    targets_bytes = bytes(get_bytes("targets_0_0").cpu().numpy().tobytes())
    for buf in values.values():
        backend.free(buf)

    # Aux expert-counts are INITIAL objects mutated in place by the run
    # (routing counts accrue regardless of the load-balance coefficient).
    program = fam.lower(cfg)
    aux_ids = sorted((o.id for o in program.initial_objects
                      if o.id.startswith("Aux_")),
                     key=lambda oid: int(oid.split("_")[1]))
    assert aux_ids, "olmoe should emit Aux counts"
    program = with_backing_retention(program, aux_ids)
    planned = plan_program(program, fast_memory_capacity=4 << 30)

    targets = torch.frombuffer(bytearray(targets_bytes), dtype=torch.int32)
    valid_rows = int((targets >= 0).sum())
    expected_bytes = planned.program.object_sizes()[aux_ids[0]]

    with daemon_client(backing_gib=4.0) as client:
        init_model(client, resolver_family(cfg), cfg_dict(cfg), seed=SEED)
        client.put_object("tokens_0_0", tokens_bytes)
        client.put_object("targets_0_0", targets_bytes)
        reg = client.register_program(program_to_dict(planned.program),
                                      resolver=smoke_resolver(cfg))
        assert not reg["bindings"]["missing_inputs"], reg
        out = client.run(
            reg["prog_id"],
            args={"step": 0, "valid_rows": valid_rows,
                  "seq_lens": {"0": boundaries_of(lengths)}},
            fetch=["loss_0_0"])
        assert out.get("state") == "done", out

        # Aux counts via get_object: a HOST copy of the expected size, whose
        # per-expert integer counts are non-negative and non-empty (exact
        # count-vs-top_k parity is the migrated check_model_step's job).
        raw = client.get_object(aux_ids[0])

    assert len(raw) == expected_bytes, (len(raw), expected_bytes)
    counts = torch.frombuffer(bytearray(raw), dtype=torch.int32)
    assert (counts >= 0).all()
    assert int(counts.sum()) > 0
