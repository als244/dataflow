"""Client-path parity reads.

Fetch engine-produced objects (gradients, block outputs, losses, MoE counts)
as HOST copies through the daemon client and map them into twin-named
comparison tensors — the same math the in-process ``gradcheck`` reads perform,
but sourced via ``client.get_object`` against objects retained on the backing
(host) tier. Because the backing tier IS host memory, the fetch is a plain host
copy: no engine device view is ever constructed or held, so this whole read
half is memory-safe by construction (the workload-test client contract).

Retention rule: an object is only capturable by ``get_object`` if the program
pins it to ``"backing"`` in ``final_locations`` (the engine copies the backing
slot into the store at run end). The in-process reads use ``"fast"`` and view
the device slot directly — that path is exactly what this module replaces.
"""
import dataclasses as dc
import functools

import torch

from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME
from dataflow_training.model_families import bridges


def with_backing_retention(program, ids):
    """Return a copy of ``program`` whose ``final_locations`` pins every id in
    ``ids`` to the backing (host) tier, so the run retains it and the engine
    captures it into the store for ``client.get_object`` to return."""
    return dc.replace(
        program,
        final_locations={**dict(program.final_locations),
                         **{oid: "backing" for oid in ids}})


def fetch_host_tensor(client, oid, dtype):
    """``client.get_object(oid)`` -> a 1-D host tensor of ``dtype`` over the
    returned bytes. The object must have been retained on the backing tier."""
    raw = client.get_object(oid)
    return torch.frombuffer(bytearray(raw), dtype=dtype)


def grad_shim(object_id, fabricated, weight_sizes):
    """Bridge name-map shim: hand ``to_reference_state_dict`` a weight-layout
    buffer whose field slots already hold the gradient values. A weight with no
    optimizer task (fully frozen) has no gradient storage, so fabricate a
    NaN-poisoned blank of the right size — every field derived from it reads
    NaN and is stripped by the caller."""
    made = fabricated.get(object_id)
    if made is not None:
        return made
    n = weight_sizes[object_id]
    blank = torch.zeros(n, dtype=torch.uint8, device="cuda")
    blank[: n - n % 2].view(torch.bfloat16).fill_(float("nan"))
    fabricated[object_id] = blank
    return blank


def client_grad_state_dict(client, cfg, program, resolver):
    """Engine dW backing -> twin-named gradient dict, sourced via the client.

    Mirror of ``testing.gradcheck.engine_grad_state_dict``: each optimizer
    task's executable resolves its own (weight_layout, grad_layout); the dW
    backing is read as HOST bytes with ``client.get_object`` (instead of an
    in-process device view), its gradient fields are packed into a fabricated
    weight-layout buffer, and the family bridge's ``to_reference_state_dict``
    supplies the twin name map. Requires the run to have retained every dW on
    the backing tier (see ``with_backing_retention``).
    """
    weight_sizes = program.object_sizes()
    fabricated: dict[str, torch.Tensor] = {}
    for task in program.tasks:
        if not task.id.startswith("optimizer_"):
            continue
        w_id = next(i for i in task.inputs if i.startswith("W_"))
        dw_id = next(i for i in task.inputs if i.startswith("dW"))
        executable = resolver(task)
        weight_layout, grad_layout = executable._layouts(
            task, weight_sizes[w_id])[:2]
        raw = fetch_host_tensor(client, dw_id, torch.uint8).cuda()
        grads = grad_layout.unpack_tensor(raw)
        grad_dtypes = {f.name: f.dtype for f in grad_layout.fields}
        fake = torch.empty(weight_layout.total_bytes, dtype=torch.uint8,
                           device="cuda")
        views = weight_layout.unpack_tensor(fake)
        for field in weight_layout.fields:
            if field.name in grads:
                if grad_dtypes[field.name] != field.dtype:
                    raise AssertionError(
                        f"{dw_id}:{field.name} grad dtype "
                        f"{grad_dtypes[field.name]} != weight dtype "
                        f"{field.dtype} — the fabricated-buffer shim would "
                        f"quantize")
                views[field.name].copy_(grads[field.name])
            else:
                views[field.name].fill_(
                    float("nan")
                    if TORCH_DTYPE_BY_NAME[field.dtype].is_floating_point
                    else 0)
        fabricated[w_id] = fake

    shim = functools.partial(grad_shim, fabricated=fabricated,
                             weight_sizes=weight_sizes)
    out = {}
    for name, tensor in bridges.to_reference_state_dict(cfg, shim).items():
        if tensor.float().isnan().all():
            continue                       # gradient-free field (frozen)
        out[name] = tensor
    return out


def client_model_step(cfg, *, seed: int = 0, grad_tol: float = 3e-2,
                      min_cosine: float = 0.99, loss_tol: float = 5e-4,
                      backing_gib: float = 4.0):
    """Client-path analog of ``testing.gradcheck.check_model_step``.

    Run one family step through the OUT-OF-PROCESS daemon and compare the loss
    and one-step gradients against the pure-torch twin, with every engine value
    read as a HOST copy via the client (no engine device view is ever held).
    Returns a ``CheckReport`` keyed like check_model_step (``loss`` +
    ``grad:{name}``) so the verdict gates identically.

    The reference twin's INIT weights are generated with a CudaBackend (the same
    reference-scaffolding the existing client parity test uses); the engine leg
    — the thing under test — goes entirely through the client.
    """
    from dataclasses import replace as dc_replace

    from dataflow.core.jsonio import program_to_dict
    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.run.driver import daemon_client, init_model
    from dataflow_training.run.presets import cfg_dict, resolver_family
    from dataflow_training.run.recipe import Recipe
    from dataflow_training.testing.client_daemon import out_of_process_daemon
    from dataflow_training.testing.gradcheck import (
        CheckReport, cos_sim, reference_model_step, rel_l2)

    cfg = dc_replace(cfg, grad_accum_rounds=1)
    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    lengths = tuple([dims.seq_len] * (dims.max_tokens // dims.seq_len))

    # reference twin from seed-S init weights (scaffolding backend)
    backend = CudaBackend()
    values = fam.initial_values(fam.lower(cfg), cfg, backend, seed=seed)
    get_bytes = bridges.get_bytes_from_values(values)
    tokens_bytes = get_bytes("tokens_0_0").cpu().numpy().tobytes()
    targets_bytes = get_bytes("targets_0_0").cpu().numpy().tobytes()
    twin = bridges.build_reference_model(cfg)
    twin_loss, twin, _states, _init, _counts = reference_model_step(
        cfg, values, seq_lens=lengths, model=twin)
    twin_grads = {name: par.grad for name, par in twin.named_parameters()
                  if par.grad is not None}
    for buf in values.values():
        backend.free(buf)

    program = fam.lower(cfg)
    dw_ids = sorted(o.id for t in program.tasks for o in t.outputs
                    if o.id.startswith("dW"))
    program = with_backing_retention(program, dw_ids)
    planned = plan_program(program,
                           fast_memory_capacity=int(backing_gib * (1 << 30)))

    targets = torch.frombuffer(bytearray(targets_bytes), dtype=torch.int32)
    valid_rows = int((targets >= 0).sum())
    recipe = Recipe(peak_lr=3e-4, min_lr=3e-5, warmup_steps=1, total_steps=1)
    resolver_spec = {"kind": "model_family", "family": resolver_family(cfg),
                     "cfg": cfg_dict(cfg), "hyper": recipe.hyper_spec()}
    boundaries = [0]
    for n in lengths:
        boundaries.append(boundaries[-1] + n)

    with out_of_process_daemon(backing_gib=backing_gib) as client:
        init_model(client, resolver_family(cfg), cfg_dict(cfg), seed=seed)
        client.put_object("tokens_0_0", tokens_bytes)
        client.put_object("targets_0_0", targets_bytes)
        reg = client.register_program(program_to_dict(planned.program),
                                      resolver=resolver_spec)
        if reg["bindings"]["missing_inputs"]:
            raise RuntimeError(f"unbound inputs: {reg['bindings']}")
        out = client.run(reg["prog_id"],
                         args={"step": 0, "valid_rows": valid_rows,
                               "seq_lens": {"0": boundaries}},
                         fetch=["loss_0_0"])
        if out.get("state") != "done":
            raise RuntimeError(f"run state {out.get('state')}: {out}")
        run_loss = out["fetched"]["loss_0_0"]
        engine_grads = client_grad_state_dict(client, cfg, planned.program,
                                              fam.build_resolver(dims))

    errors = {"loss": abs(run_loss - twin_loss) / max(abs(twin_loss), 1e-6)}
    cosines: dict[str, float] = {}
    for name, g_twin in twin_grads.items():
        g_engine = engine_grads.get(name)
        if g_engine is None or g_engine.shape != g_twin.shape:
            continue
        errors["grad:" + name] = rel_l2(g_engine, g_twin)
        cosines["grad:" + name] = cos_sim(g_engine, g_twin)
    return CheckReport(errors=errors, tol=loss_tol, cosines=cosines,
                       min_cosine=min_cosine, tol_by_prefix={"grad:": grad_tol})
