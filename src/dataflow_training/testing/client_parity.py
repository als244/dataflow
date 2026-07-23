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


def adamw_hyper_spec(hyper) -> dict:
    """JSON-able wire hyper encoding an ``AdamWHyper``'s optimizer step
    EXACTLY, so the daemon's optimizer applies the same update the
    in-process reference hyper does.

    The daemon rebuilds ``AdamWHyper(**spec)`` (``register.build_hyper``),
    so every field is a constructor kwarg; the ``schedule`` key is
    deliberately omitted, leaving the rebuilt schedule ``None`` (constant
    lr, scale 1.0 at every step). A scheduled hyper would scale lr by the
    step index and diverge from the reference's single fixed step — hence
    the guard."""
    if hyper.schedule is not None:
        raise ValueError(
            "adamw_hyper_spec requires schedule=None: a scheduled hyper "
            "scales lr per step, so the daemon's optimizer step would not "
            "match the reference optimizer's fixed step")
    return {"lr": hyper.lr, "beta1": hyper.beta1, "beta2": hyper.beta2,
            "eps": hyper.eps, "weight_decay": hyper.weight_decay,
            "momentum": hyper.momentum, "muon_lr": hyper.muon_lr}


class ClientFinalBytes:
    """get_bytes over a client run's post-step objects — the client-path
    twin of ``testing.gradcheck.EngineFinalBytes``.

    Feeds the family bridge's ``to_reference_state_dict`` for twin-space
    parameter comparison. W/O objects are persistent (backing-resident)
    initial objects the run mutates in place, so ``get_object`` returns
    their post-step bytes as a plain HOST copy; the bridge reads packed
    layouts on the device, so the copy is moved to cuda."""

    def __init__(self, client):
        self.client = client

    def __call__(self, object_id: str) -> torch.Tensor:
        return fetch_host_tensor(self.client, object_id, torch.uint8).cuda()


def client_aux_counts(client, dims, aux_ids):
    """Per-layer current-step expert counts, sourced via the client.

    Each ``Aux_{i}`` is a persistent (backing-resident) initial object the
    run mutates in place, so ``get_object`` returns its post-step bytes;
    the MoE aux layout unpacks the assignment histogram. Returns a list of
    cpu int64 count tensors aligned to ``aux_ids`` (layer order)."""
    if not aux_ids:
        return []
    from dataflow_training.blocks.modules.moe.spec import moe_aux_layout

    layout = moe_aux_layout(dims, dims.moe)
    out = []
    for oid in aux_ids:
        raw = fetch_host_tensor(client, oid, torch.uint8).cuda()
        counts = layout.unpack_tensor(raw)["expert_counts_current_step"]
        out.append(counts.long().cpu())
    return out


def stage_object(client, oid, data):
    """Stage a host object under ``oid``, replacing whatever is resident even
    if its size differs.

    ``put_object`` overwrites a same-size slot in place but rejects a size
    change (BINDING_MISMATCH) — and on a shared daemon a prior test may have
    left a differently-sized object under the same id (e.g. a batch-of-two
    packing after a full-length one). Releasing first drops any resident slot
    (a no-op when the id is absent) so the put always lands cleanly. Leases
    drop when a run completes, so a run input is free to re-stage by the next
    test."""
    client.release_object(oid, force=True)
    client.put_object(oid, data)


def client_step_fetch(client, *, cfg, seed, tokens_bytes, targets_bytes,
                      planned, resolver_spec, valid_rows, boundaries,
                      dims, fam, hyper, aux_ids):
    """One family step on an existing daemon client plus the reads the parity
    comparison needs.

    Re-seeds the model (init_model) so a shared client presents a pristine
    model, stages the tokens/targets, runs the single step, and returns the
    loss together with the engine's post-step params, gradients, and MoE
    counts — every value a HOST copy via the client. Split out of
    ``client_model_step`` so the step runs either on a freshly-spawned daemon
    or on a caller-supplied shared one."""
    from dataflow.core.jsonio import program_to_dict
    from dataflow_training.run.driver import init_model
    from dataflow_training.run.presets import cfg_dict, resolver_family

    init_model(client, resolver_family(cfg), cfg_dict(cfg), seed=seed)
    stage_object(client, "tokens_0_0", tokens_bytes)
    stage_object(client, "targets_0_0", targets_bytes)
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
    engine_state = bridges.to_reference_state_dict(cfg,
                                                   ClientFinalBytes(client))
    engine_grads = client_grad_state_dict(client, cfg, planned.program,
                                          fam.build_resolver(dims, hyper))
    engine_counts = client_aux_counts(client, dims, aux_ids)
    return (out["fetched"]["loss_0_0"], engine_state, engine_grads,
            engine_counts)


def client_model_step(cfg, *, seed: int = 0, tol: float = 3e-2,
                      field_atol: dict | None = None,
                      param_atol: dict | None = None,
                      reference_seq_lens: tuple | None = None,
                      reference_train_only: tuple | None = None,
                      optimizer: str = "adamw",
                      min_cosine: float = 0.995,
                      counts_budget: int | None = 3,
                      grad_tol: float | None = None,
                      backing_gib: float = 4.0, client=None):
    """Client-path analog of ``testing.gradcheck.check_model_step`` at full
    parity.

    Run one family step through the OUT-OF-PROCESS daemon and compare
    against the pure-torch twin, with every engine value read as a HOST
    copy via the client (no engine device view is ever held). The report
    is keyed EXACTLY like ``check_model_step`` — loss, final params
    (twin-name space, ``field_atol``/``param_atol`` raw-gap gated), one-step
    gradients (``grad:{name}``), and MoE assignment counts
    (``counts:{name}``) — so the verdict gates identically.

    The optimizer step is made to match ``check_model_step``'s engine leg
    (default ``AdamWHyper()``): the resolver spec carries a constant-lr
    hyper (no schedule) so the daemon applies ``lr`` unscaled at the single
    step, exactly as the reference twin does. Every persistent object
    (W/O/Aux) is read post-step through ``get_object`` (they are
    backing-resident and mutated in place); gradients (dW) are retained on
    the backing tier for the read.

    The reference twin's INIT weights are generated with a CudaBackend (the
    reference scaffolding); the engine leg — the thing under test — goes
    entirely through the client.

    ``client``: run the engine leg on this already-spawned daemon (re-seeded
    per call via init_model) instead of spawning a fresh one, so a family's
    tests can share one warm daemon. Defaults to spawning and tearing down a
    dedicated daemon.
    """
    from dataclasses import replace as dc_replace

    from dataflow.runtime.device.cuda import CudaBackend
    from dataflow_training.blocks.base_blocks import AdamWHyper
    from dataflow_training.lowering.planning import plan_program
    from dataflow_training.model_families.families import resolve_family
    from dataflow_training.run.presets import cfg_dict, resolver_family
    from dataflow_training.testing.client_daemon import out_of_process_daemon
    from dataflow_training.testing.gradcheck import (
        GRAD_TIER_LR, CheckReport, cos_sim, match_field_atol,
        reference_model_step, rel_l2)

    cfg = dc_replace(cfg, grad_accum_rounds=1)
    hyper = AdamWHyper()
    if optimizer == "sgd":
        cfg = dc_replace(cfg, opt_policy="sgd")
        hyper = AdamWHyper(lr=GRAD_TIER_LR)

    # the twin: DSA dense warm-up maps to a fully-selected sparse twin
    # (the engine leg keeps the true warm-up cfg); an MoE family whose
    # twin declares no load-balance form has the LBL channel zeroed on
    # both legs (symmetric) — mirror of check_model_step's twin build.
    twin_cfg = cfg
    if getattr(cfg, "sparse_mode", True) is False:
        twin_cfg = dc_replace(cfg, sparse_mode=True, index_topk=1 << 30)
    twin = bridges.build_reference_model(twin_cfg)
    if (float(getattr(cfg, "aux_coef", 0.0) or 0.0) > 0.0
            and getattr(twin, "AUX_FORM", None) is None):
        cfg = dc_replace(cfg, aux_coef=0.0)

    fam = resolve_family(cfg)
    dims = fam.derive_dims(cfg)
    if reference_seq_lens is None:
        reference_seq_lens = getattr(dims, "seq_lens", None)
    lengths = (tuple(int(x) for x in reference_seq_lens)
               if reference_seq_lens is not None
               else tuple([dims.seq_len] * (dims.max_tokens // dims.seq_len)))

    # reference twin from seed-S init weights (scaffolding backend)
    backend = CudaBackend()
    values = fam.initial_values(fam.lower(cfg), cfg, backend, seed=seed)
    get_bytes = bridges.get_bytes_from_values(values)
    tokens_bytes = get_bytes("tokens_0_0").cpu().numpy().tobytes()
    targets_bytes = get_bytes("targets_0_0").cpu().numpy().tobytes()
    twin_loss, twin, _twin_states, _init, twin_counts = reference_model_step(
        twin_cfg, values, seq_lens=reference_seq_lens,
        train_only=reference_train_only, optimizer=optimizer,
        model=twin, hyper=hyper)
    if getattr(cfg, "sparse_mode", True) is False:
        twin_loss = float(twin.indexer_loss().detach())
    twin_state = dict(twin.state_dict())
    twin_grads = {name: par.grad for name, par in twin.named_parameters()
                  if par.grad is not None}
    for buf in values.values():
        backend.free(buf)

    # retain the gradient backing: dW ids survive pool recycling after the
    # optimizer consumes them (W/O/Aux are already persistent → backing)
    program = fam.lower(cfg)
    dw_ids = {o.id for t in program.tasks for o in t.outputs
              if o.id.startswith("dW")}
    dw_ids.update(o.id for o in program.initial_objects
                  if o.id.startswith("dW"))
    program = with_backing_retention(program, sorted(dw_ids))
    planned = plan_program(program,
                           fast_memory_capacity=int(backing_gib * (1 << 30)))
    aux_ids = sorted((o.id for o in program.initial_objects
                      if o.id.startswith("Aux_")),
                     key=lambda oid: int(oid.split("_")[1]))

    targets = torch.frombuffer(bytearray(targets_bytes), dtype=torch.int32)
    valid_rows = int((targets >= 0).sum())
    resolver_spec = {"kind": "model_family", "family": resolver_family(cfg),
                     "cfg": cfg_dict(cfg), "hyper": adamw_hyper_spec(hyper)}
    boundaries = [0]
    for n in lengths:
        boundaries.append(boundaries[-1] + n)

    leg = functools.partial(
        client_step_fetch, cfg=cfg, seed=seed, tokens_bytes=tokens_bytes,
        targets_bytes=targets_bytes, planned=planned,
        resolver_spec=resolver_spec, valid_rows=valid_rows,
        boundaries=boundaries, dims=dims, fam=fam, hyper=hyper,
        aux_ids=aux_ids)
    if client is not None:
        run_loss, engine_state, engine_grads, engine_counts = leg(client)
    else:
        with out_of_process_daemon(backing_gib=backing_gib) as owned:
            run_loss, engine_state, engine_grads, engine_counts = leg(owned)

    errors: dict[str, float] = {}
    cosines: dict[str, float] = {}
    errors["loss"] = abs(run_loss - twin_loss) / max(abs(twin_loss), 1e-6)

    # final params in twin-name space: field_atol/param_atol raw-gap gate
    # the zero-init entries (one step is a per-element sign lottery where
    # the true gradient is near zero); everything else via rel_l2 + cosine
    for name, engine_tensor in engine_state.items():
        twin_tensor = twin_state.get(name)
        if twin_tensor is None:
            errors[name] = float("inf")
            continue
        atol = match_field_atol(name, field_atol)
        if atol is None:
            atol = match_field_atol(name, param_atol)
        if atol is not None:
            gap = float((engine_tensor.float().cpu()
                         - twin_tensor.float().cpu()).abs().max())
            errors[name] = 0.0 if gap <= atol else gap / atol
            continue
        errors[name] = rel_l2(engine_tensor, twin_tensor)
        cosines[name] = cos_sim(engine_tensor, twin_tensor)

    for name, g_engine in engine_grads.items():
        g_twin = twin_grads.get(name)
        if g_twin is None or g_twin.shape != g_engine.shape:
            continue        # frozen/train_only params carry no twin grad
        if match_field_atol(name, field_atol) is not None:
            continue        # enveloped fields: raw-gap gate on the param
        errors["grad:" + name] = rel_l2(g_engine, g_twin)
        cosines["grad:" + name] = cos_sim(g_engine, g_twin)

    if counts_budget is not None and aux_ids and twin_counts \
            and len(aux_ids) == len(twin_counts):
        # MoE counts parity: totals must equal tokens*top_k EXACTLY on both
        # sides; per-expert deltas bound the near-tie flipped tokens, gated
        # by the flip budget (sum|delta|/2 <= budget -> 0.0, else the flip
        # count itself, which no float tol admits)
        for (tname, tc), ec in zip(sorted(twin_counts.items()),
                                   engine_counts):
            tc = tc.long().cpu()
            if int(ec.sum()) != int(tc.sum()):
                errors["counts:" + tname] = float("inf")
                continue
            flips = int((ec - tc).abs().sum()) // 2
            errors["counts:" + tname] = (0.0 if flips <= counts_budget
                                         else float(flips))

    return CheckReport(errors=errors, tol=tol, cosines=cosines,
                       min_cosine=min_cosine,
                       tol_by_prefix=({"grad:": grad_tol}
                                      if grad_tol is not None else None))
