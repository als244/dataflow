"""The per-field optimizer update core shared by every comm variant.

One packed weight object, one pass over its layout fields: each field
resolves its rule (adamw/sgd/sgdm/muon/frozen/custom) and effective
hyperparameters through the config's optimizer policy, then steps
through its own w/g/state views. Communication is the caller's job —
this module never touches a group handle, which is exactly what makes
the same update correct standalone (warm-up), replicated (plain DP),
or narrowed to owned regions (sharded plans).
"""
from __future__ import annotations

from dataclasses import replace

from ..optim import OPTIMIZERS, hyper_for, resolve_opt_policy


def field_key(ns: str | None, name: str) -> str:
    """The policy-lookup key for a field: embed/head tables pack
    fields literally named "w", so their policies address them
    namespaced ("embed.w", "head.final_norm_w"); block fields are
    addressed bare."""
    return f"{ns}.{name}" if ns else name


def run_step(ctx) -> int:
    """The 1-based global step: service runs bind it per run
    (run_args); in-process paths keep baking it into block_params."""
    ra = getattr(ctx, "run_args", None) or {}
    return int(ra.get("step", ctx.task.block_params.get("step", 0))) + 1


def update_fields(block, ctx, kctx, wl, gl, ol, ns,
                  w_buf, g_buf, o_buf, update_regions=None) -> None:
    """Step every field of the object (or, with ``update_regions``,
    only the fields/rows this rank owns — absent fields' bytes arrive
    by the caller's post-update broadcast)."""
    step = run_step(ctx)
    op = resolve_opt_policy(getattr(block.dims, "opt_policy", None))
    layer = block.layer_of(ctx.task)
    # lr schedule: pure function of the step index; scales lr AND
    # muon_lr, applied AFTER per-field hyper overrides
    sched = block.hyper.schedule
    sched_scale = sched.scale(step) if sched is not None else 1.0
    for f in wl.fields:
        if block.update_specials is not None \
                and f.name in block.update_specials:
            # highest-priority per-field override (noaux bias rule,
            # frozen fields) — skips policy AND state
            block.update_specials[f.name](
                kctx, block.kernels,
                wl.view(w_buf, f.name).view(-1),
                gl.view(g_buf, f.name).view(-1),
            )
            continue
        if update_regions is not None and f.name not in update_regions:
            continue            # another rank's region: no state
                                # here; its bytes arrive by broadcast
        rows = (update_regions.get(f.name)
                if update_regions is not None else None)
        opt = OPTIMIZERS[op.for_field(field_key(ns, f.name), layer,
                                      f.shape)]
        if opt.name == "frozen":
            continue            # frozen: no grad storage, no update
        hp = hyper_for(op, field_key(ns, f.name), layer, block.hyper)
        if sched_scale != 1.0:
            hp = replace(hp, lr=hp.lr * sched_scale,
                         muon_lr=(hp.muon_lr * sched_scale
                                  if hp.muon_lr else hp.muon_lr))
        if opt.slots and o_buf is None:
            raise ValueError(
                f"{ctx.task.id}: field {f.name!r} wants {opt.name!r} "
                f"state but the task has no O object")
        states = {slot: ol.view(o_buf, f"{slot}_{f.name}").view(-1)
                  for slot in opt.slots}
        w_v = wl.view(w_buf, f.name)
        g_v = gl.view(g_buf, f.name)
        shape = f.shape
        if rows is not None:
            lo, hi = rows
            w_v = w_v[lo:hi]        # dim-0 slice of a packed
            g_v = g_v[lo:hi]        # field view: contiguous
            shape = (hi - lo,) + tuple(f.shape[1:])
        opt.step(kctx, block.kernels, hp, step,
                 w_v.reshape(-1), g_v.reshape(-1),
                 states, shape)
