"""Byte-equal sharded optimizer step (the rs/ag fast path).

The whole weight object is ONE flat element range (alignment gaps
included) cut into world byte-equal slices plus a world-remainder
tail: ONE reduce_scatter of the flattened grad buffer (in-place —
each rank's recv slice IS its window of the send buffer), a flat
adamw step on the owned slice with slice-sized m/v, the tail
allreduced and updated redundantly on every rank, then ONE
all_gather of the params. Elementwise sums and updates over
identically-reduced values: bitwise-identical to plain DP per weight
FIELD (the flat update rewrites the packing-gap bytes as unread
noise — identical across ranks via the gather, visible only to
whole-buffer diffs).

One flat update means ONE optimizer story for the whole object; the
plan builder guaranteed it (rs_eligible: uniform adamw at this
layer, no hyper_overrides match, uniform dtypes with param == grad)
and this module re-verifies at run time, refusing to train on policy
drift. With no group bound (warm-up), the comm skips and slice 0
updates from local grads — the same flat kernels compile at the same
shapes, results discarded at re-materialize.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view
from ..optim import OPTIMIZERS, resolve_hyper, resolve_opt_policy
from .update import field_key, run_step


def launch(block, ctx, es, kctx, wl, gl, ol, ns,
           w_buf, g_buf, o_buf, gh, sh) -> None:
    n_slice = int(sh["n_slice"])
    n_tail = int(sh["n_tail"])
    dt = TORCH_DTYPE_BY_NAME[sh["dtype"]]
    total = w_buf.size_bytes // dt.itemsize
    rank = gh.rank if gh is not None else 0
    if gh is not None and n_slice * gh.world + n_tail != total:
        raise RuntimeError(
            f"{ctx.task.id}: rs shard geometry {n_slice}*{gh.world}"
            f"+{n_tail} != packed element count {total} — plan/world "
            f"mismatch")
    n_main = total - n_tail
    g_flat = torch_view(g_buf, (total,), dt)
    w_flat = torch_view(w_buf, (total,), dt)
    own_g = g_flat[rank * n_slice:(rank + 1) * n_slice]
    if gh is not None:
        grads_ready = torch.cuda.Event()
        grads_ready.record(es)
        gh.stream.wait_event(grads_ready)
        gh.reduce_scatter(g_flat[:n_main], own_g)
        if n_tail:
            gh.allreduce(g_flat[n_main:])
        summed = torch.cuda.Event()
        summed.record(gh.stream)
        es.wait_event(summed)
    step = run_step(ctx)
    op = resolve_opt_policy(getattr(block.dims, "opt_policy", None))
    layer = block.parse_layer(ctx.task)
    first = wl.fields[0]
    opt = OPTIMIZERS[op.for_field(field_key(ns, first.name), layer,
                                  first.shape)]
    hp = resolve_hyper(op, field_key(ns, first.name), layer, block.hyper)
    # a deviation here is policy drift between plan time and run time
    # — refuse rather than train wrong
    if opt.name != "adamw":
        raise RuntimeError(
            f"{ctx.task.id}: rs shard is adamw-only, policy resolved "
            f"{opt.name!r}")
    for f in wl.fields[1:]:
        if op.for_field(field_key(ns, f.name), layer, f.shape) \
                != opt.name \
                or resolve_hyper(op, field_key(ns, f.name), layer,
                             block.hyper) != hp:
            raise RuntimeError(
                f"{ctx.task.id}: rs shard needs one optimizer story "
                f"for the whole root; field {f.name!r} deviates")
    sched = block.hyper.schedule
    if sched is not None and sched.scale(step) != 1.0:
        s_ = sched.scale(step)
        hp = replace(hp, lr=hp.lr * s_,
                     muon_lr=(hp.muon_lr * s_
                              if hp.muon_lr else hp.muon_lr))
    own_w = w_flat[rank * n_slice:(rank + 1) * n_slice]
    opt.step(kctx, block.kernels, hp, step, own_w, own_g,
             {"m": ol.view(o_buf, "m_slice").view(-1),
              "v": ol.view(o_buf, "v_slice").view(-1)},
             (n_slice,))
    if n_tail:
        opt.step(kctx, block.kernels, hp, step,
                 w_flat[n_main:], g_flat[n_main:],
                 {"m": ol.view(o_buf, "m_tail").view(-1),
                  "v": ol.view(o_buf, "v_tail").view(-1)},
                 (n_tail,))
    if gh is not None:
        w_ready = torch.cuda.Event()
        w_ready.record(es)
        gh.stream.wait_event(w_ready)
        gh.all_gather(own_w, w_flat[:n_main])
        gathered = torch.cuda.Event()
        gathered.record(gh.stream)
        es.wait_event(gathered)
