"""Field-snapped sharded optimizer step (ZeRO-1 plans and the
replicated fields of tensor-parallel plans).

The task's shard block_param is pure data: {"update": {field: rows |
null} — the regions THIS RANK owns state for; "comm": [{field, rows,
owner}] — the plan-ordered exchange list; "grads": "partial" |
"replica"}. Three phases:

1. pre-update comm (partial grads only — a data split ran): each
   sharded region reduces to its owner, replicated fields keep the
   redundant-update allreduce. "replica" grads (tensor parallelism:
   no data split) skip this — every rank's grad for a replicated
   field is already the complete gradient, a sum-reduce would double
   it; resident-narrowed fields have no comm entries at all.
2. the shared per-field update, narrowed to owned regions;
3. updated params broadcast from their owners. The final es wait
   keeps the W mutation inside THIS task's stream order — the
   planner is free to move/offload W right after the task completes.

With no group bound (warm-up), comm skips and the owned regions
update from local grads; results are discarded at re-materialize.
"""
from __future__ import annotations

import torch

from .update import update_fields, wait_group_tail


def launch(block, ctx, es, kctx, wl, gl, ol, ns,
           w_buf, g_buf, o_buf, gh, gw, sh) -> None:
    if gh is not None and sh.get("grads", "partial") == "partial":
        grads_ready = torch.cuda.Event()
        grads_ready.record(es)
        gh.stream.wait_event(grads_ready)
        sharded_names = {e["field"] for e in sh["comm"]}
        for e in sh["comm"]:
            gview = gl.view(g_buf, e["field"])
            if e["rows"] is not None:
                lo, hi = int(e["rows"][0]), int(e["rows"][1])
                gview = gview[lo:hi]
            gh.reduce(gview, root=int(e["owner"]))
        for f in gl.fields:
            if f.name in sharded_names:
                continue
            gh.allreduce(gl.view(g_buf, f.name))
        summed = torch.cuda.Event()
        summed.record(gh.stream)
        es.wait_event(summed)
    if gw is not None:
        wait_group_tail(es, gw)
    regions = {name: (tuple(rows) if rows else None)
               for name, rows in sh["update"].items()}
    update_fields(block, ctx, kctx, wl, gl, ol, ns,
                  w_buf, g_buf, o_buf, update_regions=regions)
    if gh is not None and sh["comm"]:
        w_ready = torch.cuda.Event()
        w_ready.record(es)
        gh.stream.wait_event(w_ready)
        for e in sh["comm"]:
            wview = wl.view(w_buf, e["field"])
            if e["rows"] is not None:
                lo, hi = int(e["rows"][0]), int(e["rows"][1])
                wview = wview[lo:hi]
            gh.broadcast(wview, root=int(e["owner"]))
        propagated = torch.cuda.Event()
        propagated.record(gh.stream)
        es.wait_event(propagated)
