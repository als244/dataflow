"""dsv32 adapter: DSA index selection (every layer computes its own —
no sharing) + MoE routing. Selection pinned by intercepting
dsa_topk_reference at the golden's import site; routing via the
route_ids kwarg. train_indexer=False: selections are data."""
from dataclasses import replace
from unittest.mock import patch

import torch

from dataflow.models.dsv32_reference import GoldenDsv32
from dataflow.tasks.layouts import dsv32_meta_layout
from dataflow.training.dsv32 import ShapedDsv32Config

name = "dsv32"


def make_cfg():
    return replace(ShapedDsv32Config.tiny(), train_indexer=False)


def make_golden(dims, n_layers, leaves):
    return GoldenDsv32.from_packed_bytes(dims, n_layers, *leaves)


def _refs():
    import dataflow.models.dsv32_reference as dref
    import dataflow.tasks.modules.moe.reference as mref

    return dref, mref


def capture(golden, tokens):
    dref, mref = _refs()
    real_topk, real_route = dref.dsa_topk_reference, mref.moe_topk_reference
    sels, routes = [], []

    def topk_rec(*a, **k):
        s = real_topk(*a, **k)
        sels.append(s.detach().to(torch.int32).cpu())
        return s

    def route_rec(*a, **k):
        w, ids = real_route(*a, **k)
        routes.append((w.detach().to(torch.bfloat16).cpu(),
                       ids.detach().to(torch.int32).cpu()))
        return w, ids

    captured = {"x": [], "sel": {}, "route_w": {}, "route_ids": {}}
    with patch.object(dref, "dsa_topk_reference", topk_rec), \
            patch.object(mref, "moe_topk_reference", route_rec):
        x = golden.w_embed["w"][tokens]
        for i, w in enumerate(golden.w_blocks):
            captured["x"].append(x.detach().clone())
            n_s, n_r = len(sels), len(routes)
            x, _ = golden.block_forward(x, w)
            if len(sels) > n_s:
                captured["sel"][i] = sels[-1]
            if len(routes) > n_r:
                captured["route_w"][i] = routes[-1][0]
                captured["route_ids"][i] = routes[-1][1]
    return captured, x.detach().clone()


def meta_layout(dims, i):
    return dsv32_meta_layout(dims, dims.kind_of(i))


def meta_fields(dims, i, captured):
    fields = {"dsa_idx": captured["sel"][i]}
    if i in captured["route_ids"]:
        from harness import routing_fields

        fields.update(routing_fields(dims, captured["route_ids"][i],
                                     captured["route_w"][i]))
    return fields


def pin(golden, captured):
    golden._saved = captured


def prep_layer(golden, i):
    pass


def block(golden, i, x):
    dref, _ = _refs()
    saved_sel = golden._saved["sel"][i].cuda()

    def pinned_topk(scores, topk):
        return saved_sel

    ids = golden._saved["route_ids"].get(i)
    ids = ids.cuda() if ids is not None else None
    golden._pending_counts = []
    with patch.object(dref, "dsa_topk_reference", pinned_topk):
        y, aux = golden.block_forward(x, golden.w_blocks[i], route_ids=ids)
    counts = None
    if ids is not None:
        counts = torch.bincount(ids.reshape(-1).long(),
                                minlength=golden.dims.moe.n_experts).float()
    return y, aux, counts


def adamw(golden, counts_of):
    d = golden.dims
    golden.step_count += 1
    golden._adamw_obj("embed", golden.w_embed)
    speed = d.moe.bias_update_speed
    for i, leaves in enumerate(golden.w_blocks):
        golden._adamw_obj(f"block_{i}", leaves)
        if "w_router_bias" in leaves and speed and i in counts_of:
            c = counts_of[i]
            b = leaves["w_router_bias"]
            b.data.add_(torch.sign(c.mean() - c).to(b.dtype), alpha=speed)
    golden._adamw_obj("head", golden.w_head)
