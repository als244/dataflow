"""qwen3moe adapter: MoE routing pinned via the golden's route_ids
kwarg; capture intercepts moe_topk_reference at the golden's import
site (no golden-body copies to drift)."""
from unittest.mock import patch

import torch

from dataflow.models.qwen3moe_reference import GoldenQwen3Moe
from dataflow.tasks.modules.moe.spec import moe_meta_layout
from dataflow.training.qwen3moe import ShapedQwen3MoeConfig

name = "qwen3moe"


def make_cfg():
    return ShapedQwen3MoeConfig.tiny()


def make_golden(dims, n_layers, leaves):
    return GoldenQwen3Moe.from_packed_bytes(dims, n_layers, *leaves)


def capture(golden, tokens):
    # qwen3moe delegates routing to moe_mlp_reference; intercept the
    # topk INSIDE the moe reference module
    import dataflow.tasks.modules.moe.reference as ref
    real = ref.moe_topk_reference
    rec = []

    def recorder(*a, **k):
        w, ids = real(*a, **k)
        rec.append((w.detach().to(torch.bfloat16).cpu(),
                    ids.detach().to(torch.int32).cpu()))
        return w, ids

    captured = {"x": [], "route_w": {}, "route_ids": {}}
    with patch.object(ref, "moe_topk_reference", recorder):
        x = golden.w_embed["w"][tokens]
        for i, w in enumerate(golden.w_blocks):
            captured["x"].append(x.detach().clone())
            n0 = len(rec)
            x, _ = golden.block_forward(x, w)
            if len(rec) > n0:
                captured["route_w"][i], captured["route_ids"][i] = \
                    rec[-1][0], rec[-1][1]
    return captured, x.detach().clone()


def meta_layout(dims, i):
    return moe_meta_layout(dims, dims.moe)


def meta_fields(dims, i, captured):
    if i not in captured["route_ids"]:
        return None
    from harness import routing_fields

    return routing_fields(dims, captured["route_ids"][i],
                          captured["route_w"][i])


def pin(golden, captured):
    golden._saved_route = captured["route_ids"]


def prep_layer(golden, i):
    pass


def block(golden, i, x):
    ids = golden._saved_route.get(i)
    ids = ids.cuda() if ids is not None else None
    y, aux = golden.block_forward(x, golden.w_blocks[i], route_ids=ids)
    counts = None
    if ids is not None:
        counts = torch.bincount(ids.reshape(-1).long(),
                                minlength=golden.dims.moe.n_experts).float()
    return y, aux, counts


def adamw(golden, counts_of):
    golden.step_count += 1
    golden._opt_obj("embed", golden.w_embed)
    for i, leaves in enumerate(golden.w_blocks):
        golden._opt_obj(f"block_{i}", leaves)
    golden._opt_obj("head", golden.w_head)
