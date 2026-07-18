"""glm52 adapter: the richest case — IndexShare selections shared
across layer groups (leader M consumed by followers) + MoE routing.
train_indexer=False is load-bearing: selections are DATA."""
from dataclasses import replace

import torch

from dataflow_training.blocks.modules.dsa_forms import dsa_mask_from_idx
from dataflow_training.blocks.layouts import glm52_aux_temp_layout
from dataflow_training.model_families.glm52 import ShapedGlm52Config

from pinned_golden import RlGlm52

name = "glm52"


def make_cfg():
    # routing is PINNED in the RL weave: the noaux balance rule is inert
    # (and the engine's per-step counts come from the forwards this
    # program deliberately lacks) — freeze it on both sides
    return replace(ShapedGlm52Config.tiny(), train_indexer=False,
                   bias_update_speed=0.0)


def make_golden(dims, n_layers, leaves):
    g = RlGlm52.from_packed_bytes(dims, n_layers, *leaves)
    return g


def capture(golden, tokens):
    golden.capture = True
    golden.reset_capture()
    golden._layer_ptr = 0
    golden._group_scores = golden._group_mask = None
    x = golden.w_embed["w"][tokens]
    for w in golden.w_blocks:
        golden.captured["x"].append(x.detach().clone())
        x, _ = golden.block_forward(x, w)
    cap = golden.captured
    # block_forward already recorded sel/route; x list recorded here so
    # disable its internal x-capture double-record
    cap["x"] = cap["x"][:len(golden.w_blocks)]
    golden.capture = False
    return cap, x.detach().clone()


def meta_layout(dims, i):
    return glm52_aux_temp_layout(dims, dims.kinds[i])


def meta_fields(dims, i, captured):
    kind = dims.kinds[i]
    fields = {}
    if kind in ("gdl", "gml"):
        fields["dsa_idx"] = captured["sel"][i]
    if kind in ("gml", "gmf"):
        from harness import routing_fields

        fields.update(routing_fields(dims, captured["route_ids"][i],
                                     captured["route_w"][i]))
    return fields


def pin(golden, captured):
    golden.saved = {"sel": captured["sel"],
                    "route_ids": captured["route_ids"]}


def prep_layer(golden, i):
    d = golden.dims
    golden._layer_ptr = i
    lead = d.leader_index(i)
    golden._group_mask = dsa_mask_from_idx(
        golden.saved["sel"][lead].cuda(), d, d.tokens)
    golden._group_scores = None


def block(golden, i, x):
    golden._pending_counts = []
    golden._layer_ptr = i
    y, aux = golden.block_forward(x, golden.w_blocks[i])
    counts = golden._pending_counts[-1] if golden._pending_counts else None
    return y, aux, counts


def adamw(golden, counts_of):
    d = golden.dims
    golden.step_count += 1
    golden._opt_obj("embed", golden.w_embed)
    speed = d.moe.bias_update_speed
    for i, leaves in enumerate(golden.w_blocks):
        golden._opt_obj(f"block_{i}", leaves)
        if "w_router_bias" in leaves and speed and i in counts_of:
            c = counts_of[i]
            b = leaves["w_router_bias"]
            b.data.add_(torch.sign(c.mean() - c).to(b.dtype), alpha=speed)
    golden._opt_obj("head", golden.w_head)
