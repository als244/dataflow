"""Execution-grade Qwen3.5 lowering: heterogeneous per-layer sizes.

Embeddings follow the config: the 9B is UNTIED (bare-table W_embed +
[table | final_norm_w] W_head, llama3/qwen3 shape); the 2B-style tied
variant packs the head layout into the single W_embed."""
from __future__ import annotations

from typing import Mapping

from dataflow.core import Program
from dataflow.tasks.layouts import (
    embed_weight_layout,
    grad_layout,
    head_weight_layout,
    opt_state_layout,
    qwen35_attn_context_layout,
    qwen35_attn_weight_layout,
    qwen35_lin_context_layout,
    qwen35_lin_weight_layout,
)
from .llama3_lowering import apply_exact_sizes, fill_weight_fields
from .shaped_llama3 import ShapedHardware
from .shaped_qwen35 import ShapedQwen35Config, build_shaped_qwen35, dims_of_qwen35


def _size_of_factory(cfg: ShapedQwen35Config):
    dims = dims_of_qwen35(cfg)
    p = dims.dtypes
    hl = head_weight_layout(dims)  # [table | final_norm_w]
    el = embed_weight_layout(dims)  # bare table (untied W_embed)
    wl = {
        "lin": qwen35_lin_weight_layout(dims),
        "full": qwen35_attn_weight_layout(dims),
    }
    cl = {
        "lin": qwen35_lin_context_layout(dims),
        "full": qwen35_attn_context_layout(dims),
    }
    # tied (2B-style): ONE W_embed carries [table | final_norm_w] and serves
    # embed AND head (policy-addressed as head.*). untied (the 9B): W_embed
    # is the bare table (embed.*), the head layout rides W_head.
    tied = cfg.tied_embeddings
    embed_wl, embed_ns = (hl, "head") if tied else (el, "embed")
    embed_bytes = embed_wl.total_bytes
    dw_e = grad_layout(embed_wl, p, ns=embed_ns).total_bytes
    dw_h = grad_layout(hl, p, ns="head").total_bytes
    dw_block = {k: grad_layout(v, p).total_bytes for k, v in wl.items()}
    o_e = opt_state_layout(embed_wl, p, ns=embed_ns).total_bytes
    o_h = opt_state_layout(hl, p, ns="head").total_bytes
    o_block = {k: opt_state_layout(v, p).total_bytes for k, v in wl.items()}

    def kind(layer: int) -> str:
        return dims.kind_of(layer)

    def size_of(oid: str) -> int | None:
        if oid == "W_embed":
            return embed_bytes
        if oid.startswith("dW_embed"):
            return dw_e
        if oid == "O_embed":
            return o_e
        if oid == "W_head":
            return hl.total_bytes
        if oid.startswith("dW_head"):
            return dw_h
        if oid == "O_head":
            return o_h
        if oid.startswith("A_"):            # A_{s}_{r}_{i}
            return cl[kind(int(oid.rsplit("_", 1)[1]))].total_bytes
        if oid.startswith("dW_"):           # dW_{s}_{i}
            return dw_block[kind(int(oid.rsplit("_", 1)[1]))]
        if oid.startswith("O_"):            # O_{i}
            return o_block[kind(int(oid.split("_")[1]))]
        if oid.startswith("W_"):            # W_{i}
            return wl[kind(int(oid.split("_")[1]))].total_bytes
        return None

    return size_of


def lower_qwen35(
    cfg: ShapedQwen35Config,
    *,
    hw: ShapedHardware | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    fast_memory_capacity: int | None = None,
) -> Program:
    shaped = build_shaped_qwen35(
        cfg, hw=hw, recompute_levels=recompute_levels, fast_memory_capacity=fast_memory_capacity,
    )
    return apply_exact_sizes(shaped, {}, "qwen35-exact-v1", size_of=_size_of_factory(cfg))


def initial_values_qwen35(program: Program, cfg: ShapedQwen35Config, backend, *, seed: int = 0):
    """Weights N(0, 0.02) bf16, *_norm_w fields ones (per-KIND layouts),
    A_log ~ log U(1,16) and dt_bias zeros (the DeltaNet decay
    parameterization), optimizer state zeros, tokens/targets ints."""
    import torch

    from dataflow.tasks.interop import torch_view

    dims = dims_of_qwen35(cfg)
    wl = {
        "lin": qwen35_lin_weight_layout(dims),
        "full": qwen35_attn_weight_layout(dims),
    }
    # decay magnitudes ~ U(1, 16) in log space (GDN convention)
    special = {
        "A_log": lambda n, g: torch.empty(n).uniform_(1.0, 16.0, generator=g).log(),
        "dt_bias": lambda n, g: torch.zeros(n),
    }
    # [table | final_norm_w] rides W_embed when tied, W_head when untied;
    # the untied W_embed is the bare table
    embed_wl = head_weight_layout(dims) if cfg.tied_embeddings else embed_weight_layout(dims)
    gen = torch.Generator().manual_seed(seed)
    buffers = {}
    for spec in program.initial_objects:
        if spec.id in buffers:
            continue
        buf = backend.alloc("backing", spec.size_bytes)
        buffers[spec.id] = buf
        if spec.id.startswith("W_") and spec.id not in ("W_embed", "W_head"):
            layer = int(spec.id.split("_")[1])
            fill_weight_fields(buf, wl[dims.kind_of(layer)], gen, special=special)
        elif spec.id == "W_embed":
            fill_weight_fields(buf, embed_wl, gen)
        elif spec.id == "W_head":
            fill_weight_fields(buf, head_weight_layout(dims), gen)
        elif spec.id.startswith("O_"):
            torch_view(buf, (spec.size_bytes,), torch.uint8).zero_()
        elif spec.id.startswith(("tokens_", "targets_")):
            ids = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32)
            torch_view(buf, (dims.tokens,), torch.int32).copy_(ids)
    return buffers
