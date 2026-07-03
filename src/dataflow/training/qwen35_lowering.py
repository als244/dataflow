"""Execution-grade Qwen3.5 lowering: heterogeneous per-layer sizes.

Embeddings follow the config: the 9B is UNTIED (bare-table W_embed +
[table | final_norm_w] W_head, llama3/qwen3 shape); the 2B-style tied
variant packs the head layout into the single W_embed."""
from __future__ import annotations

from typing import Mapping

from dataflow.core import Program
from dataflow.tasks.layouts import (
    adamw_state_layout,
    head_weight_layout,
    qwen35_attn_context_layout,
    qwen35_attn_weight_layout,
    qwen35_lin_context_layout,
    qwen35_lin_weight_layout,
)
from .llama3_lowering import apply_exact_sizes, fill_initial_values
from .shaped_llama3 import ShapedHardware
from .shaped_qwen35 import ShapedQwen35Config, build_shaped_qwen35, dims_of_qwen35


def _size_of_factory(cfg: ShapedQwen35Config):
    dims = dims_of_qwen35(cfg)
    hl = head_weight_layout(dims)  # [table | final_norm_w]
    table_bytes = 2 * dims.vocab_size * dims.d_model
    wl = {
        "lin": qwen35_lin_weight_layout(dims),
        "full": qwen35_attn_weight_layout(dims),
    }
    cl = {
        "lin": qwen35_lin_context_layout(dims),
        "full": qwen35_attn_context_layout(dims),
    }
    # tied (2B-style): ONE W_embed carries [table | final_norm_w] and serves
    # embed AND head. untied (the 9B): W_embed is the bare table, the head
    # layout rides W_head — exactly the llama3/qwen3 shape.
    embed_bytes = hl.total_bytes if cfg.tied_embeddings else table_bytes
    o_e = adamw_state_layout(embed_bytes // 2).total_bytes
    o_h = adamw_state_layout(hl.total_bytes // 2).total_bytes
    o_block = {
        k: adamw_state_layout(v.total_bytes // 2).total_bytes for k, v in wl.items()
    }

    def kind(layer: int) -> str:
        return dims.kind_of(layer)

    def size_of(oid: str) -> int | None:
        if oid == "W_embed" or oid.startswith("dW_embed"):
            return embed_bytes
        if oid == "O_embed":
            return o_e
        if oid == "W_head" or oid.startswith("dW_head"):
            return hl.total_bytes
        if oid == "O_head":
            return o_h
        if oid.startswith("A_"):            # A_{s}_{r}_{i}
            return cl[kind(int(oid.rsplit("_", 1)[1]))].total_bytes
        if oid.startswith("dW_"):           # dW_{s}_{i}
            return wl[kind(int(oid.rsplit("_", 1)[1]))].total_bytes
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
    import math

    import torch

    from dataflow.tasks.interop import torch_view
    from dataflow.tasks.layouts import head_weight_layout as _hl

    dims = dims_of_qwen35(cfg)
    wl = {
        "lin": qwen35_lin_weight_layout(dims),
        "full": qwen35_attn_weight_layout(dims),
    }
    gen = torch.Generator().manual_seed(seed)
    buffers = {}
    for spec in program.initial_objects:
        if spec.id in buffers:
            continue
        buf = backend.alloc("backing", spec.size_bytes)
        buffers[spec.id] = buf
        if spec.id.startswith("W_") and spec.id not in ("W_embed", "W_head"):
            layer = int(spec.id.split("_")[1])
            layout = wl[dims.kind_of(layer)]
            flat = torch_view(buf, (spec.size_bytes // 2,), torch.bfloat16)
            flat.copy_(torch.randn(spec.size_bytes // 2, generator=gen) * 0.02)
            for f in layout.fields:
                start = f.offset_bytes // 2
                n = int(math.prod(f.shape))
                if f.name.endswith("_norm_w"):
                    flat[start : start + n] = 1.0
                elif f.name == "A_log":
                    # decay magnitudes ~ U(1, 16) in log space (GDN convention)
                    a0 = torch.empty(n).uniform_(1.0, 16.0, generator=gen).log()
                    flat[start : start + n] = a0.to(torch.bfloat16)
                elif f.name == "dt_bias":
                    flat[start : start + n] = 0.0
        elif spec.id in ("W_embed", "W_head"):
            flat = torch_view(buf, (spec.size_bytes // 2,), torch.bfloat16)
            flat.copy_(torch.randn(spec.size_bytes // 2, generator=gen) * 0.02)
            # [table | final_norm_w] layout rides W_embed when tied, W_head
            # when untied; the untied W_embed is a bare table (no norm field)
            if spec.id == "W_head" or cfg.tied_embeddings:
                for f in _hl(dims).fields:
                    if f.name.endswith("_norm_w"):
                        start = f.offset_bytes // 2
                        flat[start : start + f.shape[0]] = 1.0
        elif spec.id.startswith("O_"):
            torch_view(buf, (spec.size_bytes // 2,), torch.bfloat16).zero_()
        elif spec.id.startswith(("tokens_", "targets_")):
            ids = torch.randint(0, dims.vocab_size, (dims.tokens,), generator=gen, dtype=torch.int32)
            torch_view(buf, (dims.tokens,), torch.int32).copy_(ids)
    return buffers
