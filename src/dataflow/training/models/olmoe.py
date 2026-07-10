"""OLMoE family: config + declarations over the generic machinery.

First MoE family on the pluggable module (tasks/moe/):
qwen3-shaped dense attention (full-row qk-norm, no GQA at 7B, rope 1e4)
with a routed SwiGLU MoE FFN every layer — E=64, top-8, F=1024,
softmax_then_topk (norm_topk_prob=false), gradient-injected load-balance
aux at alpha=0.01, vocab 50304, untied. ~6.92B params; W+dW+O at bf16
~55 GB pinned (full scale fits this box).

Roofline seed convention for MoE kinds: FLOPs from the ACTIVE params
(top-k experts per token), weight BYTES from the FULL expert stack —
every expert's weights are read by the grouped GEMMs regardless of
routing; that asymmetry (tiny compute, huge streamed weights) is the
regime under study.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from dataflow.core import Program
from dataflow.tasks.layouts import (
    DTypePolicy,
    OlmoeDims,
    embed_weight_layout,
    head_weight_layout,
    olmoe_activation_layout,
    olmoe_weight_layout,
)
from dataflow.tasks.modules.moe.spec import MoESpec, moe_aux_temp_layout

from ..lowering import FamilyLayouts, LayerLayout, apply_exact_sizes, initial_values_from_layouts, size_of_factory
from ..shaped_program import BF16, LayerKindSpec, ShapedHardware, build_shaped_program


@dataclass(frozen=True)
class ShapedOlmoeConfig:
    n_layers: int = 16
    d_model: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 16
    head_dim: int = 128
    n_experts: int = 64
    top_k: int = 8
    d_ff_expert: int = 1024
    routing_mode: str = "softmax_then_topk"   # norm_topk_prob=false
    aux_coef: float = 0.01
    vocab_size: int = 50_304
    seq_len: int = 4096
    batch: int = 1
    grad_accum_rounds: int = 1
    num_steps: int = 1
    optimizer_placement: str = "interleaved"
    # per-field optimizer assignment (tasks/optim.py): "adamw" (default,
    # historical behavior) | "sgd" | "sgdm" | "muon" | an OptPolicy with
    # fnmatch overrides. update_specials (noaux bias, frozen) stay the
    # highest-priority per-field override on top of this.
    opt_policy: object = "adamw"
    rope_base: float = 10_000.0
    dtypes: DTypePolicy = DTypePolicy()
    seq_lens: tuple[int, ...] | None = None

    @property
    def tokens(self) -> int:
        if self.seq_lens is not None:
            return sum(self.seq_lens)
        return self.seq_len * self.batch

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def d_ff(self) -> int:  # metadata consumers; per-expert width
        return self.d_ff_expert

    # -- parameter counts (duck-typed for the shared chain builder) -----------
    @property
    def block_params(self) -> int:
        d, q, kv, f = self.d_model, self.q_dim, self.kv_dim, self.d_ff_expert
        attn = d * q + 2 * d * kv + q * d
        moe = d * self.n_experts + self.n_experts * 3 * f * d
        norms = 2 * d + q + kv
        return attn + moe + norms

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def head_params(self) -> int:
        return self.vocab_size * self.d_model

    @classmethod
    def tiny(cls) -> "ShapedOlmoeConfig":
        return cls(
            n_layers=2, d_model=128, n_heads=4, n_kv_heads=4, head_dim=32,
            n_experts=8, top_k=2, d_ff_expert=128, vocab_size=512,
            seq_len=128, batch=1,
        )

    @classmethod
    def olmoe_7b(cls, *, seq_len: int = 4096, batch: int = 1,
                 grad_accum_rounds: int = 1, num_steps: int = 1) -> "ShapedOlmoeConfig":
        return cls(seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)


def moe_spec_of(cfg: ShapedOlmoeConfig) -> MoESpec:
    return MoESpec(
        n_experts=cfg.n_experts, top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert,
        routing_mode=cfg.routing_mode, aux_coef=cfg.aux_coef,
    )


def dims_of_olmoe(cfg: ShapedOlmoeConfig) -> OlmoeDims:
    return OlmoeDims(
        opt_policy=cfg.opt_policy,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        d_ff=cfg.d_ff_expert,
        vocab_size=cfg.vocab_size,
        tokens=cfg.tokens,
        seq_len=cfg.seq_len,
        rope_base=cfg.rope_base,
        dtypes=getattr(cfg, "dtypes", None) or DTypePolicy(),
        seq_lens=getattr(cfg, "seq_lens", None),
        moe=moe_spec_of(cfg),
    )


def _kind_spec(cfg: ShapedOlmoeConfig, hw: ShapedHardware) -> LayerKindSpec:
    """One MoE-attention kind. FLOPs = active params; bytes = full stack."""
    dims = dims_of_olmoe(cfg)
    wl = olmoe_weight_layout(dims, layer=0)
    cl = olmoe_activation_layout(dims)
    t, d, seq = cfg.tokens, cfg.d_model, cfg.seq_len
    q, kv, f, k = cfg.q_dim, cfg.kv_dim, cfg.d_ff_expert, cfg.top_k

    total_params = sum(int(math.prod(fl.shape)) for fl in wl.fields)
    active_mm = d * q + 2 * d * kv + q * d + d * cfg.n_experts + k * 3 * f * d
    mm_flops = 2.0 * t * active_mm
    # weight bytes: the FULL expert stack is read; activation bytes include
    # the permuted (t*K) dispatch/combine traffic
    mm_bytes = BF16 * (total_params + t * 4 * d + t * k * (3 * f + 2 * d))
    attn_flops = 2.0 * t * seq * q
    attn_bytes = BF16 * t * (2 * q + 2 * kv)

    fwd = hw.matmul_us(mm_flops, mm_bytes) + hw.attn_us(attn_flops, attn_bytes)
    bwd = hw.matmul_us(2.0 * mm_flops, 2.0 * mm_bytes) \
        + hw.attn_us(2.5 * attn_flops, 2.0 * attn_bytes)
    sub_fwd = [
        {"kind": "roofline", "name": "moeattn_matmuls", "flops": int(mm_flops),
         "memory_bytes": int(mm_bytes), "efficiency": "matmul"},
        {"kind": "roofline", "name": "attention", "flops": int(attn_flops),
         "memory_bytes": int(attn_bytes), "efficiency": "attention"},
    ]
    sub_bwd = [
        {"kind": "roofline", "name": "moeattn_matmuls_bwd", "flops": int(2 * mm_flops),
         "memory_bytes": int(2 * mm_bytes), "efficiency": "matmul"},
        {"kind": "roofline", "name": "attention_bwd", "flops": int(2.5 * attn_flops),
         "memory_bytes": int(2 * attn_bytes), "efficiency": "attention"},
    ]
    return LayerKindSpec(
        key_prefix="moeattn",
        w_bytes=wl.total_bytes,
        a_bytes=cl.total_bytes,
        aux_temp_bytes=moe_aux_temp_layout(dims, dims.moe).total_bytes,
        fwd_us=fwd, bwd_us=bwd, recompute_us=fwd,
        optimizer_us=hw.mem_us(BF16 * 7.0 * total_params),
        fwd_subops=sub_fwd, bwd_subops=sub_bwd, recompute_subops=list(sub_fwd),
        optimizer_subops=[{"kind": "roofline", "name": "adamw", "flops": 0,
                           "memory_bytes": int(BF16 * 7 * total_params),
                           "efficiency": "memory"}],
    )


def build_shaped_olmoe(
    cfg: ShapedOlmoeConfig,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels=None,
    name: str | None = None,
):
    hw = hw or ShapedHardware()
    from ..freeze_plan import derive_freeze_plan

    dims_fp, fl_fp = family_layouts(cfg)
    freeze_plan = derive_freeze_plan(
        dims_fp, cfg.n_layers,
        lambda i: [f.name for f in fl_fp.layers[i].weights.fields],
        tied_embeddings=bool(getattr(cfg, "tied_embeddings", False)),
    )
    return build_shaped_program(
        cfg, hw=hw, family="olmoe-shaped",
        kinds={"moe": _kind_spec(cfg, hw)},
        fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=name,
        freeze=freeze_plan,
    )


def family_layouts(cfg: ShapedOlmoeConfig) -> tuple[OlmoeDims, FamilyLayouts]:
    dims = dims_of_olmoe(cfg)
    cl = olmoe_activation_layout(dims)
    return dims, FamilyLayouts(
        layers=[LayerLayout(kind="moe",
                            weights=olmoe_weight_layout(dims, layer=i),
                            activations=cl,
                            aux_temp=moe_aux_temp_layout(dims, dims.moe))
                for i in range(cfg.n_layers)],
        embed=embed_weight_layout(dims),
        head=head_weight_layout(dims),
    )


def lower_olmoe(
    cfg: ShapedOlmoeConfig,
    *,
    hw: ShapedHardware | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    fast_memory_capacity: int | None = None,
) -> Program:
    dims, fl = family_layouts(cfg)
    if dims.moe.is_partial:
        raise NotImplementedError(
            "partial expert ownership (expert_ids) is accounting-only in v1 — "
            "program lowering needs the multi-rank runtime (EP)"
        )
    shaped = build_shaped_olmoe(
        cfg, hw=hw, recompute_levels=recompute_levels, fast_memory_capacity=fast_memory_capacity,
    )
    return apply_exact_sizes(shaped, "olmoe-exact", size_of=size_of_factory(dims, fl))


def initial_values_olmoe(program: Program, cfg: ShapedOlmoeConfig, backend, *, seed: int = 0, into=None):
    dims, fl = family_layouts(cfg)
    return initial_values_from_layouts(program, dims, fl, backend, seed=seed, into=into)
