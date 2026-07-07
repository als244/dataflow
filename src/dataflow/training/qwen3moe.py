"""Qwen3-MoE family: config + declarations over the generic machinery.

Third family on the pluggable MoE module and the lightest plug-in yet —
qwen3's dense attention reused verbatim (per-head qk-norm, GQA, rope 1e6),
FFN swapped for the routed SwiGLU MoE: E=128, top-8,
``topk_then_softmax`` (norm_topk_prob=true), gradient-injected
load-balance aux at alpha=0.001, NO shared expert, all layers sparse,
vocab 151,936, untied (HF Qwen/Qwen3-30B-A3B + Qwen3-235B-A22B configs;
their rms eps 1e-6 vs our global 1e-5 — ours kept, same note as the
other MoE families).

Presets and the host-RAM reality (this box: ~175 GiB usable):
- ``qwen3moe_30b``  — full 48L/30.5B. W+dW+O bf16 ~183 GiB pinned: OVER
  this host's ceiling; lowering/planning-validated, not trainable here.
- ``qwen3moe_30b_24l`` — the perf config (qwen35moe-20l precedent):
  half depth, ~15.6B, ~94 GiB pinned.
- ``qwen3moe_235b`` — 94L/d4096/235B; definition + lowering/planning
  validation only (~1.4 TB pinned).

Roofline seed convention (MoE kinds): FLOPs from ACTIVE params (top-k),
weight BYTES from the FULL expert stack.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from dataflow.core import Program
from dataflow.tasks.layouts import (
    DTypePolicy,
    Qwen3MoeDims,
    embed_weight_layout,
    head_weight_layout,
    qwen3moe_context_layout,
    qwen3moe_weight_layout,
)
from dataflow.tasks.moe.spec import MoESpec, moe_meta_layout

from .lowering import FamilyLayouts, apply_exact_sizes, initial_values_from_layouts, size_of_factory
from .shaped_program import BF16, LayerKindSpec, ShapedHardware, build_shaped_program


@dataclass(frozen=True)
class ShapedQwen3MoeConfig:
    n_layers: int = 48
    d_model: int = 2048
    n_heads: int = 32
    n_kv_heads: int = 4
    head_dim: int = 128
    n_experts: int = 128
    top_k: int = 8
    d_ff_expert: int = 768
    routing_mode: str = "topk_then_softmax"   # norm_topk_prob=true
    aux_coef: float = 0.001
    vocab_size: int = 151_936
    seq_len: int = 4096
    batch: int = 1
    grad_accum_rounds: int = 1
    num_steps: int = 1
    optimizer_placement: str = "interleaved"
    rope_base: float = 1_000_000.0
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
        norms = 2 * d + 2 * self.head_dim   # per-head qk-norm weights
        return attn + moe + norms

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def head_params(self) -> int:
        return self.vocab_size * self.d_model

    @classmethod
    def tiny(cls) -> "ShapedQwen3MoeConfig":
        # GQA exercised (4 q / 2 kv heads) — the real models are 32/4, 64/4
        return cls(
            n_layers=2, d_model=128, n_heads=4, n_kv_heads=2, head_dim=32,
            n_experts=8, top_k=2, d_ff_expert=64, vocab_size=512,
            seq_len=128, batch=1,
        )

    @classmethod
    def qwen3moe_30b(cls, *, seq_len: int = 4096, batch: int = 1,
                     grad_accum_rounds: int = 1, num_steps: int = 1,
                     ) -> "ShapedQwen3MoeConfig":
        return cls(seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)

    @classmethod
    def qwen3moe_30b_24l(cls, *, seq_len: int = 4096, batch: int = 1,
                         grad_accum_rounds: int = 1, num_steps: int = 1,
                         ) -> "ShapedQwen3MoeConfig":
        return cls(n_layers=24, seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)

    @classmethod
    def qwen3moe_235b(cls, *, seq_len: int = 4096, batch: int = 1,
                      grad_accum_rounds: int = 1, num_steps: int = 1,
                      ) -> "ShapedQwen3MoeConfig":
        return cls(n_layers=94, d_model=4096, n_heads=64, n_kv_heads=4,
                   d_ff_expert=1536, seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)


def moe_spec_of(cfg: ShapedQwen3MoeConfig) -> MoESpec:
    return MoESpec(
        n_experts=cfg.n_experts, top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert,
        routing_mode=cfg.routing_mode, aux_coef=cfg.aux_coef,
    )


def dims_of_qwen3moe(cfg: ShapedQwen3MoeConfig) -> Qwen3MoeDims:
    return Qwen3MoeDims(
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


def _kind_spec(cfg: ShapedQwen3MoeConfig, hw: ShapedHardware) -> LayerKindSpec:
    """One MoE-attention kind. FLOPs = active params; bytes = full stack."""
    dims = dims_of_qwen3moe(cfg)
    wl = qwen3moe_weight_layout(dims, layer=0)
    cl = qwen3moe_context_layout(dims)
    t, d, seq = cfg.tokens, cfg.d_model, cfg.seq_len
    q, kv, f, k = cfg.q_dim, cfg.kv_dim, cfg.d_ff_expert, cfg.top_k

    total_params = sum(int(math.prod(fl.shape)) for fl in wl.fields)
    active_mm = d * q + 2 * d * kv + q * d + d * cfg.n_experts + k * 3 * f * d
    mm_flops = 2.0 * t * active_mm
    mm_bytes = BF16 * (total_params + t * 4 * d + t * k * (3 * f + 2 * d))
    attn_flops = 2.0 * t * seq * q
    attn_bytes = BF16 * t * (2 * q + 2 * kv)

    fwd = hw.matmul_us(mm_flops, mm_bytes) + hw.attn_us(attn_flops, attn_bytes)
    bwd = hw.matmul_us(2.0 * mm_flops, 2.0 * mm_bytes) \
        + hw.attn_us(2.5 * attn_flops, 2.0 * attn_bytes)
    sub_fwd = [
        {"kind": "roofline", "name": "q3moeattn_matmuls", "flops": int(mm_flops),
         "memory_bytes": int(mm_bytes), "efficiency": "matmul"},
        {"kind": "roofline", "name": "attention", "flops": int(attn_flops),
         "memory_bytes": int(attn_bytes), "efficiency": "attention"},
    ]
    sub_bwd = [
        {"kind": "roofline", "name": "q3moeattn_matmuls_bwd", "flops": int(2 * mm_flops),
         "memory_bytes": int(2 * mm_bytes), "efficiency": "matmul"},
        {"kind": "roofline", "name": "attention_bwd", "flops": int(2.5 * attn_flops),
         "memory_bytes": int(2 * attn_bytes), "efficiency": "attention"},
    ]
    return LayerKindSpec(
        key_prefix="q3moeattn",
        w_bytes=wl.total_bytes,
        a_bytes=cl.total_bytes,
        meta_bytes=moe_meta_layout(dims, dims.moe).total_bytes,
        fwd_us=fwd, bwd_us=bwd, recompute_us=fwd,
        optimizer_us=hw.mem_us(BF16 * 7.0 * total_params),
        fwd_subops=sub_fwd, bwd_subops=sub_bwd, recompute_subops=list(sub_fwd),
        optimizer_subops=[{"kind": "roofline", "name": "adamw", "flops": 0,
                           "memory_bytes": int(BF16 * 7 * total_params),
                           "efficiency": "memory"}],
    )


def build_shaped_qwen3moe(
    cfg: ShapedQwen3MoeConfig,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels=None,
    name: str | None = None,
):
    hw = hw or ShapedHardware()
    return build_shaped_program(
        cfg, hw=hw, family="qwen3moe-shaped",
        kinds={"moe": _kind_spec(cfg, hw)},
        fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=name,
    )


def family_layouts(cfg: ShapedQwen3MoeConfig) -> tuple[Qwen3MoeDims, FamilyLayouts]:
    dims = dims_of_qwen3moe(cfg)
    cl = qwen3moe_context_layout(dims)
    return dims, FamilyLayouts(
        n_layers=cfg.n_layers,
        block_weight_at=lambda i: qwen3moe_weight_layout(dims, layer=i),
        block_context_at=lambda i: cl,
        embed=embed_weight_layout(dims),
        head=head_weight_layout(dims),
        block_meta_at=lambda i: moe_meta_layout(dims, dims.moe),
    )


def lower_qwen3moe(
    cfg: ShapedQwen3MoeConfig,
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
    shaped = build_shaped_qwen3moe(
        cfg, hw=hw, recompute_levels=recompute_levels, fast_memory_capacity=fast_memory_capacity,
    )
    return apply_exact_sizes(shaped, "qwen3moe-exact", size_of=size_of_factory(dims, fl))


def initial_values_qwen3moe(program: Program, cfg: ShapedQwen3MoeConfig, backend, *, seed: int = 0):
    dims, fl = family_layouts(cfg)
    return initial_values_from_layouts(program, dims, fl, backend, seed=seed)
