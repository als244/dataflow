"""DeepSeek-V3 family: config + declarations over the generic machinery.

MLA attention (tasks/mla_reference.py conventions; compressed-latent ctx)
+ hybrid depth: ``first_k_dense`` dense-SwiGLU layers, then MoE layers
with sigmoid_noaux_tc routing (group-limited biased selection, raw-
sigmoid renormalized weights x routed_scaling, NON-GRADIENT balance bias
with the per-step sign rule) and the UNGATED shared expert. Sequence-
wise complementary aux at alpha=1e-4. HF deepseek-ai/DeepSeek-V3 config;
no MTP (main model only — Shein call); their eps 1e-6 vs our global
1e-5 kept, standing note.

Presets vs host RAM (this box ~175 GiB usable):
- ``tiny``      — 3L (1 dense + 2 moe), ladder scale.
- ``dsv3_mini`` — the perf config for THIS box: 18L (first 2 dense),
  d=2048, 16 heads, MLA ranks 512/256, head dims 64/32/64, E=128 with
  V3's 8-group/4-kept routing, F=1024 (+shared 1024), dense FFN 8192,
  vocab 129,280 — ~15.3B params, ~92 GiB pinned.
- ``dsv3_671b`` — faithful 61L/d7168/128-head config; lowering/planning
  validation only (~1.34 TB weights bf16 — the big-machine target).

Roofline seeds: MoE FLOPs from ACTIVE params, weight bytes from the full
stack; flash priced at the PADDED head_dim (qk) — the pad cost is real
and belongs in the seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import torch

from dataflow.core import Program
from dataflow.tasks.layouts import (
    Dsv3Dims,
    DTypePolicy,
    ParamDTypes,
    dsv3_dense_context_layout,
    dsv3_dense_weight_layout,
    dsv3_moe_context_layout,
    dsv3_moe_weight_layout,
    embed_weight_layout,
    head_weight_layout,
)
from dataflow.tasks.moe.spec import MoESpec

from .lowering import FamilyLayouts, apply_exact_sizes, initial_values_from_layouts, size_of_factory
from .shaped_program import BF16, LayerKindSpec, ShapedHardware, build_shaped_program

_DSV3_DTYPES = DTypePolicy(overrides=(
    # the balance bias is fp32 end-to-end: bf16 ulp at bias ~0.1 is half
    # the 1e-3 update step; its dW slot carries fp32 COUNTS
    ("w_router_bias", ParamDTypes("fp32", "fp32", "fp32")),
))


@dataclass(frozen=True)
class ShapedDsv3Config:
    n_layers: int = 61
    d_model: int = 7168
    n_heads: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_dim: int = 128
    qk_rope_dim: int = 64
    v_head_dim: int = 128
    d_ff_dense: int = 18432
    first_k_dense: int = 3
    n_experts: int = 256
    top_k: int = 8
    d_ff_expert: int = 2048
    n_group: int = 8
    topk_group: int = 4
    routed_scaling: float = 2.5
    bias_update_speed: float = 0.001
    aux_coef: float = 1e-4
    n_shared_experts: int = 1
    d_ff_shared: int = 2048
    vocab_size: int = 129_280
    seq_len: int = 4096
    batch: int = 1
    grad_accum_rounds: int = 1
    num_steps: int = 1
    optimizer_placement: str = "interleaved"
    rope_base: float = 10_000.0
    dtypes: DTypePolicy = field(default_factory=lambda: _DSV3_DTYPES)
    seq_lens: tuple[int, ...] | None = None

    @property
    def tokens(self) -> int:
        if self.seq_lens is not None:
            return sum(self.seq_lens)
        return self.seq_len * self.batch

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_dim + self.qk_rope_dim

    @property
    def d_ff(self) -> int:  # metadata consumers
        return self.d_ff_dense

    @property
    def n_kv_heads(self) -> int:  # metadata duck-type: MLA is MHA
        return self.n_heads

    @property
    def head_dim(self) -> int:  # metadata duck-type: padded attention dim
        return self.qk_head_dim

    # -- parameter counts (duck-typed roofline seed; per-kind specs carry
    # layout-exact sizes) ------------------------------------------------------
    @property
    def block_params(self) -> int:
        d, h = self.d_model, self.n_heads
        mla = (
            d * self.q_lora_rank
            + self.q_lora_rank * h * self.qk_head_dim
            + d * (self.kv_lora_rank + self.qk_rope_dim)
            + self.kv_lora_rank * h * (self.qk_nope_dim + self.v_head_dim)
            + h * self.v_head_dim * d
        )
        moe = (
            d * self.n_experts
            + self.n_experts * 3 * self.d_ff_expert * d
            + self.n_shared_experts * 3 * self.d_ff_shared * d
        )
        return mla + moe + 2 * d

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def head_params(self) -> int:
        return self.vocab_size * self.d_model

    @classmethod
    def tiny(cls) -> "ShapedDsv3Config":
        return cls(
            n_layers=3, d_model=128, n_heads=4, q_lora_rank=64,
            kv_lora_rank=32, qk_nope_dim=16, qk_rope_dim=8, v_head_dim=16,
            d_ff_dense=256, first_k_dense=1,
            n_experts=8, top_k=2, d_ff_expert=32, n_group=4, topk_group=2,
            d_ff_shared=32, vocab_size=512, seq_len=128, batch=1,
        )

    @classmethod
    def dsv3_mini(cls, *, seq_len: int = 4096, batch: int = 1,
                  grad_accum_rounds: int = 1, num_steps: int = 1,
                  ) -> "ShapedDsv3Config":
        return cls(
            n_layers=18, d_model=2048, n_heads=16, q_lora_rank=512,
            kv_lora_rank=256, qk_nope_dim=64, qk_rope_dim=32, v_head_dim=64,
            d_ff_dense=8192, first_k_dense=2,
            n_experts=128, top_k=8, d_ff_expert=1024, n_group=8, topk_group=4,
            d_ff_shared=1024,
            seq_len=seq_len, batch=batch,
            grad_accum_rounds=grad_accum_rounds, num_steps=num_steps,
        )

    @classmethod
    def dsv3_671b(cls, *, seq_len: int = 4096, batch: int = 1,
                  grad_accum_rounds: int = 1, num_steps: int = 1,
                  ) -> "ShapedDsv3Config":
        return cls(seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)


def moe_spec_of(cfg: ShapedDsv3Config) -> MoESpec:
    return MoESpec(
        n_experts=cfg.n_experts, top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert,
        routing_mode="sigmoid_noaux_tc", aux_coef=cfg.aux_coef,
        n_shared_experts=cfg.n_shared_experts, d_ff_shared=cfg.d_ff_shared,
        shared_gate=False,
        n_group=cfg.n_group, topk_group=cfg.topk_group,
        routed_scaling=cfg.routed_scaling,
        bias_update_speed=cfg.bias_update_speed,
    )


def dims_of_dsv3(cfg: ShapedDsv3Config) -> Dsv3Dims:
    return Dsv3Dims(
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        q_lora_rank=cfg.q_lora_rank, kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_dim=cfg.qk_nope_dim, qk_rope_dim=cfg.qk_rope_dim,
        v_head_dim=cfg.v_head_dim,
        d_ff=cfg.d_ff_dense, first_k_dense=cfg.first_k_dense,
        vocab_size=cfg.vocab_size,
        tokens=cfg.tokens, seq_len=cfg.seq_len, rope_base=cfg.rope_base,
        dtypes=getattr(cfg, "dtypes", None) or _DSV3_DTYPES,
        seq_lens=getattr(cfg, "seq_lens", None),
        moe=moe_spec_of(cfg),
    )


def _kind_specs(cfg: ShapedDsv3Config, hw: ShapedHardware) -> dict[str, LayerKindSpec]:
    dims = dims_of_dsv3(cfg)
    t, d, seq, h = cfg.tokens, cfg.d_model, cfg.seq_len, cfg.n_heads
    qk = cfg.qk_head_dim
    mla_active = (
        d * cfg.q_lora_rank + cfg.q_lora_rank * h * qk
        + d * (cfg.kv_lora_rank + cfg.qk_rope_dim)
        + cfg.kv_lora_rank * h * (cfg.qk_nope_dim + cfg.v_head_dim)
        + h * cfg.v_head_dim * d
    )
    # flash at the PADDED head_dim: scores AND PV both run at qk width
    attn_flops = 2.0 * t * seq * h * qk
    attn_bytes = BF16 * t * 4 * h * qk

    def spec(prefix, wl, cl, ffn_active, extra_traffic):
        total_params = sum(int(math.prod(fl.shape)) for fl in wl.fields)
        mm_flops = 2.0 * t * (mla_active + ffn_active)
        mm_bytes = BF16 * (total_params + 4 * t * d) + extra_traffic
        fwd = hw.matmul_us(mm_flops, mm_bytes) + hw.attn_us(attn_flops, attn_bytes)
        bwd = hw.matmul_us(2.0 * mm_flops, 2.0 * mm_bytes) \
            + hw.attn_us(2.5 * attn_flops, 2.0 * attn_bytes)
        sub_fwd = [
            {"kind": "roofline", "name": f"{prefix}_matmuls", "flops": int(mm_flops),
             "memory_bytes": int(mm_bytes), "efficiency": "matmul"},
            {"kind": "roofline", "name": "attention", "flops": int(attn_flops),
             "memory_bytes": int(attn_bytes), "efficiency": "attention"},
        ]
        sub_bwd = [
            {"kind": "roofline", "name": f"{prefix}_matmuls_bwd", "flops": int(2 * mm_flops),
             "memory_bytes": int(2 * mm_bytes), "efficiency": "matmul"},
            {"kind": "roofline", "name": "attention_bwd", "flops": int(2.5 * attn_flops),
             "memory_bytes": int(2 * attn_bytes), "efficiency": "attention"},
        ]
        return LayerKindSpec(
            key_prefix=prefix,
            w_bytes=wl.total_bytes,
            a_bytes=cl.total_bytes,
            fwd_us=fwd, bwd_us=bwd, recompute_us=fwd,
            optimizer_us=hw.mem_us(BF16 * 7.0 * total_params),
            fwd_subops=sub_fwd, bwd_subops=sub_bwd, recompute_subops=list(sub_fwd),
            optimizer_subops=[{"kind": "roofline", "name": "adamw", "flops": 0,
                               "memory_bytes": int(BF16 * 7 * total_params),
                               "efficiency": "memory"}],
        )

    dense = spec(
        "mladense", dsv3_dense_weight_layout(dims), dsv3_dense_context_layout(dims),
        3 * cfg.d_ff_dense * d, 0.0,
    )
    f, fs, k = cfg.d_ff_expert, cfg.d_ff_shared, cfg.top_k
    moe_active = (
        d * cfg.n_experts + k * 3 * f * d
        + cfg.n_shared_experts * 3 * fs * d
    )
    moe_traffic = BF16 * t * k * (3 * f + 2 * d)
    moe = spec(
        "mlamoe", dsv3_moe_weight_layout(dims), dsv3_moe_context_layout(dims),
        moe_active, moe_traffic,
    )
    return {"dense": dense, "moe": moe}


def build_shaped_dsv3(
    cfg: ShapedDsv3Config,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels=None,
    name: str | None = None,
):
    hw = hw or ShapedHardware()
    dims = dims_of_dsv3(cfg)
    return build_shaped_program(
        cfg, hw=hw, family="dsv3-shaped",
        kinds=_kind_specs(cfg, hw), kind_of=dims.kind_of,
        fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=name,
    )


_WEIGHT_BUILDERS = {"dense": dsv3_dense_weight_layout, "moe": dsv3_moe_weight_layout}


def family_layouts(cfg: ShapedDsv3Config) -> tuple[Dsv3Dims, FamilyLayouts]:
    dims = dims_of_dsv3(cfg)
    ctx = {
        "dense": dsv3_dense_context_layout(dims),
        "moe": dsv3_moe_context_layout(dims),
    }
    return dims, FamilyLayouts(
        n_layers=cfg.n_layers,
        block_weight_at=lambda i: _WEIGHT_BUILDERS[dims.kind_of(i)](dims, layer=i),
        block_context_at=lambda i: ctx[dims.kind_of(i)],
        embed=embed_weight_layout(dims),
        head=head_weight_layout(dims),
        init_specials={
            # the balance bias starts at ZERO (V3) — never randn
            "w_router_bias": lambda n, gen: torch.zeros(n),
        },
    )


def lower_dsv3(
    cfg: ShapedDsv3Config,
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
    shaped = build_shaped_dsv3(
        cfg, hw=hw, recompute_levels=recompute_levels, fast_memory_capacity=fast_memory_capacity,
    )
    return apply_exact_sizes(shaped, "dsv3-exact-v1", size_of=size_of_factory(dims, fl))


def initial_values_dsv3(program: Program, cfg: ShapedDsv3Config, backend, *, seed: int = 0):
    dims, fl = family_layouts(cfg)
    return initial_values_from_layouts(program, dims, fl, backend, seed=seed)
