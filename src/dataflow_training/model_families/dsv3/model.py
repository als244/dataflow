"""DeepSeek-V3 family: config + declarations over the generic machinery.

MLA attention (blocks/modules/mla_forms.py conventions; compressed-latent ctx)
+ hybrid depth: ``first_k_dense`` dense-SwiGLU layers, then MoE layers
with sigmoid_noaux_tc routing (group-limited biased selection, raw-
sigmoid renormalized weights x routed_scaling, NON-GRADIENT balance bias
with the per-step sign rule) and the UNGATED shared expert. Sequence-
wise complementary aux at alpha=1e-4. HF deepseek-ai/DeepSeek-V3 config;
no MTP (main model only — a locked scope decision); their eps 1e-6 vs our global
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
from dataflow_training.blocks.layouts import (
    PackedLayout,
    Dsv3Dims,
    DTypePolicy,
    ParamDTypes,
    dsv3_dense_activation_layout,
    dsv3_dense_weight_layout,
    dsv3_moe_activation_layout,
    dsv3_moe_weight_layout,
    embed_weight_layout,
    head_weight_layout,
)
from dataflow_training.blocks.optim import freeze
from dataflow_training.blocks.modules.moe.spec import MoESpec, moe_aux_layout, moe_aux_temp_layout

from ...lowering.emit import FamilyLayouts, LayerLayout, apply_exact_sizes, initial_values_from_layouts, object_size_factory
from ...lowering.shaped_program import optimizer_cost_seed, BF16, LayerKindSpec, ShapedHardware, build_shaped_program

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
    # per-field optimizer assignment (tasks/optim.py): "adamw" (default,
    # historical behavior) | "sgd" | "sgdm" | "muon" | an OptPolicy with
    # fnmatch overrides. update_specials (noaux bias, frozen) stay the
    # highest-priority per-field override on top of this.
    opt_policy: object = "adamw"
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

    @classmethod
    def kimi_k2(cls, *, seq_len: int = 4096, batch: int = 1,
                grad_accum_rounds: int = 1, num_steps: int = 1,
                ) -> "ShapedDsv3Config":
        """Kimi K2 family (Moonshot): a faithful DeepSeek-V3-arch model —
        HF architectures=DeepseekV3ForCausalLM — at 1.03T/32B-active.
        Verified against moonshotai/Kimi-K2-Instruct and Kimi-K2.6
        config.json (2026-07-07): K2, K2.5, K2.6 and K2.7-Code are
        SHAPE-IDENTICAL (the 2.5+ releases wrap the same LM core in a
        multimodal shell; K2.7-Code's card: 'same DeepseekV3 backbone,
        same MLA dimensions, same 384+1 MoE topology'). Deltas vs
        dsv3_671b: 64 heads (not 128), E=384, first_k_dense=1, groups
        1/1, scaling 2.827, vocab 163840, rope 50k. Trained upstream
        with MuonClip; we train AdamW (shape preset, not optimizer
        parity)."""
        return cls(
            n_layers=61, d_model=7168, n_heads=64,
            q_lora_rank=1536, kv_lora_rank=512,
            qk_nope_dim=128, qk_rope_dim=64, v_head_dim=128,
            d_ff_dense=18432, first_k_dense=1,
            n_experts=384, top_k=8, d_ff_expert=2048,
            n_group=1, topk_group=1, routed_scaling=2.827,
            n_shared_experts=1, d_ff_shared=2048,
            vocab_size=163_840, rope_base=50_000.0,
            seq_len=seq_len, batch=batch,
            grad_accum_rounds=grad_accum_rounds, num_steps=num_steps,
        )

    # K2.5 / K2.6 / K2.7-Code: shape-identical to K2 (see kimi_k2
    # docstring) — named aliases so CONFIGS and sweep notes can say
    # which model a run models without implying a shape difference.
    kimi_k25 = kimi_k2
    kimi_k26 = kimi_k2
    kimi_k27 = kimi_k2


def derive_moe_spec(cfg: ShapedDsv3Config) -> MoESpec:
    return MoESpec(
        n_experts=cfg.n_experts, top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert,
        routing_mode="sigmoid_noaux_tc", aux_coef=cfg.aux_coef,
        n_shared_experts=cfg.n_shared_experts, d_ff_shared=cfg.d_ff_shared,
        shared_gate=False,
        n_group=cfg.n_group, topk_group=cfg.topk_group,
        routed_scaling=cfg.routed_scaling,
        bias_update_speed=cfg.bias_update_speed,
    )


def derive_dims(cfg: ShapedDsv3Config) -> Dsv3Dims:
    return Dsv3Dims(
        opt_policy=freeze(cfg.opt_policy, fields=("w_router_bias",)),
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        q_lora_rank=cfg.q_lora_rank, kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_dim=cfg.qk_nope_dim, qk_rope_dim=cfg.qk_rope_dim,
        v_head_dim=cfg.v_head_dim,
        d_ff=cfg.d_ff_dense, first_k_dense=cfg.first_k_dense,
        vocab_size=cfg.vocab_size,
        tokens=cfg.tokens, seq_len=cfg.seq_len, rope_base=cfg.rope_base,
        dtypes=getattr(cfg, "dtypes", None) or _DSV3_DTYPES,
        seq_lens=getattr(cfg, "seq_lens", None),
        moe=derive_moe_spec(cfg),
        kinds=tuple("dense" if i < cfg.first_k_dense else "moe"
                    for i in range(cfg.n_layers)),
    )


def _kind_specs(cfg: ShapedDsv3Config, hw: ShapedHardware) -> dict[str, LayerKindSpec]:
    dims = derive_dims(cfg)
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

    def spec(prefix, wl, cl, ffn_active, extra_traffic, aux_temp_bytes=0,
             aux_bytes=0):
        total_params = sum(int(math.prod(fl.shape)) for fl in wl.fields)
        opt_us, sub_opt = optimizer_cost_seed(
            cfg, hw, [(f.name, f.shape) for f in wl.fields])
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
            aux_temp_bytes=aux_temp_bytes,
            aux_bytes=aux_bytes,
            fwd_us=fwd, bwd_us=bwd, recompute_us=fwd,
            optimizer_us=opt_us,
            fwd_subops=sub_fwd, bwd_subops=sub_bwd, recompute_subops=list(sub_fwd),
            optimizer_subops=sub_opt,
        )

    dense = spec(
        "mladense", dsv3_dense_weight_layout(dims), dsv3_dense_activation_layout(dims),
        3 * cfg.d_ff_dense * d, 0.0,
    )
    f, fs, k = cfg.d_ff_expert, cfg.d_ff_shared, cfg.top_k
    moe_active = (
        d * cfg.n_experts + k * 3 * f * d
        + cfg.n_shared_experts * 3 * fs * d
    )
    moe_traffic = BF16 * t * k * (3 * f + 2 * d)
    moe = spec(
        "mlamoe", dsv3_moe_weight_layout(dims), dsv3_moe_activation_layout(dims),
        moe_active, moe_traffic,
        aux_temp_bytes=moe_aux_temp_layout(dims, dims.moe).total_bytes,
        aux_bytes=moe_aux_layout(dims, dims.moe).total_bytes,
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
    dims = derive_dims(cfg)
    from ...lowering.freeze_plan import derive_freeze_plan

    dims_fp, fl_fp = family_layouts(cfg)
    freeze_plan = derive_freeze_plan(
        dims_fp, cfg.n_layers,
        lambda i: [f.name for f in fl_fp.layers[i].weights.fields],
        tied_embeddings=bool(getattr(cfg, "tied_embeddings", False)),
    )
    return build_shaped_program(
        cfg, hw=hw, family="dsv3-shaped",
        kinds=_kind_specs(cfg, hw), layer_kinds=dims.kinds,
        bias_update_in_bwd=True,
        fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=name,
        freeze=freeze_plan,
    )


_WEIGHT_BUILDERS = {"dense": dsv3_dense_weight_layout, "moe": dsv3_moe_weight_layout}


def family_layouts(cfg: ShapedDsv3Config) -> tuple[Dsv3Dims, FamilyLayouts]:
    dims = derive_dims(cfg)
    ctx = {
        "dense": dsv3_dense_activation_layout(dims),
        "moe": dsv3_moe_activation_layout(dims),
    }
    return dims, FamilyLayouts(
        layers=[LayerLayout(kind=dims.kinds[i],
                            weights=_WEIGHT_BUILDERS[dims.kinds[i]](dims, layer=i),
                            activations=ctx[dims.kinds[i]],
                            aux_temp=(moe_aux_temp_layout(dims, dims.moe)
                                      if dims.kinds[i] == "moe"
                                      else PackedLayout.build([])),
                            aux=(moe_aux_layout(dims, dims.moe)
                                 if dims.kinds[i] == "moe" else None))
                for i in range(cfg.n_layers)],
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
    return apply_exact_sizes(shaped, "dsv3-exact", object_size=object_size_factory(dims, fl))


def initial_values_dsv3(program: Program, cfg: ShapedDsv3Config, backend, *, seed: int = 0, into=None):
    dims, fl = family_layouts(cfg)
    from dataflow_training.model_families.init_policy import build_init_policy

    return initial_values_from_layouts(
        program, dims, fl, backend, seed=seed, into=into,
        init_policy=build_init_policy(getattr(cfg, "init_policy", None)))
