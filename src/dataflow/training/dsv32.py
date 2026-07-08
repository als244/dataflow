"""DeepSeek-V3.2 family: config + declarations over the generic machinery.

dsv3's backbone (MLA + noaux_tc MoE + hybrid depth) with DSA — lightning
indexer + fine-grained top-k selection — in EVERY layer's attention.
Sparse mode only in v1 (Shein order: sparse correctness -> sparse perf ->
dense warm-up last); ``sparse_mode=False`` = the dense warm-up mode.

Presets:
- ``tiny``       — 3L (1 dense + 2 moe), indexer 8x32, k=24, ladder scale.
- ``dsv32_mini`` — the perf config for THIS box: dsv3_mini dims + indexer
  H_I=8/d_I=64 (V3.2 ratios: H_I = h/2, d_I = nope), k=1024, DEFAULT
  seq_len 4096 so sparsity is ACTIVE (65,536 tok/step = 16 seqs at s4k;
  at s1k, k=1024 selects everything). ~12.8B, ~78 GiB pinned.
- ``dsv32_671b`` — faithful 61L config (indexer 64x128, k=2048);
  lowering/planning validation only.

Roofline seeds price DSA explicitly: indexer scores ~ 2*t*sbar*H_I*d_I
(sbar = seq/2 causal average), sparse core ~ 2*t*min(k, sbar)*h*qk*2
instead of dense attention, both fwd and (x2 + target-recompute) bwd.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import torch

from dataflow.tasks.optim import OptPolicy
from dataflow.core import Program
from dataflow.tasks.layouts import (
    Dsv32Dims,
    dsv32_meta_layout,
    DTypePolicy,
    ParamDTypes,
    dsv32_dense_context_layout,
    dsv32_dense_weight_layout,
    dsv32_moe_context_layout,
    dsv32_moe_weight_layout,
    embed_weight_layout,
    head_weight_layout,
)
from dataflow.tasks.modules.moe.spec import MoESpec

from .lowering import FamilyLayouts, apply_exact_sizes, initial_values_from_layouts, size_of_factory
from .shaped_program import BF16, LayerKindSpec, ShapedHardware, build_shaped_program

_DSV32_DTYPES = DTypePolicy(overrides=(
    ("w_router_bias", ParamDTypes("fp32", "fp32", "fp32")),
    # the indexer weights projection is fp32 in the reference model
    ("w_idx_w", ParamDTypes("fp32", "fp32", "fp32")),
))


@dataclass(frozen=True)
class ShapedDsv32Config:
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
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    sparse_mode: bool = True
    train_indexer: bool = True
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
    dtypes: DTypePolicy = field(default_factory=lambda: _DSV32_DTYPES)
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
    def d_ff(self) -> int:
        return self.d_ff_dense

    @property
    def n_kv_heads(self) -> int:  # metadata duck-type
        return self.n_heads

    @property
    def head_dim(self) -> int:  # metadata duck-type
        return self.qk_head_dim

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
        idx = (
            self.q_lora_rank * self.index_n_heads * self.index_head_dim
            + d * self.index_head_dim + d * self.index_n_heads
        )
        moe = (
            d * self.n_experts
            + self.n_experts * 3 * self.d_ff_expert * d
            + self.n_shared_experts * 3 * self.d_ff_shared * d
        )
        return mla + idx + moe + 2 * d

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def head_params(self) -> int:
        return self.vocab_size * self.d_model

    @classmethod
    def tiny(cls) -> "ShapedDsv32Config":
        return cls(
            n_layers=3, d_model=128, n_heads=4, q_lora_rank=64,
            kv_lora_rank=32, qk_nope_dim=16, qk_rope_dim=8, v_head_dim=16,
            d_ff_dense=256, first_k_dense=1,
            n_experts=8, top_k=2, d_ff_expert=32, n_group=4, topk_group=2,
            d_ff_shared=32,
            index_n_heads=8, index_head_dim=32, index_topk=24,
            vocab_size=512, seq_len=128, batch=1,
        )

    @classmethod
    def dsv32_mini(cls, *, seq_len: int = 4096, batch: int = 1,
                   grad_accum_rounds: int = 1, num_steps: int = 1,
                   sparse_mode: bool = True,
                   ) -> "ShapedDsv32Config":
        return cls(
            n_layers=18, d_model=2048, n_heads=16, q_lora_rank=512,
            kv_lora_rank=256, qk_nope_dim=64, qk_rope_dim=32, v_head_dim=64,
            d_ff_dense=8192, first_k_dense=2,
            n_experts=128, top_k=8, d_ff_expert=1024, n_group=8, topk_group=4,
            d_ff_shared=1024,
            index_n_heads=8, index_head_dim=64, index_topk=1024,
            seq_len=seq_len, batch=batch, sparse_mode=sparse_mode,
            grad_accum_rounds=grad_accum_rounds, num_steps=num_steps,
        )

    @classmethod
    def dsv32_671b(cls, *, seq_len: int = 4096, batch: int = 1,
                   grad_accum_rounds: int = 1, num_steps: int = 1,
                   ) -> "ShapedDsv32Config":
        return cls(seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)

    @classmethod
    def glm5(cls, *, seq_len: int = 4096, batch: int = 1,
             grad_accum_rounds: int = 1, num_steps: int = 1,
             sparse_mode: bool = True,
             ) -> "ShapedDsv32Config":
        """GLM-5 family (Zhipu): MLA + DeepSeek Sparse Attention — HF
        model_type glm_moe_dsa. Verified against zai-org/GLM-5 AND
        GLM-5.1 config.json (2026-07-07): the two are SHAPE-IDENTICAL.
        Deltas vs dsv32_671b: 78L, d 6144, 64 heads, q_lora 2048,
        nope 192 (qk=256), v_head 256 (== qk: NO pad asymmetry
        anywhere), dense ff 12288, E=256 F=2048, first_k_dense=3,
        groups 1/1, indexer 32x128 k=2048, vocab 154880, rope 1e6.
        Absorbed dim = 512+64 = 576: FlashMLA-sparse compatible.
        DEVIATION (documented): num_nextn_predict_layers=1 (MTP head)
        is NOT modeled — same no-MTP scoping as the dsv3 family.
        GLM-5.2's IndexShare (indexer reuse across 4 layers) is a
        different architecture; not covered by this preset."""
        return cls(
            n_layers=78, d_model=6144, n_heads=64,
            q_lora_rank=2048, kv_lora_rank=512,
            qk_nope_dim=192, qk_rope_dim=64, v_head_dim=256,
            d_ff_dense=12288, first_k_dense=3,
            n_experts=256, top_k=8, d_ff_expert=2048,
            n_group=1, topk_group=1, routed_scaling=2.5,
            n_shared_experts=1, d_ff_shared=2048,
            index_n_heads=32, index_head_dim=128, index_topk=2048,
            vocab_size=154_880, rope_base=1_000_000.0,
            seq_len=seq_len, batch=batch, sparse_mode=sparse_mode,
            grad_accum_rounds=grad_accum_rounds, num_steps=num_steps,
        )

    # GLM-5.1: shape-identical to GLM-5 (both configs fetched + diffed)
    glm51 = glm5


def moe_spec_of(cfg: ShapedDsv32Config) -> MoESpec:
    return MoESpec(
        n_experts=cfg.n_experts, top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert,
        routing_mode="sigmoid_noaux_tc", aux_coef=cfg.aux_coef,
        n_shared_experts=cfg.n_shared_experts, d_ff_shared=cfg.d_ff_shared,
        shared_gate=False,
        n_group=cfg.n_group, topk_group=cfg.topk_group,
        routed_scaling=cfg.routed_scaling,
        bias_update_speed=cfg.bias_update_speed,
    )


def dims_of_dsv32(cfg: ShapedDsv32Config) -> Dsv32Dims:
    # dense warm-up: freezing is an OPTIMIZER POLICY — every field
    # defaults to "frozen" (zero grad storage, zero opt state; the
    # lowering prunes empty dW/O objects and purposeless optimizer/
    # embed_bwd tasks) and only the indexer trains, with adamw.
    warmup_policy = OptPolicy(
        default="frozen",
        overrides=(("w_idx_q", "adamw"), ("w_idx_k", "adamw"), ("idx_k_ln_w", "adamw"), ("idx_k_ln_b", "adamw"), ("w_idx_w", "adamw"),),
    )
    return Dsv32Dims(
        opt_policy=cfg.opt_policy if cfg.sparse_mode else warmup_policy,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        q_lora_rank=cfg.q_lora_rank, kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_dim=cfg.qk_nope_dim, qk_rope_dim=cfg.qk_rope_dim,
        v_head_dim=cfg.v_head_dim,
        d_ff=cfg.d_ff_dense, first_k_dense=cfg.first_k_dense,
        vocab_size=cfg.vocab_size,
        tokens=cfg.tokens, seq_len=cfg.seq_len, rope_base=cfg.rope_base,
        dtypes=getattr(cfg, "dtypes", None) or _DSV32_DTYPES,
        seq_lens=getattr(cfg, "seq_lens", None),
        moe=moe_spec_of(cfg),
        index_n_heads=cfg.index_n_heads, index_head_dim=cfg.index_head_dim,
        index_topk=cfg.index_topk, sparse_mode=cfg.sparse_mode,
        train_indexer=cfg.train_indexer,
    )


def _kind_specs(cfg: ShapedDsv32Config, hw: ShapedHardware) -> dict[str, LayerKindSpec]:
    dims = dims_of_dsv32(cfg)
    t, d, seq, h = cfg.tokens, cfg.d_model, cfg.seq_len, cfg.n_heads
    qk = cfg.qk_head_dim
    sbar = seq / 2.0
    k_eff = min(cfg.index_topk, sbar)
    mla_active = (
        d * cfg.q_lora_rank + cfg.q_lora_rank * h * qk
        + d * (cfg.kv_lora_rank + cfg.qk_rope_dim)
        + cfg.kv_lora_rank * h * (cfg.qk_nope_dim + cfg.v_head_dim)
        + h * cfg.v_head_dim * d
        + cfg.q_lora_rank * cfg.index_n_heads * cfg.index_head_dim
        + d * cfg.index_head_dim + d * cfg.index_n_heads
    )
    # DSA: indexer scores over the causal prefix + SPARSE core over k_eff
    # (dense warm-up: the core runs the FULL prefix like dsv3, and the
    # bwd's KL target sweeps the full prefix too — no dsa_idx bytes)
    idx_flops = 2.0 * t * sbar * cfg.index_n_heads * cfg.index_head_dim
    core = k_eff if cfg.sparse_mode else sbar
    attn_flops = 2.0 * t * core * h * qk * 2.0 + idx_flops
    attn_bytes = BF16 * t * 4 * h * qk + (
        4.0 * t * cfg.index_topk if cfg.sparse_mode else 0.0)

    def spec(prefix, wl, cl, ffn_active, extra_traffic, meta_bytes=0):
        total_params = sum(int(math.prod(fl.shape)) for fl in wl.fields)
        mm_flops = 2.0 * t * (mla_active + ffn_active)
        mm_bytes = BF16 * (total_params + 4 * t * d) + extra_traffic
        fwd = hw.matmul_us(mm_flops, mm_bytes) + hw.attn_us(attn_flops, attn_bytes)
        # bwd re-runs the sparse core twice-ish + the indexer target pass
        bwd = hw.matmul_us(2.0 * mm_flops, 2.0 * mm_bytes) \
            + hw.attn_us(3.0 * attn_flops, 2.0 * attn_bytes)
        sub_fwd = [
            {"kind": "roofline", "name": f"{prefix}_matmuls", "flops": int(mm_flops),
             "memory_bytes": int(mm_bytes), "efficiency": "matmul"},
            {"kind": "roofline", "name": "dsa_attention", "flops": int(attn_flops),
             "memory_bytes": int(attn_bytes), "efficiency": "attention"},
        ]
        sub_bwd = [
            {"kind": "roofline", "name": f"{prefix}_matmuls_bwd", "flops": int(2 * mm_flops),
             "memory_bytes": int(2 * mm_bytes), "efficiency": "matmul"},
            {"kind": "roofline", "name": "dsa_attention_bwd", "flops": int(3 * attn_flops),
             "memory_bytes": int(2 * attn_bytes), "efficiency": "attention"},
        ]
        return LayerKindSpec(
            key_prefix=prefix,
            w_bytes=wl.total_bytes,
            a_bytes=cl.total_bytes,
            meta_bytes=meta_bytes,
            fwd_us=fwd, bwd_us=bwd, recompute_us=fwd,
            optimizer_us=hw.mem_us(BF16 * 7.0 * total_params),
            fwd_subops=sub_fwd, bwd_subops=sub_bwd, recompute_subops=list(sub_fwd),
            optimizer_subops=[{"kind": "roofline", "name": "adamw", "flops": 0,
                               "memory_bytes": int(BF16 * 7 * total_params),
                               "efficiency": "memory"}],
        )

    dense = spec(
        "dsadense", dsv32_dense_weight_layout(dims), dsv32_dense_context_layout(dims),
        3 * cfg.d_ff_dense * d, 0.0,
        meta_bytes=dsv32_meta_layout(dims, "dense").total_bytes,
    )
    f, fs, k = cfg.d_ff_expert, cfg.d_ff_shared, cfg.top_k
    moe_active = (
        d * cfg.n_experts + k * 3 * f * d
        + cfg.n_shared_experts * 3 * fs * d
    )
    moe_traffic = BF16 * t * k * (3 * f + 2 * d)
    moe = spec(
        "dsamoe", dsv32_moe_weight_layout(dims), dsv32_moe_context_layout(dims),
        moe_active, moe_traffic,
        meta_bytes=dsv32_meta_layout(dims, "moe").total_bytes,
    )
    return {"dense": dense, "moe": moe}


def build_shaped_dsv32(
    cfg: ShapedDsv32Config,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels=None,
    name: str | None = None,
):
    hw = hw or ShapedHardware()
    dims = dims_of_dsv32(cfg)
    # each layer's metadata is self-contained (its own M object); no
    # cross-layer sharing — glm52 is the family that passes meta_shared
    return build_shaped_program(
        cfg, hw=hw, family="dsv32-shaped",
        kinds=_kind_specs(cfg, hw), kind_of=dims.kind_of,
        fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=name,
    )


_WEIGHT_BUILDERS = {"dense": dsv32_dense_weight_layout, "moe": dsv32_moe_weight_layout}


def family_layouts(cfg: ShapedDsv32Config) -> tuple[Dsv32Dims, FamilyLayouts]:
    dims = dims_of_dsv32(cfg)
    ctx = {
        "dense": dsv32_dense_context_layout(dims),
        "moe": dsv32_moe_context_layout(dims),
    }
    return dims, FamilyLayouts(
        n_layers=cfg.n_layers,
        block_weight_at=lambda i: _WEIGHT_BUILDERS[dims.kind_of(i)](dims, layer=i),
        block_context_at=lambda i: ctx[dims.kind_of(i)],
        embed=embed_weight_layout(dims),
        head=head_weight_layout(dims),
        init_specials={
            "w_router_bias": lambda n, gen: torch.zeros(n),
            # LayerNorm affine init: weight ones, bias zeros
            "idx_k_ln_w": lambda n, gen: torch.ones(n),
            "idx_k_ln_b": lambda n, gen: torch.zeros(n),
        },
        block_meta_at=lambda i: dsv32_meta_layout(dims, dims.kind_of(i)),
    )


def lower_dsv32(
    cfg: ShapedDsv32Config,
    *,
    hw: ShapedHardware | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    fast_memory_capacity: int | None = None,
) -> Program:
    dims, fl = family_layouts(cfg)
    if not cfg.sparse_mode and not cfg.train_indexer:
        raise ValueError("dense warm-up trains ONLY the indexer; "
                         "train_indexer=False there trains nothing")
    if dims.moe.is_partial:
        raise NotImplementedError(
            "partial expert ownership (expert_ids) is accounting-only in v1"
        )
    shaped = build_shaped_dsv32(
        cfg, hw=hw, recompute_levels=recompute_levels, fast_memory_capacity=fast_memory_capacity,
    )
    return apply_exact_sizes(shaped, "dsv32-exact",
                             size_of=size_of_factory(dims, fl))


def initial_values_dsv32(program: Program, cfg: ShapedDsv32Config, backend, *, seed: int = 0):
    dims, fl = family_layouts(cfg)
    return initial_values_from_layouts(program, dims, fl, backend, seed=seed)
