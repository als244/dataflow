"""GLM-5.2 IndexShare family: config + declarations + the S/P grammar.

dsv32's DSA backbone with CROSS-LAYER INDEX REUSE (arXiv 2603.12201, the
IndexCache paper; zai-org/GLM-5.2): "full" layers run their own lightning
indexer; "shared" layers carry NO indexer weights and reuse the nearest
preceding full layer's top-k selection. The pattern is greedy-searched
upstream, so it is an explicit per-layer tuple here, never a formula.

The selection crosses layer boundaries as a FIRST-CLASS OBJECT (Shein's
"explicit obj" call):

- ``S_{s}_{r}_{g}`` (t, k) int32 — emitted by the leader's fwd (extra
  output), consumed by follower fwds, ALL member recomputes (the leader
  included: nobody re-selects), and ALL member bwds. glm52 saved-ctx is
  dsv3-shaped for every kind — no dsa_idx field anywhere.
- ``P_{s}_{r}_{g}`` (t, k) fp32 — the multi-layer distillation target
  accumulator (paper: L^I_multi = mean_j KL(p^(l+j) || sigma^(l));
  Proposition 1: gradient == centroid alignment, dI = sigma - P/N).
  Backward visits members in reverse layer order, so the LAST member's
  bwd CREATES P (its own head-summed, L1-normalized target gathered at
  S), intermediate members accumulate via the existing mutates grammar,
  and the leader's bwd — which runs LAST — consumes P, adds its own
  target, and chains sigma - P/N through its indexer weights. The order
  is fixed by the chain: deterministic and plan-invariant. Singleton
  groups (leaders serving nobody) get no P object and degenerate to
  dsv32's per-layer KL exactly.

Both objects are injected by a lowering POST-PASS over the shaped
program, so shaped_program stays family-generic; recompute variants are
covered by construction (the planner re-lowers per level assignment and
the post-pass sees whatever rc tasks exist).

v1 scope: sparse mode only. The paper's dense warm-up (1k steps, frozen
main, L^I_multi with FULL-PREFIX targets) is M-I2b: same machinery with
P_warm (t, seq_len) fp32 — block-diagonal, each row needs only its own
sequence's columns. ``sparse_mode=False`` raises until then.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping

import torch

from dataflow.tasks.optim import OptPolicy
from dataflow.core import Program
from dataflow.tasks.layouts import (
    DTypePolicy,
    Glm52Dims,
    glm52_meta_layout,
    ParamDTypes,
    dsv3_dense_context_layout,
    dsv3_moe_context_layout,
    dsv3_moe_weight_layout,
    dsv32_dense_weight_layout,
    dsv32_moe_weight_layout,
    embed_weight_layout,
    head_weight_layout,
)
from dataflow.tasks.modules.moe.spec import MoESpec

from ..lowering import FamilyLayouts, apply_exact_sizes, initial_values_from_layouts, size_of_factory
from ..shaped_program import (
    BF16,
    LayerKindSpec,
    MetaShare,
    ShapedHardware,
    build_shaped_program,
)

_GLM52_DTYPES = DTypePolicy(overrides=(
    ("w_router_bias", ParamDTypes("fp32", "fp32", "fp32")),
    ("w_idx_w", ParamDTypes("fp32", "fp32", "fp32")),
))


def _pattern(n_layers: int, first_k_full: int, period: int = 4) -> tuple[str, ...]:
    """F * first_k_full then [F S S S...] repeating — helper for presets
    whose upstream pattern IS periodic; real configs pass the raw list."""
    out = ["full"] * first_k_full
    while len(out) < n_layers:
        out.append("full")
        for _ in range(period - 1):
            if len(out) < n_layers:
                out.append("shared")
    return tuple(out[:n_layers])


# GLM-5.2's greedy-searched pattern, verbatim from config.json indexer_types
_GLM52_TYPES = tuple(
    ["full"] * 3 + ["shared"] * 3 + ["full", "shared", "shared", "shared"] * 18
)
assert len(_GLM52_TYPES) == 78


@dataclass(frozen=True)
class ShapedGlm52Config:
    n_layers: int = 78
    d_model: int = 6144
    n_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_dim: int = 192
    qk_rope_dim: int = 64
    v_head_dim: int = 256
    d_ff_dense: int = 12288
    first_k_dense: int = 3
    n_experts: int = 256
    top_k: int = 8
    d_ff_expert: int = 2048
    n_group: int = 1
    topk_group: int = 1
    routed_scaling: float = 2.5
    bias_update_speed: float = 0.001
    aux_coef: float = 1e-4
    n_shared_experts: int = 1
    d_ff_shared: int = 2048
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    indexer_types: tuple[str, ...] = _GLM52_TYPES
    sparse_mode: bool = True
    train_indexer: bool = True
    vocab_size: int = 154_880
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
    rope_base: float = 8_000_000.0
    dtypes: DTypePolicy = field(default_factory=lambda: _GLM52_DTYPES)
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
        # moe-LEADER block (the largest kind) — metadata only
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
    def tiny(cls) -> "ShapedGlm52Config":
        # 6L, first_k=1: roles cover dense-leader, an unequal group, a
        # leader serving nobody, and a trailing short group
        return cls(
            n_layers=6, d_model=128, n_heads=4, q_lora_rank=64,
            kv_lora_rank=32, qk_nope_dim=16, qk_rope_dim=8, v_head_dim=16,
            d_ff_dense=256, first_k_dense=1,
            n_experts=8, top_k=2, d_ff_expert=32, n_group=4, topk_group=2,
            d_ff_shared=32,
            index_n_heads=8, index_head_dim=32, index_topk=24,
            indexer_types=("full", "full", "shared", "shared", "full", "shared"),
            vocab_size=512, seq_len=128, batch=1,
        )

    @classmethod
    def glm52_mini(cls, *, seq_len: int = 4096, batch: int = 1,
                   grad_accum_rounds: int = 1, num_steps: int = 1,
                   sparse_mode: bool = True,
                   ) -> "ShapedGlm52Config":
        return cls(
            n_layers=18, d_model=2048, n_heads=16, q_lora_rank=512,
            kv_lora_rank=256, qk_nope_dim=64, qk_rope_dim=32, v_head_dim=64,
            d_ff_dense=8192, first_k_dense=2,
            n_experts=128, top_k=8, d_ff_expert=1024, n_group=8, topk_group=4,
            d_ff_shared=1024,
            index_n_heads=8, index_head_dim=64, index_topk=1024,
            indexer_types=_pattern(18, first_k_full=2),
            vocab_size=129_280, rope_base=10_000.0,
            seq_len=seq_len, batch=batch,
            grad_accum_rounds=grad_accum_rounds, num_steps=num_steps, sparse_mode=sparse_mode,
        )

    @classmethod
    def glm52_mini_warmup(cls, *, seq_len: int = 4096, batch: int = 4,
                          grad_accum_rounds: int = 4, num_steps: int = 1,
                          ) -> "ShapedGlm52Config":
        """The paper's dense warm-up stage as a first-class preset: full
        causal attention, main model frozen (no dW/O beyond the leaders'
        indexer fields), leaders train on group-averaged FULL-PREFIX
        targets. Unique task graph vs the sparse preset: NO head/CE (the
        loss object carries the group-KL objective), no dy chain,
        no embed_bwd, follower/embed/head optimizer tasks pruned."""
        return cls.glm52_mini(seq_len=seq_len, batch=batch,
                              grad_accum_rounds=grad_accum_rounds,
                              num_steps=num_steps, sparse_mode=False)

    @classmethod
    def glm52(cls, *, seq_len: int = 4096, batch: int = 1,
              grad_accum_rounds: int = 1, num_steps: int = 1,
              ) -> "ShapedGlm52Config":
        return cls(seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)


def moe_spec_of(cfg: ShapedGlm52Config) -> MoESpec:
    return MoESpec(
        n_experts=cfg.n_experts, top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert,
        routing_mode="sigmoid_noaux_tc", aux_coef=cfg.aux_coef,
        n_shared_experts=cfg.n_shared_experts, d_ff_shared=cfg.d_ff_shared,
        shared_gate=False,
        n_group=cfg.n_group, topk_group=cfg.topk_group,
        routed_scaling=cfg.routed_scaling,
        bias_update_speed=cfg.bias_update_speed,
    )


def dims_of_glm52(cfg: ShapedGlm52Config) -> Glm52Dims:
    types = tuple(cfg.indexer_types)
    if len(types) != cfg.n_layers:
        raise ValueError(f"indexer_types has {len(types)} entries for "
                         f"{cfg.n_layers} layers")
    if any(r not in ("full", "shared") for r in types):
        raise ValueError(f"indexer_types entries must be full|shared: {types}")
    if types[0] != "full":
        raise ValueError("layer 0 must be 'full' (it seeds the indices — "
                         "paper: 'the first layer is always F')")
    for i in range(cfg.first_k_dense):
        if types[i] != "full":
            raise NotImplementedError(
                "dense-FFN shared layers are not supported (GLM-5.2's dense "
                "layers are all full); needed only if a real config appears"
            )
    # dense warm-up: freezing is an OPTIMIZER POLICY — every field
    # defaults to "frozen" (zero grad storage, zero opt state; the
    # lowering prunes empty dW/O objects and purposeless optimizer/
    # embed_bwd tasks) and only the indexer trains, with adamw.
    warmup_policy = OptPolicy(
        default="frozen",
        overrides=(("w_idx_q", "adamw"), ("w_idx_k", "adamw"), ("idx_k_ln_w", "adamw"), ("idx_k_ln_b", "adamw"), ("w_idx_w", "adamw"),),
    )
    return Glm52Dims(
        opt_policy=cfg.opt_policy if cfg.sparse_mode else warmup_policy,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        q_lora_rank=cfg.q_lora_rank, kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_dim=cfg.qk_nope_dim, qk_rope_dim=cfg.qk_rope_dim,
        v_head_dim=cfg.v_head_dim,
        d_ff=cfg.d_ff_dense, first_k_dense=cfg.first_k_dense,
        vocab_size=cfg.vocab_size,
        tokens=cfg.tokens, seq_len=cfg.seq_len, rope_base=cfg.rope_base,
        dtypes=getattr(cfg, "dtypes", None) or _GLM52_DTYPES,
        seq_lens=getattr(cfg, "seq_lens", None),
        moe=moe_spec_of(cfg),
        index_n_heads=cfg.index_n_heads, index_head_dim=cfg.index_head_dim,
        index_topk=cfg.index_topk, sparse_mode=cfg.sparse_mode,
        train_indexer=cfg.train_indexer,
        indexer_types=types,
    )


_LEADER_WEIGHTS = {"gdl": dsv32_dense_weight_layout, "gml": dsv32_moe_weight_layout}


def _weight_layout_for(dims: Glm52Dims, kind: str):
    if kind == "gmf":
        return dsv3_moe_weight_layout(dims)
    return _LEADER_WEIGHTS[kind](dims)


def _ctx_layout_for(dims: Glm52Dims, kind: str):
    # selection-object grammar: no dsa_idx anywhere (S object) and the
    # routing pack lives in the per-layer SEL object (moe kinds)
    from dataflow.tasks.modules.moe.spec import moe_context_specs
    from dataflow.tasks.layouts import PackedLayout, _dsv3_attn_ctx_specs

    if kind == "gdl":
        return dsv3_dense_context_layout(dims)
    return PackedLayout.build(
        _dsv3_attn_ctx_specs(dims) + moe_context_specs(dims, dims.moe, meta=True)
    )


def _kind_specs(cfg: ShapedGlm52Config, hw: ShapedHardware) -> dict[str, LayerKindSpec]:
    dims = dims_of_glm52(cfg)
    t, d, seq, h = cfg.tokens, cfg.d_model, cfg.seq_len, cfg.n_heads
    qk = cfg.qk_head_dim
    sbar = seq / 2.0
    k_eff = min(cfg.index_topk, sbar) if cfg.sparse_mode else sbar
    mla_active = (
        d * cfg.q_lora_rank + cfg.q_lora_rank * h * qk
        + d * (cfg.kv_lora_rank + cfg.qk_rope_dim)
        + cfg.kv_lora_rank * h * (cfg.qk_nope_dim + cfg.v_head_dim)
        + h * cfg.v_head_dim * d
    )
    idx_active = (
        cfg.q_lora_rank * cfg.index_n_heads * cfg.index_head_dim
        + d * cfg.index_head_dim + d * cfg.index_n_heads
    )
    idx_flops = 2.0 * t * sbar * cfg.index_n_heads * cfg.index_head_dim
    core_flops = 2.0 * t * k_eff * h * qk * 2.0
    s_bytes = 4.0 * t * cfg.index_topk if cfg.sparse_mode else 0.0

    def spec(prefix, leader, wl, cl, ffn_active, extra_traffic,
             meta_bytes=0):
        total_params = sum(int(math.prod(fl.shape)) for fl in wl.fields)
        attn_flops = core_flops + (idx_flops if leader else 0.0)
        attn_bytes = BF16 * t * 4 * h * qk + s_bytes
        mm_active = mla_active + ffn_active + (idx_active if leader else 0)
        mm_flops = 2.0 * t * mm_active
        mm_bytes = BF16 * (total_params + 4 * t * d) + extra_traffic
        fwd = hw.matmul_us(mm_flops, mm_bytes) + hw.attn_us(attn_flops, attn_bytes)
        # bwd: sparse core x~2.5; leaders add the KL passes; members add
        # the P contribution (one probs pass + gather)
        bwd_attn = 2.5 * core_flops + idx_flops + (2.0 * idx_flops if leader else 0.0)
        bwd = hw.matmul_us(2.0 * mm_flops, 2.0 * mm_bytes) \
            + hw.attn_us(bwd_attn, 2.0 * attn_bytes + 4.0 * t * cfg.index_topk)
        sub_fwd = [
            {"kind": "roofline", "name": f"{prefix}_matmuls", "flops": int(mm_flops),
             "memory_bytes": int(mm_bytes), "efficiency": "matmul"},
            {"kind": "roofline", "name": "dsa_attention", "flops": int(attn_flops),
             "memory_bytes": int(attn_bytes), "efficiency": "attention"},
        ]
        sub_bwd = [
            {"kind": "roofline", "name": f"{prefix}_matmuls_bwd", "flops": int(2 * mm_flops),
             "memory_bytes": int(2 * mm_bytes), "efficiency": "matmul"},
            {"kind": "roofline", "name": "dsa_attention_bwd", "flops": int(bwd_attn),
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

    f, fs, k = cfg.d_ff_expert, cfg.d_ff_shared, cfg.top_k
    moe_active = (
        d * cfg.n_experts + k * 3 * f * d + cfg.n_shared_experts * 3 * fs * d
    )
    moe_traffic = BF16 * t * k * (3 * f + 2 * d)
    return {
        "gdl": spec("gdl", True, _weight_layout_for(dims, "gdl"),
                    _ctx_layout_for(dims, "gdl"), 3 * cfg.d_ff_dense * d, 0.0,
                    meta_bytes=glm52_meta_layout(dims, "gdl").total_bytes),
        "gml": spec("gml", True, _weight_layout_for(dims, "gml"),
                    _ctx_layout_for(dims, "gml"), moe_active, moe_traffic,
                    meta_bytes=glm52_meta_layout(dims, "gml").total_bytes),
        "gmf": spec("gmf", False, _weight_layout_for(dims, "gmf"),
                    _ctx_layout_for(dims, "gmf"), moe_active, moe_traffic,
                    meta_bytes=glm52_meta_layout(dims, "gmf").total_bytes),
    }


def _warmup_plan(dims, n_layers):
    """Dense warm-up as a FreezePlan: all-frozen-except-indexer policy
    (already injected on dims), indexer-KL objective, loss contributed
    by the LEADERS (followers deposit into dM instead)."""
    from ..freeze_plan import derive_freeze_plan

    contributors = tuple(i for i in range(n_layers)
                         if dims.leader_of(i) == i)
    return derive_freeze_plan(
        dims, n_layers,
        lambda i: [f.name for f in _weight_layout_for(dims, dims.kind_of(i)).fields],
        objective="indexer_kl", loss_contributors=contributors,
    )


def build_shaped_glm52(
    cfg: ShapedGlm52Config,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels=None,
    name: str | None = None,
):
    hw = hw or ShapedHardware()
    dims = dims_of_glm52(cfg)
    shares = [
        MetaShare(producer=ld, consumers=dims.group_members(ld)[1:],
                  # frozen indexer => no KL anywhere => no dM chain
                  grad_bytes=(0 if not dims.train_indexer
                              else 4 * dims.tokens * (
                                  dims.index_topk if dims.sparse_mode
                                  else cfg.seq_len)))
        for ld in dims.leaders()
        if len(dims.group_members(ld)) > 1
    ]
    return build_shaped_program(
        cfg, hw=hw, family="glm52-shaped",
        kinds=_kind_specs(cfg, hw), kind_of=dims.kind_of,
        fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=name,
        meta_shared=shares,
        freeze=(None if cfg.sparse_mode else _warmup_plan(dims, cfg.n_layers)),
    )


def family_layouts(cfg: ShapedGlm52Config) -> tuple[Glm52Dims, FamilyLayouts]:
    dims = dims_of_glm52(cfg)
    return dims, FamilyLayouts(
        n_layers=cfg.n_layers,
        block_weight_at=lambda i: _weight_layout_for(dims, dims.kind_of(i)),
        block_context_at=lambda i: _ctx_layout_for(dims, dims.kind_of(i)),
        embed=embed_weight_layout(dims),
        head=head_weight_layout(dims),
        init_specials={
            "w_router_bias": lambda n, gen: torch.zeros(n),
            "idx_k_ln_w": lambda n, gen: torch.ones(n),
            "idx_k_ln_b": lambda n, gen: torch.zeros(n),
        },
        block_meta_at=lambda i: glm52_meta_layout(dims, dims.kind_of(i)),
    )


def lower_glm52(
    cfg: ShapedGlm52Config,
    *,
    hw: ShapedHardware | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    fast_memory_capacity: int | None = None,
) -> Program:
    dims, fl = family_layouts(cfg)
    if dims.moe.is_partial:
        raise NotImplementedError(
            "partial expert ownership (expert_ids) is accounting-only in v1"
        )
    shaped = build_shaped_glm52(
        cfg, hw=hw, recompute_levels=recompute_levels,
        fast_memory_capacity=fast_memory_capacity,
    )
    base_size = size_of_factory(dims, fl)
    t_tokens = dims.tokens
    dm_cols = dims.index_topk if dims.sparse_mode else cfg.seq_len

    def size_of(oid: str):
        if oid.startswith("dM_"):
            return 4 * t_tokens * dm_cols
        return base_size(oid)

    return apply_exact_sizes(shaped, "glm52-exact", size_of=size_of)


def initial_values_glm52(program: Program, cfg: ShapedGlm52Config, backend, *, seed: int = 0):
    dims, fl = family_layouts(cfg)
    return initial_values_from_layouts(program, dims, fl, backend, seed=seed)
