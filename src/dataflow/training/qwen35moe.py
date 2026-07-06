"""Qwen3.5-MoE family: config + declarations over the generic machinery.

The dense qwen35 hybrid (3x GatedDeltaNet : 1x gated attention) with every
layer's dense SwiGLU replaced by the routed MoE tail: E=256 top-8 F=512 +
ONE sigmoid-gated shared expert (F_s=512), topk_then_softmax
(norm_topk_prob=true), aux alpha=0.001, vocab 248320, untied. ~34.7B
params at the faithful 35B-A3B shape.

HOST-RAM REALITY (docs/notes/moe-design.md par 8): the full 35B needs
~277 GB pinned W+dW+O — this box (188 GB) cannot run it; the config exists
for lowering/planning validation + tiny-scale correctness. Perf rows use
``qwen35moe_20l`` (Shein-confirmed): 20 layers (15 lin + 5 full), E=256,
everything else stock — ~17.8B params, ~143 GB pinned, recompute-dominant
plans expected, and it preserves the mechanisms under stress (E=256
router, ~1.7 GB W_i / ~3.4 GB O_i objects, hybrid 3:1, shared expert,
248k-vocab head).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from dataflow.core import Program
from dataflow.tasks.layouts import (
    DTypePolicy,
    Qwen35MoeDims,
    embed_weight_layout,
    head_weight_layout,
    qwen35moe_attn_context_layout,
    qwen35moe_attn_weight_layout,
    qwen35moe_lin_context_layout,
    qwen35moe_lin_weight_layout,
)
from dataflow.tasks.moe.spec import MoESpec

from .lowering import FamilyLayouts, apply_exact_sizes, initial_values_from_layouts, size_of_factory
from .qwen35 import _a_log_init, _dt_bias_init
from .shaped_program import BF16, LayerKindSpec, ShapedHardware, build_shaped_program


@dataclass(frozen=True)
class ShapedQwen35MoeConfig:
    n_layers: int = 40
    d_model: int = 2048
    full_attention_interval: int = 4
    # full-attention sub-block (gated, per-head qk-norm, partial rope)
    n_heads: int = 16
    n_kv_heads: int = 2
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    # linear-attention sub-block (GatedDeltaNet, same as dense qwen35)
    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    conv_kernel: int = 4
    # MoE FFN (every layer): routed + one sigmoid-gated shared expert
    n_experts: int = 256
    top_k: int = 8
    d_ff_expert: int = 512
    n_shared_experts: int = 1
    d_ff_shared: int = 512
    routing_mode: str = "topk_then_softmax"   # norm_topk_prob=true
    aux_coef: float = 0.001
    # shared
    vocab_size: int = 248_320
    seq_len: int = 4096
    batch: int = 1
    grad_accum_rounds: int = 1
    num_steps: int = 1
    optimizer_placement: str = "interleaved"
    tied_embeddings: bool = False  # the 35B is untied; family supports untied only
    rope_base: float = 10_000_000.0
    dtypes: DTypePolicy = DTypePolicy()
    seq_lens: tuple[int, ...] | None = None

    @property
    def tokens(self) -> int:
        if self.seq_lens is not None:
            return sum(self.seq_lens)
        return self.seq_len * self.batch

    @property
    def d_ff(self) -> int:  # metadata consumers; per-expert width
        return self.d_ff_expert

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    # -- parameter counts (duck-typed for the shared chain builder) -----------
    @property
    def block_params(self) -> int:
        # lin-kind count (30/40 layers); roofline seed only — per-kind specs
        # carry layout-exact sizes
        d, f, fs = self.d_model, self.d_ff_expert, self.d_ff_shared
        dims = dims_of_qwen35moe(self)
        attn = (
            d * dims.qkvz_dim + d * dims.ba_dim
            + dims.conv_dim * self.conv_kernel + dims.value_dim * d
        )
        moe = (
            d * self.n_experts + self.n_experts * 3 * f * d
            + self.n_shared_experts * (d + 3 * fs * d)
        )
        return attn + moe + 2 * d

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def head_params(self) -> int:
        return self.vocab_size * self.d_model

    @classmethod
    def tiny(cls) -> "ShapedQwen35MoeConfig":
        return cls(
            n_layers=4, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
            num_k_heads=2, num_v_heads=4, head_k_dim=32, head_v_dim=32,
            n_experts=8, top_k=2, d_ff_expert=128, d_ff_shared=128,
            vocab_size=512, seq_len=128, batch=1,
        )

    @classmethod
    def qwen35moe_35b(cls, *, seq_len: int = 4096, batch: int = 1,
                      grad_accum_rounds: int = 1, num_steps: int = 1) -> "ShapedQwen35MoeConfig":
        """The faithful 35B-A3B. NOT runnable on a 188 GB-host box
        (~277 GB pinned); lowering/planning validation only."""
        return cls(seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)

    @classmethod
    def qwen35moe_20l(cls, *, seq_len: int = 4096, batch: int = 1,
                      grad_accum_rounds: int = 1, num_steps: int = 1) -> "ShapedQwen35MoeConfig":
        """The perf-rows config: 20L (15 lin + 5 full), stock otherwise —
        ~17.8B params, ~143 GB pinned (fits, near the ceiling)."""
        return cls(n_layers=20, seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)


def moe_spec_of(cfg: ShapedQwen35MoeConfig) -> MoESpec:
    return MoESpec(
        n_experts=cfg.n_experts, top_k=cfg.top_k, d_ff_expert=cfg.d_ff_expert,
        routing_mode=cfg.routing_mode, aux_coef=cfg.aux_coef,
        n_shared_experts=cfg.n_shared_experts, d_ff_shared=cfg.d_ff_shared,
    )


def dims_of_qwen35moe(cfg: ShapedQwen35MoeConfig) -> Qwen35MoeDims:
    return Qwen35MoeDims(
        d_model=cfg.d_model, n_layers=cfg.n_layers,
        full_attention_interval=cfg.full_attention_interval,
        n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        partial_rotary_factor=cfg.partial_rotary_factor,
        num_k_heads=cfg.num_k_heads, num_v_heads=cfg.num_v_heads,
        head_k_dim=cfg.head_k_dim, head_v_dim=cfg.head_v_dim,
        conv_kernel=cfg.conv_kernel, d_ff=cfg.d_ff_expert,
        vocab_size=cfg.vocab_size,
        tokens=cfg.tokens, seq_len=cfg.seq_len, rope_base=cfg.rope_base,
        dtypes=getattr(cfg, "dtypes", None) or DTypePolicy(),
        seq_lens=getattr(cfg, "seq_lens", None),
        moe=moe_spec_of(cfg),
    )


def _kind_specs(cfg: ShapedQwen35MoeConfig, hw: ShapedHardware) -> dict[str, LayerKindSpec]:
    """Two LayerKindSpecs. MoE roofline convention: FLOPs from ACTIVE
    params (top-k + shared), weight BYTES from the FULL expert stack."""
    dims = dims_of_qwen35moe(cfg)
    t, d, seq = cfg.tokens, cfg.d_model, cfg.seq_len
    f, fs, k = cfg.d_ff_expert, cfg.d_ff_shared, cfg.top_k
    moe_active = (
        d * cfg.n_experts + k * 3 * f * d
        + cfg.n_shared_experts * (d + 3 * fs * d)
    )
    moe_traffic = BF16 * t * k * (3 * f + 2 * d)  # permuted dispatch/combine

    def spec(prefix, wl, cl, attn_active, attn_flops, attn_bytes, extra_mem_bytes):
        total_params = sum(int(math.prod(fl.shape)) for fl in wl.fields)
        mm_flops = 2.0 * t * (attn_active + moe_active)
        mm_bytes = BF16 * (total_params + 4 * t * d) + moe_traffic
        fwd = hw.matmul_us(mm_flops, mm_bytes) + hw.attn_us(attn_flops, attn_bytes) \
            + hw.mem_us(extra_mem_bytes)
        bwd = hw.matmul_us(2.0 * mm_flops, 2.0 * mm_bytes) \
            + hw.attn_us(2.5 * attn_flops, 2.0 * attn_bytes) \
            + hw.mem_us(2.0 * extra_mem_bytes)
        sub_fwd = [
            {"kind": "roofline", "name": f"{prefix}_matmuls", "flops": int(mm_flops),
             "memory_bytes": int(mm_bytes), "efficiency": "matmul"},
            {"kind": "roofline", "name": f"{prefix}_mix", "flops": int(attn_flops),
             "memory_bytes": int(attn_bytes + extra_mem_bytes), "efficiency": "attention"},
        ]
        sub_bwd = [
            {"kind": "roofline", "name": f"{prefix}_matmuls_bwd", "flops": int(2 * mm_flops),
             "memory_bytes": int(2 * mm_bytes), "efficiency": "matmul"},
            {"kind": "roofline", "name": f"{prefix}_mix_bwd", "flops": int(2.5 * attn_flops),
             "memory_bytes": int(2 * (attn_bytes + extra_mem_bytes)), "efficiency": "attention"},
        ]
        return LayerKindSpec(
            key_prefix=prefix,
            w_bytes=wl.total_bytes,
            a_bytes=cl.total_bytes,
            fwd_us=fwd, bwd_us=bwd, recompute_us=fwd,
            optimizer_us=hw.mem_us(BF16 * 7.0 * total_params),
            fwd_subops=sub_fwd, bwd_subops=sub_bwd, recompute_subops=sub_fwd,
            optimizer_subops=[{"kind": "roofline", "name": "adamw", "flops": 0,
                               "memory_bytes": int(BF16 * 7 * total_params),
                               "efficiency": "memory"}],
        )

    lin_attn = d * dims.qkvz_dim + d * dims.ba_dim + dims.value_dim * d
    lin_scan_flops = 2.0 * t * dims.num_v_heads * dims.head_k_dim * dims.head_v_dim * 2
    lin_mem = BF16 * t * (2 * dims.conv_dim + 2 * dims.value_dim)
    lin = spec("linmoe", qwen35moe_lin_weight_layout(dims), qwen35moe_lin_context_layout(dims),
               lin_attn, lin_scan_flops, BF16 * t * 2 * dims.value_dim, lin_mem)

    full_attn = d * 2 * dims.attn_dim + 2 * d * dims.kv_dim + dims.attn_dim * d
    attn_flops = 2.0 * t * seq * dims.attn_dim
    attn_bytes = BF16 * t * (2 * dims.attn_dim + 2 * dims.kv_dim)
    full = spec("gattnmoe", qwen35moe_attn_weight_layout(dims), qwen35moe_attn_context_layout(dims),
                full_attn, attn_flops, attn_bytes, 0.0)

    return {"lin": lin, "full": full}


def build_shaped_qwen35moe(
    cfg: ShapedQwen35MoeConfig,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels=None,
    name: str | None = None,
):
    hw = hw or ShapedHardware()
    dims = dims_of_qwen35moe(cfg)
    label = name or (
        f"qwen35moe-shaped-{cfg.n_layers}L-d{cfg.d_model}-s{cfg.seq_len}-b{cfg.batch}"
        f"-r{cfg.grad_accum_rounds}-steps{cfg.num_steps}"
    )
    return build_shaped_program(
        cfg, hw=hw, family="qwen35moe-shaped",
        fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=label,
        kinds=_kind_specs(cfg, hw), kind_of=dims.kind_of,
    )


_WEIGHT_BUILDERS = {"lin": qwen35moe_lin_weight_layout, "full": qwen35moe_attn_weight_layout}


def family_layouts(cfg: ShapedQwen35MoeConfig) -> tuple[Qwen35MoeDims, FamilyLayouts]:
    dims = dims_of_qwen35moe(cfg)
    ctx = {
        "lin": qwen35moe_lin_context_layout(dims),
        "full": qwen35moe_attn_context_layout(dims),
    }
    return dims, FamilyLayouts(
        n_layers=cfg.n_layers,
        block_weight_at=lambda i: _WEIGHT_BUILDERS[dims.kind_of(i)](dims, layer=i),
        block_context_at=lambda i: ctx[dims.kind_of(i)],
        embed=embed_weight_layout(dims),
        head=head_weight_layout(dims),
        init_specials={"A_log": _a_log_init, "dt_bias": _dt_bias_init},
    )


def lower_qwen35moe(
    cfg: ShapedQwen35MoeConfig,
    *,
    hw: ShapedHardware | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    fast_memory_capacity: int | None = None,
) -> Program:
    if cfg.tied_embeddings:
        raise NotImplementedError("qwen35moe is untied (the 35B config)")
    if cfg.n_shared_experts != 1:
        raise ValueError("qwen35moe carries exactly one shared expert")
    dims, fl = family_layouts(cfg)
    if dims.moe.is_partial:
        raise NotImplementedError(
            "partial expert ownership (expert_ids) is accounting-only in v1 — "
            "program lowering needs the multi-rank runtime (EP)"
        )
    shaped = build_shaped_qwen35moe(
        cfg, hw=hw, recompute_levels=recompute_levels, fast_memory_capacity=fast_memory_capacity,
    )
    return apply_exact_sizes(shaped, "qwen35moe-exact-v1", size_of=size_of_factory(dims, fl))


def initial_values_qwen35moe(program: Program, cfg: ShapedQwen35MoeConfig, backend, *, seed: int = 0):
    dims, fl = family_layouts(cfg)
    return initial_values_from_layouts(program, dims, fl, backend, seed=seed)
