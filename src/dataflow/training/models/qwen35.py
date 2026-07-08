"""Qwen3.5-dense family: config + declarations (hybrid DeltaNet + gated attention).

The first heterogeneous family: layers alternate between two block kinds
(``lin`` = Gated DeltaNet, ``full`` = gated GQA attention) on the model's
``full_attention_interval`` (Qwen3.5-9B: LLLF x8) — two ``LayerKindSpec``s
(sizes from the packed layouts, roofline cost seeds) into the generic
builder, and a per-kind ``FamilyLayouts`` into the generic lowering.
Embeddings follow the config: the 9B is UNTIED; the 2B-style tied variant
packs the head layout into the single W_embed (policy-addressed head.*).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from dataflow.core import Program
from dataflow.tasks.layouts import (
    DTypePolicy,
    Qwen35Dims,
    embed_weight_layout,
    head_weight_layout,
    qwen35_attn_context_layout,
    qwen35_attn_weight_layout,
    qwen35_lin_context_layout,
    qwen35_lin_weight_layout,
)
from ..lowering import FamilyLayouts, apply_exact_sizes, initial_values_from_layouts, size_of_factory
from ..shaped_program import LayerKindSpec, ShapedHardware, build_shaped_program


BF16 = 2


@dataclass(frozen=True)
class ShapedQwen35Config:
    n_layers: int = 32
    d_model: int = 4096
    full_attention_interval: int = 4
    # full-attention sub-block
    n_heads: int = 16
    n_kv_heads: int = 4
    head_dim: int = 256
    partial_rotary_factor: float = 0.25
    # linear-attention sub-block
    lin_k_heads: int = 16
    lin_v_heads: int = 32
    lin_k_head_dim: int = 128
    lin_v_head_dim: int = 128
    lin_conv_kernel: int = 4
    # shared
    d_ff: int = 12288
    vocab_size: int = 248_320
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
    # Qwen3.5-9B does NOT tie (config.json tie_word_embeddings: false;
    # untied ~8.96B params = the "9B"). The 2B DOES tie — tied stays a
    # supported config choice, exercised by the tiny_tied ladder tests.
    tied_embeddings: bool = False
    dtypes: DTypePolicy = DTypePolicy()
    # ragged packing: explicit per-round sequence lengths (sum = tokens
    # per round); None = uniform batch x seq_len
    seq_lens: tuple[int, ...] | None = None
    rope_base: float = 10_000_000.0

    @property
    def tokens(self) -> int:
        if self.seq_lens is not None:
            return sum(self.seq_lens)
        return self.seq_len * self.batch

    # the generic builder's embed/head/loss costs read these:
    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def head_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    # uniform-block properties the generic _Costs seeds require; the per-kind
    # specs override everything block-related, so these only seed embed/head/
    # loss/optimizer-embed costs and are otherwise unused.
    @property
    def block_params(self) -> int:
        return self.d_model * self.d_ff * 3 + 2 * self.d_model

    @property
    def saved_ctx_width(self) -> int:
        return 4 * self.d_model + 2 * self.d_ff

    @classmethod
    def tiny(cls) -> "ShapedQwen35Config":
        return cls(
            n_layers=4, d_model=256, full_attention_interval=4,
            n_heads=4, n_kv_heads=2, head_dim=64, partial_rotary_factor=0.25,
            lin_k_heads=2, lin_v_heads=4, lin_k_head_dim=32, lin_v_head_dim=32,
            lin_conv_kernel=4, d_ff=512, vocab_size=512, seq_len=128, batch=1,
        )

    @classmethod
    def tiny_tied(cls) -> "ShapedQwen35Config":
        """The 2B-style tied variant (one W_embed serves embed AND head)."""
        return replace(cls.tiny(), tied_embeddings=True)

    @classmethod
    def qwen35_9b(cls, *, seq_len: int = 4096, batch: int = 1,
                  grad_accum_rounds: int = 1, num_steps: int = 1) -> "ShapedQwen35Config":
        return cls(seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)


def dims_of_qwen35(cfg: ShapedQwen35Config) -> Qwen35Dims:
    return Qwen35Dims(
        opt_policy=cfg.opt_policy,
        d_model=cfg.d_model, n_layers=cfg.n_layers,
        full_attention_interval=cfg.full_attention_interval,
        n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        partial_rotary_factor=cfg.partial_rotary_factor,
        lin_k_heads=cfg.lin_k_heads, lin_v_heads=cfg.lin_v_heads,
        lin_k_head_dim=cfg.lin_k_head_dim, lin_v_head_dim=cfg.lin_v_head_dim,
        lin_conv_kernel=cfg.lin_conv_kernel, d_ff=cfg.d_ff, vocab_size=cfg.vocab_size,
        tokens=cfg.tokens, seq_len=cfg.seq_len, rope_base=cfg.rope_base,
        dtypes=getattr(cfg, "dtypes", None) or DTypePolicy(),
        seq_lens=getattr(cfg, "seq_lens", None),
    )


def _kind_specs(cfg: ShapedQwen35Config, hw: ShapedHardware) -> dict[str, LayerKindSpec]:
    """Two LayerKindSpecs with layout-exact sizes and roofline cost seeds."""
    import math

    dims = dims_of_qwen35(cfg)
    t, d, seq, ff = cfg.tokens, cfg.d_model, cfg.seq_len, cfg.d_ff

    def spec(prefix, wl, cl, mm_params, attn_flops, attn_bytes, extra_mem_bytes):
        mm_flops = 2.0 * t * mm_params
        mm_bytes = BF16 * (mm_params + 4 * t * d)
        fwd = hw.matmul_us(mm_flops, mm_bytes) + hw.attn_us(attn_flops, attn_bytes) \
            + hw.mem_us(extra_mem_bytes)
        bwd = hw.matmul_us(2.0 * mm_flops, 2.0 * mm_bytes) \
            + hw.attn_us(2.5 * attn_flops, 2.0 * attn_bytes) \
            + hw.mem_us(2.0 * extra_mem_bytes)
        w_bytes = wl.total_bytes
        params = sum(int(math.prod(f.shape)) for f in wl.fields)
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
        sub_opt = [{"kind": "roofline", "name": "adamw", "flops": 0,
                    "memory_bytes": int(BF16 * 7 * params), "efficiency": "memory"}]
        return LayerKindSpec(
            key_prefix=prefix,
            w_bytes=w_bytes,
            a_bytes=cl.total_bytes,
            fwd_us=fwd, bwd_us=bwd, recompute_us=fwd,
            optimizer_us=hw.mem_us(BF16 * 7.0 * params),
            fwd_subops=sub_fwd, bwd_subops=sub_bwd,
            recompute_subops=sub_fwd, optimizer_subops=sub_opt,
        )

    # linear-attn: projections dominate; delta-rule ~O(T * HV * K * V / 64)
    # chunked work seeded through the attention-efficiency term
    lin_mm = d * dims.qkvz_dim + d * dims.ba_dim + dims.value_dim * d + 3 * d * ff
    lin_scan_flops = 2.0 * t * dims.lin_v_heads * dims.lin_k_head_dim * dims.lin_v_head_dim * 2
    lin_mem = BF16 * t * (2 * dims.conv_dim + 2 * dims.value_dim)
    lin = spec("linattn", qwen35_lin_weight_layout(dims), qwen35_lin_context_layout(dims),
               lin_mm, lin_scan_flops, BF16 * t * 2 * dims.value_dim, lin_mem)

    # full-attn: causal flash over head_dim with GQA
    full_mm = d * 2 * dims.attn_dim + 2 * d * dims.kv_dim + dims.attn_dim * d + 3 * d * ff
    attn_flops = 2.0 * t * seq * dims.attn_dim
    attn_bytes = BF16 * t * (2 * dims.attn_dim + 2 * dims.kv_dim)
    full = spec("gattn", qwen35_attn_weight_layout(dims), qwen35_attn_context_layout(dims),
                full_mm, attn_flops, attn_bytes, 0.0)

    return {"lin": lin, "full": full}


def build_shaped_qwen35(
    cfg: ShapedQwen35Config,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels=None,
    name: str | None = None,
):
    hw = hw or ShapedHardware()
    dims = dims_of_qwen35(cfg)
    label = name or (
        f"qwen35-shaped-{cfg.n_layers}L-d{cfg.d_model}-s{cfg.seq_len}-b{cfg.batch}"
        f"-r{cfg.grad_accum_rounds}-steps{cfg.num_steps}"
    )
    from ..freeze_plan import derive_freeze_plan

    dims_fp, fl_fp = family_layouts(cfg)
    freeze_plan = derive_freeze_plan(
        dims_fp, cfg.n_layers,
        lambda i: [f.name for f in fl_fp.block_weight_at(i).fields],
        tied_embeddings=bool(getattr(cfg, "tied_embeddings", False)),
    )
    return build_shaped_program(
        cfg, hw=hw, family="qwen35-shaped",
        fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=label,
        kinds=_kind_specs(cfg, hw), kind_of=dims.kind_of,
        freeze=freeze_plan,
    )


_WEIGHT_BUILDERS = {
    "lin": qwen35_lin_weight_layout,
    "full": qwen35_attn_weight_layout,
}


def _a_log_init(n, gen):
    # decay magnitudes ~ U(1, 16) in log space (GDN convention)
    import torch

    return torch.empty(n).uniform_(1.0, 16.0, generator=gen).log()


def _dt_bias_init(n, gen):
    import torch

    return torch.zeros(n)


def family_layouts(cfg: ShapedQwen35Config):
    dims = dims_of_qwen35(cfg)
    ctx = {
        "lin": qwen35_lin_context_layout(dims),
        "full": qwen35_attn_context_layout(dims),
    }
    hl = head_weight_layout(dims)
    tied = cfg.tied_embeddings
    return dims, FamilyLayouts(
        n_layers=cfg.n_layers,
        block_weight_at=lambda i: _WEIGHT_BUILDERS[dims.kind_of(i)](dims, layer=i),
        block_context_at=lambda i: ctx[dims.kind_of(i)],
        embed=hl if tied else embed_weight_layout(dims),
        head=hl,
        embed_ns="head" if tied else "embed",
        init_specials={"A_log": _a_log_init, "dt_bias": _dt_bias_init},
    )


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
    dims, fl = family_layouts(cfg)
    return apply_exact_sizes(shaped, "qwen35-exact", size_of=size_of_factory(dims, fl))


def initial_values_qwen35(program: Program, cfg: ShapedQwen35Config, backend, *, seed: int = 0):
    dims, fl = family_layouts(cfg)
    return initial_values_from_layouts(program, dims, fl, backend, seed=seed)
