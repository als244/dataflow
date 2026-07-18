"""Qwen3-dense family: config + declarations over the generic machinery.

Same contract as every family. Qwen3-vs-llama differences that reach this
layer: qk-norm weights in the block parameter count and packed layout,
head_dim decoupled from d_model/n_heads (q projection is n_heads*head_dim
wide), rope theta 1e6, larger vocab, untied embed/head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from dataflow.core import Program
from dataflow_training.blocks.layouts import (
    DTypePolicy,
    Qwen3Dims,
    embed_weight_layout,
    head_weight_layout,
    qwen3_activation_layout,
    qwen3_weight_layout,
)
from ..lowering.emit import FamilyLayouts, LayerLayout, apply_exact_sizes, initial_values_from_layouts, size_of_factory
from ..lowering.shaped_program import ShapedHardware, build_shaped_program, roofline_block_kind_spec


@dataclass(frozen=True)
class ShapedQwen3Config:
    n_layers: int = 36
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    d_ff: int = 12288
    vocab_size: int = 151_936
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
    dtypes: DTypePolicy = DTypePolicy()
    # ragged packing: explicit per-round sequence lengths (sum = tokens
    # per round); None = uniform batch x seq_len
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

    # -- parameter counts (duck-typed for the shared chain builder) -----------
    @property
    def block_params(self) -> int:
        d, q, kv, ff = self.d_model, self.q_dim, self.kv_dim, self.d_ff
        attn = d * q + 2 * d * kv + q * d
        mlp = 3 * d * ff
        norms = 2 * d + 2 * self.head_dim  # block norms + q/k norms
        return attn + mlp + norms

    @property
    def embed_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def head_params(self) -> int:
        return self.vocab_size * self.d_model

    @property
    def saved_ctx_width(self) -> int:
        """Rough elements/token saved per block (roofline seed only — the
        lowering replaces every size with the packed layout's exact bytes):
        qm/attn_out (2q), km/v (2kv), h_mid (d), x1/x3 (2ff), plus the fp32
        rstds and lse counted as 2 bf16-equivalent elements each."""
        d, q, kv, ff = self.d_model, self.q_dim, self.kv_dim, self.d_ff
        h, kvh = self.n_heads, self.n_kv_heads
        return 2 * q + 2 * kv + d + 2 * ff + 2 * (2 + h + kvh) + 2 * h

    @classmethod
    def tiny(cls) -> "ShapedQwen3Config":
        return cls(
            n_layers=2, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
            d_ff=512, vocab_size=512, seq_len=64, batch=1,
        )

    @classmethod
    def qwen3_8b(cls, *, seq_len: int = 4096, batch: int = 1,
                 grad_accum_rounds: int = 1, num_steps: int = 1) -> "ShapedQwen3Config":
        return cls(seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)


def build_shaped_qwen3(
    cfg: ShapedQwen3Config,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels=None,
    name: str | None = None,
):
    hw = hw or ShapedHardware()
    from ..lowering.freeze_plan import derive_freeze_plan

    dims_fp, fl_fp = family_layouts(cfg)
    freeze_plan = derive_freeze_plan(
        dims_fp, cfg.n_layers,
        lambda i: [f.name for f in fl_fp.layers[i].weights.fields],
        tied_embeddings=bool(getattr(cfg, "tied_embeddings", False)),
    )
    return build_shaped_program(
        cfg, hw=hw, family="qwen3-shaped",
        kinds={"block": roofline_block_kind_spec(cfg, hw)},
        fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=name,
        freeze=freeze_plan,
    )


def dims_of_qwen3(cfg: ShapedQwen3Config) -> Qwen3Dims:
    return Qwen3Dims(
        opt_policy=cfg.opt_policy,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        tokens=cfg.tokens,
        seq_len=cfg.seq_len,
        dtypes=getattr(cfg, "dtypes", None) or DTypePolicy(),
        seq_lens=getattr(cfg, "seq_lens", None),
    )


def family_layouts(cfg: ShapedQwen3Config) -> tuple[Qwen3Dims, FamilyLayouts]:
    dims = dims_of_qwen3(cfg)
    cl = qwen3_activation_layout(dims)
    return dims, FamilyLayouts(
        layers=[LayerLayout(kind="block",
                            weights=qwen3_weight_layout(dims, layer=i),
                            activations=cl)
                for i in range(cfg.n_layers)],
        embed=embed_weight_layout(dims),
        head=head_weight_layout(dims),
    )


def lower_qwen3(
    cfg: ShapedQwen3Config,
    *,
    hw: ShapedHardware | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    fast_memory_capacity: int | None = None,
) -> Program:
    shaped = build_shaped_qwen3(
        cfg, hw=hw, recompute_levels=recompute_levels, fast_memory_capacity=fast_memory_capacity,
    )
    dims, fl = family_layouts(cfg)
    return apply_exact_sizes(shaped, "qwen3-exact", size_of=size_of_factory(dims, fl))


def initial_values_qwen3(program: Program, cfg: ShapedQwen3Config, backend, *, seed: int = 0, into=None):
    dims, fl = family_layouts(cfg)
    return initial_values_from_layouts(program, dims, fl, backend, seed=seed, into=into)
