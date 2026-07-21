"""GPT-2 family: config + declarations over the generic machinery.

The nanogpt-speedrun BASELINE model (llm.c GPT-2 124M): pre-LN LayerNorm
blocks with biases everywhere, fused c_attn QKV, full MHA over LEARNED
positions (no rope), GELU-tanh MLP, untied embed/head by default with
``tied_embeddings=True`` restoring the classic GPT-2 tying (the qwen35
tied convention: one W_embed packs the head's fields too).

Emitted-chain naming is the family-invariant shape (see llama3/model.py);
the only structural novelty is W_embed packing TWO tables ([w | wpe]) —
the learned-position table rides the embed object, its gather rows are
``Segments.positions`` (positions restart per sequence, so packed varlen
rounds are position-correct by construction).

Init reproduces the GPT-2/nanoGPT recipe as an InitPolicy: N(0, 0.02)
everywhere, ZEROS for every bias, ones/zeros for LayerNorm gain/bias, and
the residual projections (wo, w_proj) at N(0, 0.02/sqrt(2*n_layers)) —
the paper's depth-scaled residual init.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from dataflow.core import Program
from dataflow_training.blocks.layouts import (
    DTypePolicy,
    Gpt2Dims,
    gpt2_activation_layout,
    gpt2_embed_layout,
    gpt2_head_layout,
    gpt2_weight_layout,
)
from ...lowering.emit import FamilyLayouts, LayerLayout, apply_exact_sizes, initial_values_from_layouts, object_size_factory
from ...lowering.shaped_program import ShapedHardware, build_shaped_program, roofline_block_kind_spec


@dataclass(frozen=True)
class ShapedGpt2Config:
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_ff: int = 3072
    vocab_size: int = 50304          # 50257 padded to 64x (the nanoGPT pad)
    seq_len: int = 1024
    batch: int = 1
    grad_accum_rounds: int = 1
    num_steps: int = 1
    # learned-position table rows; None = seq_len. Every sequence of a
    # round (uniform seq_len or ragged seq_lens entry) must fit inside it.
    n_ctx: int | None = None
    # tied embed/head (classic GPT-2). Default UNTIED — the repo baseline
    # convention (llama3 ladder, modded-nanogpt).
    tied_embeddings: bool = False
    # biases in Linears AND LayerNorms (the nanoGPT flag): True = classic
    # GPT-2; False = the bias-free variant (like llama3; the speedrun's
    # own first simplification)
    use_bias: bool = True
    optimizer_placement: str = "interleaved"
    # per-field optimizer assignment (tasks/optim.py): "adamw" (default) |
    # "sgd" | "sgdm" | "muon" | an OptPolicy with fnmatch overrides.
    opt_policy: object = "adamw"
    dtypes: DTypePolicy = DTypePolicy()
    # ragged packing: explicit per-round sequence lengths (sum = tokens
    # per round); None = uniform batch x seq_len
    seq_lens: tuple[int, ...] | None = None
    # init-policy wire override (model_families/init_policy.py). None =
    # the family's GPT-2 recipe (NOT the plain repo default).
    init_policy: object = None

    @property
    def max_tokens(self) -> int:
        if self.seq_lens is not None:
            return sum(self.seq_lens)
        return self.seq_len * self.batch

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_kv_heads(self) -> int:
        return self.n_heads          # full MHA (shaped metadata reads this)

    @property
    def kv_dim(self) -> int:
        return self.d_model          # full MHA (roofline spec reads this)

    @property
    def position_rows(self) -> int:
        return self.n_ctx if self.n_ctx is not None else self.seq_len

    # -- parameter counts (roofline seeds; exact sizes come from layouts) ------
    @property
    def block_params(self) -> int:
        d, ff = self.d_model, self.d_ff
        attn = d * 3 * d + d * d
        mlp = 2 * d * ff
        norms = 2 * d
        if self.use_bias:
            attn += 3 * d + d                     # b_qkv, b_o
            mlp += ff + d                         # b_fc, b_proj
            norms += 2 * d                        # LN biases
        return attn + mlp + norms

    @property
    def embed_params(self) -> int:
        n = self.vocab_size * self.d_model + self.position_rows * self.d_model
        if self.tied_embeddings:
            n += (2 if self.use_bias else 1) * self.d_model  # final LN rides W_embed
        return n

    @property
    def head_params(self) -> int:
        return (self.vocab_size * self.d_model
                + (2 if self.use_bias else 1) * self.d_model)

    # -- activation widths (elements per token, roofline seed) -----------------
    @property
    def saved_ctx_width(self) -> int:
        d, ff, h = self.d_model, self.d_ff, self.n_heads
        return 6 * d + h + ff + 4                 # q/k/v/attn_out/h_mid + lse + x_fc + stats

    @classmethod
    def tiny(cls) -> "ShapedGpt2Config":
        return cls(
            n_layers=2,
            d_model=64,
            n_heads=4,
            d_ff=160,
            vocab_size=512,
            seq_len=64,
            batch=1,
        )

    @classmethod
    def tiny_tied(cls) -> "ShapedGpt2Config":
        """The classic tied variant (one W_embed serves embed AND head)."""
        from dataclasses import replace

        return replace(cls.tiny(), tied_embeddings=True)

    @classmethod
    def tiny_nobias(cls) -> "ShapedGpt2Config":
        """The bias-free variant (nanoGPT bias=False; llama3-like linears)."""
        from dataclasses import replace

        return replace(cls.tiny(), use_bias=False)

    @classmethod
    def gpt2_124m(cls, *, seq_len: int = 1024, batch: int = 1,
                  grad_accum_rounds: int = 1, num_steps: int = 1) -> "ShapedGpt2Config":
        return cls(seq_len=seq_len, batch=batch,
                   grad_accum_rounds=grad_accum_rounds, num_steps=num_steps)


def derive_dims(cfg: ShapedGpt2Config) -> Gpt2Dims:
    if cfg.d_model % cfg.n_heads != 0:
        raise ValueError(f"d_model {cfg.d_model} not divisible by "
                         f"n_heads {cfg.n_heads}")
    n_ctx = cfg.position_rows
    if cfg.seq_lens is not None:
        if max(cfg.seq_lens) > n_ctx:
            raise ValueError(
                f"segment length {max(cfg.seq_lens)} exceeds n_ctx {n_ctx} "
                f"(learned positions cannot extend past the table)")
    elif cfg.seq_len > n_ctx:
        raise ValueError(f"seq_len {cfg.seq_len} exceeds n_ctx {n_ctx}")
    return Gpt2Dims(
        opt_policy=cfg.opt_policy,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        max_tokens=cfg.max_tokens,
        seq_len=cfg.seq_len,
        n_ctx=n_ctx,
        tied=cfg.tied_embeddings,
        use_bias=cfg.use_bias,
        dtypes=getattr(cfg, "dtypes", None) or DTypePolicy(),
        seq_lens=getattr(cfg, "seq_lens", None),
    )


def build_shaped_gpt2(
    cfg: ShapedGpt2Config,
    *,
    hw: ShapedHardware | None = None,
    fast_memory_capacity: int | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    name: str | None = None,
) -> Program:
    """GPT-2 through the generic builder: one dense-transformer kind."""
    hw = hw or ShapedHardware()
    from ...lowering.freeze_plan import derive_freeze_plan

    d = derive_dims(cfg)
    plan = derive_freeze_plan(
        d, cfg.n_layers,
        lambda i: [f.name for f in gpt2_weight_layout(d, layer=i).fields],
        tied_embeddings=cfg.tied_embeddings,
    )
    return build_shaped_program(
        cfg, hw=hw, family="gpt2-shaped",
        kinds={"block": roofline_block_kind_spec(
            cfg, hw,
            weight_fields=[(f.name, f.shape)
                           for f in gpt2_weight_layout(d).fields])},
        fast_memory_capacity=fast_memory_capacity,
        recompute_levels=recompute_levels, name=name,
        freeze=plan,
    )


def family_layouts(cfg: ShapedGpt2Config) -> tuple[Gpt2Dims, FamilyLayouts]:
    dims = derive_dims(cfg)
    cl = gpt2_activation_layout(dims)
    layers = [LayerLayout(kind="block",
                          weights=gpt2_weight_layout(dims, layer=i),
                          activations=cl)
              for i in range(cfg.n_layers)]
    embed = gpt2_embed_layout(dims)
    return dims, FamilyLayouts(
        layers=layers,
        embed=embed,
        head=embed if dims.tied else gpt2_head_layout(dims),
        embed_ns="head" if dims.tied else "embed",
    )


def lower_gpt2(
    cfg: ShapedGpt2Config,
    *,
    hw: ShapedHardware | None = None,
    recompute_levels: Mapping[str, int] | None = None,
    fast_memory_capacity: int | None = None,
) -> Program:
    shaped = build_shaped_gpt2(
        cfg, hw=hw, recompute_levels=recompute_levels, fast_memory_capacity=fast_memory_capacity,
    )
    dims, fl = family_layouts(cfg)
    return apply_exact_sizes(shaped, "gpt2-exact", object_size=object_size_factory(dims, fl))


def gpt2_init_policy(n_layers: int):
    """The GPT-2/nanoGPT init as an InitPolicy: N(0, 0.02) default, ones
    for LN gains, ZEROS for LN biases and every linear bias, and the
    residual projections depth-scaled to N(0, 0.02/sqrt(2*n_layers))."""
    from dataflow_training.model_families.init_policy import InitPolicy, InitRule

    scaled = InitRule("scaled_normal",
                      {"std": 0.02, "divide_by": math.sqrt(2 * n_layers)})
    return InitPolicy(overrides=(
        ("*_norm_w", InitRule("constant", {"value": 1.0})),
        ("*_norm_b", InitRule("constant", {"value": 0.0})),
        ("b_*", InitRule("constant", {"value": 0.0})),
        ("wo", scaled),
        ("w_proj", scaled),
    ))


def initial_values(program: Program, cfg: ShapedGpt2Config, backend, *,
                   seed: int = 0, into=None):
    from dataflow_training.model_families.init_policy import build_init_policy

    dims, fl = family_layouts(cfg)
    policy = (build_init_policy(cfg.init_policy) if cfg.init_policy is not None
              else gpt2_init_policy(cfg.n_layers))
    return initial_values_from_layouts(program, dims, fl, backend,
                                       seed=seed, into=into,
                                       init_policy=policy)
