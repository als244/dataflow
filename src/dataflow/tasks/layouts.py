"""Packed buffer layouts: many named tensors inside one runtime object.

A block's weights (`W_i`), gradients (`dW_i`), optimizer state (`O_i`), and
saved backward context (`A_*`) are each ONE dataflow object; these layouts
define the field offsets inside them. The layout is the single source of
truth for object sizes (exact by construction — the lowering asks
`layout.total_bytes`) and for the torch views executables operate on.

Import-light: dtype names + byte math only (usable by the lowering without
torch); `view()`/`views()` import torch interop lazily.

Offsets are 256-byte aligned (copy-engine/vector-load safe).
"""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

from dataflow.core import DTYPE_BITS
from dataflow.runtime.device.base import Buffer

if TYPE_CHECKING:  # torch-free at runtime: moe.spec imports lazily below
    from .moe.spec import MoESpec

_ALIGN = 256


@dataclass(frozen=True)
class ParamDTypes:
    """Storage dtypes for one trainable field: the parameter itself, its
    gradient, and its AdamW moments (design: docs/notes/dtype-policy-design.md).
    Defaults reproduce the historical all-bf16 convention."""

    param: str = "bf16"
    grad: str = "bf16"
    opt: str = "bf16"

    def __post_init__(self) -> None:
        for role in ("param", "grad", "opt"):
            name = getattr(self, role)
            if name not in DTYPE_BITS:
                raise ValueError(f"unknown dtype {name!r} for {role} "
                                 f"(known: {sorted(DTYPE_BITS)})")


@dataclass(frozen=True)
class DTypePolicy:
    """Per-field dtype selection over packed WEIGHT-layout field names.

    ``overrides`` is an ordered tuple of (fnmatch pattern, ParamDTypes);
    the FIRST matching pattern wins, else ``default``. Field names are the
    user-visible parameter unit everywhere else in the system ("w_qkvz",
    "A_log", "attn_norm_w", ...), so patterns like "*_norm_w" select the
    natural groups.
    """

    default: ParamDTypes = ParamDTypes()
    overrides: tuple[tuple[str, ParamDTypes], ...] = ()
    # depth-dependent policies: ordered (layer-index tuple, sub-policy);
    # the FIRST entry containing the layer wins and its sub-policy answers
    # ALL lookups for that layer (no fallthrough into the outer overrides —
    # one policy owns a layer). Loose objects (embed/head tables) have no
    # layer and always use the outer policy. Indices are explicit ints —
    # write tuple(range(4)) for "the first four layers".
    layer_overrides: tuple[tuple[tuple[int, ...], "DTypePolicy"], ...] = ()

    def for_layer(self, layer: int | None) -> "DTypePolicy":
        if layer is not None:
            for layers, sub in self.layer_overrides:
                if layer in layers:
                    return sub
        return self

    def for_field(self, name: str, layer: int | None = None) -> ParamDTypes:
        pol = self.for_layer(layer)
        for pattern, dts in pol.overrides:
            if fnmatchcase(name, pattern):
                return dts
        return pol.default

    @property
    def depth_dependent(self) -> bool:
        return bool(self.layer_overrides)


@dataclass(frozen=True)
class Field:
    name: str
    shape: tuple[int, ...]
    dtype: str  # dataflow dtype name ("bf16", "fp32", "int32", ...)
    offset_bytes: int
    nbytes: int


@dataclass(frozen=True)
class PackedLayout:
    fields: tuple[Field, ...]
    total_bytes: int

    @classmethod
    def build(cls, specs: list[tuple[str, tuple[int, ...], str]]) -> "PackedLayout":
        fields: list[Field] = []
        offset = 0
        for name, shape, dtype in specs:
            n = 1
            for d in shape:
                n *= int(d)
            nbytes = n * DTYPE_BITS[dtype] // 8
            fields.append(Field(name=name, shape=shape, dtype=dtype, offset_bytes=offset, nbytes=nbytes))
            offset += (nbytes + _ALIGN - 1) // _ALIGN * _ALIGN
        return cls(fields=tuple(fields), total_bytes=offset)

    def field(self, name: str) -> Field:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    def view(self, buffer: Buffer, name: str):
        from .interop import TORCH_DTYPE_BY_NAME, torch_view

        f = self.field(name)
        return torch_view(buffer, f.shape, TORCH_DTYPE_BY_NAME[f.dtype], offset_bytes=f.offset_bytes)

    def views(self, buffer: Buffer) -> dict:
        from .interop import TORCH_DTYPE_BY_NAME, torch_view

        return {
            f.name: torch_view(buffer, f.shape, TORCH_DTYPE_BY_NAME[f.dtype], offset_bytes=f.offset_bytes)
            for f in self.fields
        }

    def unpack_tensor(self, flat) -> dict:
        """Views into a flat uint8 torch tensor (golden-reference side)."""
        import torch

        from .interop import TORCH_DTYPE_BY_NAME

        out = {}
        for f in self.fields:
            n = 1
            for d in f.shape:
                n *= int(d)
            dt = TORCH_DTYPE_BY_NAME[f.dtype]
            sl = flat[f.offset_bytes : f.offset_bytes + f.nbytes]
            out[f.name] = sl.view(dt).view(f.shape) if n else sl.view(dt)
        return out


@dataclass(frozen=True)
class LlamaDims:
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    tokens: int
    seq_len: int
    rope_base: float = 500_000.0
    dtypes: DTypePolicy = DTypePolicy()
    # explicit per-sequence lengths for ragged packing (sum == tokens);
    # None = uniform sequences of seq_len (varlen-first design note)
    seq_lens: tuple[int, ...] | None = None

    @property
    def seq_spec(self):
        """int (uniform) or tuple (ragged) — the ops-layer seq argument."""
        return self.seq_lens if self.seq_lens is not None else self.seq_len

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


@dataclass(frozen=True)
class Qwen3Dims:
    """Qwen3-dense dimensions. Differences from llama that matter here:
    qk-norm (per-head RMSNorm on q/k between projection and rope, one shared
    (head_dim,) weight each), rope theta 1e6, and head_dim DECOUPLED from
    d_model/n_heads (Qwen3-4B/32B project q to n_heads*head_dim != d_model;
    for 8B they coincide). No biases anywhere; embed/head untied at 8B."""

    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab_size: int
    tokens: int
    seq_len: int
    rope_base: float = 1_000_000.0
    dtypes: DTypePolicy = DTypePolicy()
    # explicit per-sequence lengths for ragged packing (sum == tokens);
    # None = uniform sequences of seq_len (varlen-first design note)
    seq_lens: tuple[int, ...] | None = None

    @property
    def seq_spec(self):
        """int (uniform) or tuple (ragged) — the ops-layer seq argument."""
        return self.seq_lens if self.seq_lens is not None else self.seq_len

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


def _param_specs(dims, names_shapes, ns: str | None = None,
                 layer: int | None = None,
                 ) -> list[tuple[str, tuple[int, ...], str]]:
    """Trainable-field specs at the dims' dtype policy (param role).

    ``ns`` prefixes the POLICY LOOKUP name (not the field name): the embed
    and head tables both pack a field literally named "w", so policies
    address them as "embed.w" / "head.w" / "head.final_norm_w". Block
    fields are looked up bare ("wq", "A_log", "*_norm_w", ...).
    """
    policy = getattr(dims, "dtypes", None) or DTypePolicy()
    key = (lambda n: f"{ns}.{n}") if ns else (lambda n: n)
    return [(n, s, policy.for_field(key(n), layer).param) for n, s in names_shapes]


def _policy_of(dims) -> DTypePolicy:
    return getattr(dims, "dtypes", None) or DTypePolicy()


def _lse_spec(dims, n_heads: int) -> tuple[str, tuple[int, ...], str]:
    """Flash lse context field. Uniform batches keep the historical
    (batch*heads, seq_len) shape; ragged packing stores (heads, tokens)
    (same element count — ops.flash_fwd emits the matching layout)."""
    t = dims.tokens
    if getattr(dims, "seq_lens", None) is not None and len(set(dims.seq_lens)) > 1:
        return ("lse", (n_heads, t), "fp32")
    return ("lse", ((t // dims.seq_len) * n_heads, dims.seq_len), "fp32")


def grad_layout(weight: PackedLayout, policy: DTypePolicy,
                ns: str | None = None, layer: int | None = None) -> PackedLayout:
    """dW layout mirroring a weight layout field-by-field at grad dtypes."""
    key = (lambda n: f"{ns}.{n}") if ns else (lambda n: n)
    return PackedLayout.build(
        [(f.name, f.shape, policy.for_field(key(f.name), layer).grad)
         for f in weight.fields]
    )


def opt_state_layout(weight: PackedLayout, policy: DTypePolicy,
                     ns: str | None = None, layer: int | None = None) -> PackedLayout:
    """AdamW state for one weight object: per-field first/second moments at
    the policy's opt dtype. Under the all-bf16 default this packs to the
    same total bytes as the historical flat bf16-halves layout (every
    m/v span is the field's byte span, alignment included) but the interior
    MAPPING is per-field [m_f | v_f] pairs, never covering padding gaps."""
    key = (lambda n: f"{ns}.{n}") if ns else (lambda n: n)
    specs: list[tuple[str, tuple[int, ...], str]] = []
    for f in weight.fields:
        o = policy.for_field(key(f.name), layer).opt
        specs.append((f"m_{f.name}", f.shape, o))
        specs.append((f"v_{f.name}", f.shape, o))
    return PackedLayout.build(specs)


def weight_layout(dims: LlamaDims, layer: int | None = None) -> PackedLayout:
    d, kv, ff = dims.d_model, dims.kv_dim, dims.d_ff
    return PackedLayout.build(_param_specs(dims, [
        ("attn_norm_w", (d,)),
        ("wq", (d, d)),
        ("wk", (d, kv)),
        ("wv", (d, kv)),
        ("wo", (d, d)),
        ("ffn_norm_w", (d,)),
        ("w1", (d, ff)),
        ("w3", (d, ff)),
        ("w2", (ff, d)),
    ], layer=layer))


def context_layout(dims: LlamaDims) -> PackedLayout:
    """Saved backward context for one block forward (the `A_*` object).

    Saves exactly what block backward needs beyond (x, W): post-rope q/k, v,
    flash lse + attention output, the post-attention residual (h_mid), both
    rmsnorm rstds, and the two MLP projections.
    """
    t, d, kv, ff, h = dims.tokens, dims.d_model, dims.kv_dim, dims.d_ff, dims.n_heads
    return PackedLayout.build([
        ("rstd_attn", (t,), "fp32"),
        ("q", (t, d), "bf16"),
        ("k", (t, kv), "bf16"),
        ("v", (t, kv), "bf16"),
        _lse_spec(dims, h),
        ("attn_out", (t, d), "bf16"),
        ("h_mid", (t, d), "bf16"),
        ("rstd_ffn", (t,), "fp32"),
        ("x1", (t, ff), "bf16"),
        ("x3", (t, ff), "bf16"),
    ])


def qwen3_weight_layout(dims: Qwen3Dims, layer: int | None = None) -> PackedLayout:
    d, q, kv, ff, hd = dims.d_model, dims.q_dim, dims.kv_dim, dims.d_ff, dims.head_dim
    return PackedLayout.build(_param_specs(dims, [
        ("attn_norm_w", (d,)),
        ("wq", (d, q)),
        ("wk", (d, kv)),
        ("wv", (d, kv)),
        ("q_norm_w", (hd,)),
        ("k_norm_w", (hd,)),
        ("wo", (q, d)),
        ("ffn_norm_w", (d,)),
        ("w1", (d, ff)),
        ("w3", (d, ff)),
        ("w2", (ff, d)),
    ], layer=layer))


@dataclass(frozen=True)
class OlmoeDims(Qwen3Dims):
    """OLMoE dimensions: qwen3-shaped attention with two differences —
    FULL-ROW qk-norm (one RMSNorm over the whole (t, q_dim)/(t, kv_dim)
    rows, weights (q_dim,)/(kv_dim,), one rstd per token — NOT per-head)
    and no GQA at 7B (n_kv_heads == n_heads) — plus a MoE FFN described by
    ``moe`` (d_ff aliases d_ff_expert, metadata only). rope theta 1e4."""

    moe: MoESpec | None = None


def olmoe_weight_layout(dims: OlmoeDims, layer: int | None = None) -> PackedLayout:
    from .moe.spec import moe_weight_specs

    d, q, kv = dims.d_model, dims.q_dim, dims.kv_dim
    return PackedLayout.build(_param_specs(dims, [
        ("attn_norm_w", (d,)),
        ("wq", (d, q)),
        ("wk", (d, kv)),
        ("wv", (d, kv)),
        ("q_norm_w", (q,)),
        ("k_norm_w", (kv,)),
        ("wo", (q, d)),
        ("ffn_norm_w", (d,)),
    ] + moe_weight_specs(dims, dims.moe), layer=layer))


def olmoe_context_layout(dims: OlmoeDims) -> PackedLayout:
    """Saved backward context for one OLMoE block: qwen3's save-pre-norm
    convention with FULL-ROW rstds (one per token), plus the MoE tail's
    routing decision + pre-activations (tasks/moe/spec.py)."""
    from .moe.spec import moe_context_specs

    t, d, q, kv, h = dims.tokens, dims.d_model, dims.q_dim, dims.kv_dim, dims.n_heads
    return PackedLayout.build([
        ("rstd_attn", (t,), "fp32"),
        ("qm", (t, q), "bf16"),
        ("km", (t, kv), "bf16"),
        ("rstd_q", (t,), "fp32"),
        ("rstd_k", (t,), "fp32"),
        ("v", (t, kv), "bf16"),
        _lse_spec(dims, h),
        ("attn_out", (t, q), "bf16"),
        ("h_mid", (t, d), "bf16"),
        ("rstd_ffn", (t,), "fp32"),
    ] + moe_context_specs(dims, dims.moe))


def qwen3_context_layout(dims: Qwen3Dims) -> PackedLayout:
    """Saved backward context for one Qwen3 block forward.

    qk-norm changes what is worth saving: instead of post-rope q/k we save
    the PRE-norm projections (qm/km) plus the per-head rstds — backward then
    re-applies norm+rope (cheap elementwise) to rebuild flash-bwd's q/k, and
    has exactly the tensors rmsnorm_bwd needs for the qk-norm gradient. v,
    lse, attn_out, h_mid, both block rstds and the MLP projections are saved
    as in llama."""
    t, d, q, kv, ff = dims.tokens, dims.d_model, dims.q_dim, dims.kv_dim, dims.d_ff
    h, kvh = dims.n_heads, dims.n_kv_heads
    return PackedLayout.build([
        ("rstd_attn", (t,), "fp32"),
        ("qm", (t, q), "bf16"),
        ("km", (t, kv), "bf16"),
        ("rstd_q", (t * h,), "fp32"),
        ("rstd_k", (t * kvh,), "fp32"),
        ("v", (t, kv), "bf16"),
        _lse_spec(dims, h),
        ("attn_out", (t, q), "bf16"),
        ("h_mid", (t, d), "bf16"),
        ("rstd_ffn", (t,), "fp32"),
        ("x1", (t, ff), "bf16"),
        ("x3", (t, ff), "bf16"),
    ])


@dataclass(frozen=True)
class Qwen3MoeDims(Qwen3Dims):
    """Qwen3-MoE dimensions (Qwen3-30B-A3B / 235B-A22B): qwen3's attention
    VERBATIM (per-head qk-norm with (head_dim,) weights, GQA, rope 1e6) —
    only the FFN changes, to a routed SwiGLU MoE described by ``moe``
    (E=128, top-8, norm_topk_prob=true -> topk_then_softmax, aux 0.001,
    NO shared expert, all layers sparse). d_ff aliases d_ff_expert."""

    moe: MoESpec | None = None


def qwen3moe_weight_layout(dims: Qwen3MoeDims, layer: int | None = None) -> PackedLayout:
    from .moe.spec import moe_weight_specs

    d, q, kv, hd = dims.d_model, dims.q_dim, dims.kv_dim, dims.head_dim
    return PackedLayout.build(_param_specs(dims, [
        ("attn_norm_w", (d,)),
        ("wq", (d, q)),
        ("wk", (d, kv)),
        ("wv", (d, kv)),
        ("q_norm_w", (hd,)),
        ("k_norm_w", (hd,)),
        ("wo", (q, d)),
        ("ffn_norm_w", (d,)),
    ] + moe_weight_specs(dims, dims.moe), layer=layer))


def qwen3moe_context_layout(dims: Qwen3MoeDims) -> PackedLayout:
    """qwen3's save-pre-norm convention with PER-HEAD rstds, plus the MoE
    tail's routing decision + pre-activations (tasks/moe/spec.py)."""
    from .moe.spec import moe_context_specs

    t, d, q, kv = dims.tokens, dims.d_model, dims.q_dim, dims.kv_dim
    h, kvh = dims.n_heads, dims.n_kv_heads
    return PackedLayout.build([
        ("rstd_attn", (t,), "fp32"),
        ("qm", (t, q), "bf16"),
        ("km", (t, kv), "bf16"),
        ("rstd_q", (t * h,), "fp32"),
        ("rstd_k", (t * kvh,), "fp32"),
        ("v", (t, kv), "bf16"),
        _lse_spec(dims, h),
        ("attn_out", (t, q), "bf16"),
        ("h_mid", (t, d), "bf16"),
        ("rstd_ffn", (t,), "fp32"),
    ] + moe_context_specs(dims, dims.moe))


@dataclass(frozen=True)
class Dsv3Dims:
    """DeepSeek-V3 dimensions: MLA attention + hybrid dense/MoE depth.

    MLA: q through the low-rank stack (d -> q_lora_rank -> n_heads *
    (qk_nope_dim + qk_rope_dim), RMSNorm mid-stack); kv through
    d -> (kv_lora_rank + qk_rope_dim) where the LAST qk_rope_dim columns
    are the ONE shared decoupled k_rope per token; latent RMSNorm then
    -> n_heads * (qk_nope_dim + v_head_dim). Per-head attention dims:
    qk = nope + rope, v = v_head_dim (flash runs at shared head_dim=qk
    with zero-padded v — exact; see tasks/mla_reference.py). rope 1e4 on
    rope dims only. First ``first_k_dense`` layers use a dense SwiGLU FFN
    (d_ff aliases d_ff_dense for the shared dense stages); the rest are
    MoE per ``moe`` (sigmoid_noaux_tc, ungated shared expert).
    """

    d_model: int
    n_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_dim: int
    qk_rope_dim: int
    v_head_dim: int
    d_ff: int                     # DENSE-kind FFN width (d_ff_dense)
    first_k_dense: int
    vocab_size: int
    tokens: int
    seq_len: int
    rope_base: float = 10_000.0
    dtypes: DTypePolicy = DTypePolicy()
    seq_lens: tuple[int, ...] | None = None
    moe: MoESpec | None = None

    @property
    def seq_spec(self):
        return self.seq_lens if self.seq_lens is not None else self.seq_len

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_dim + self.qk_rope_dim

    @property
    def q_dim(self) -> int:       # full q width after w_q_b
        return self.n_heads * self.qk_head_dim

    @property
    def v_dim(self) -> int:       # attention-output width fed to wo
        return self.n_heads * self.v_head_dim

    def kind_of(self, layer: int) -> str:
        return "dense" if layer < self.first_k_dense else "moe"


def _dsv3_attn_weight_specs(dims: Dsv3Dims) -> list[tuple[str, tuple[int, ...]]]:
    d, h = dims.d_model, dims.n_heads
    return [
        ("attn_norm_w", (d,)),
        ("w_q_a", (d, dims.q_lora_rank)),
        ("q_a_norm_w", (dims.q_lora_rank,)),
        ("w_q_b", (dims.q_lora_rank, h * dims.qk_head_dim)),
        ("w_kv_a", (d, dims.kv_lora_rank + dims.qk_rope_dim)),
        ("kv_a_norm_w", (dims.kv_lora_rank,)),
        ("w_kv_b", (dims.kv_lora_rank, h * (dims.qk_nope_dim + dims.v_head_dim))),
        ("wo", (h * dims.v_head_dim, d)),
        ("ffn_norm_w", (d,)),
    ]


def _dsv3_attn_ctx_specs(dims: Dsv3Dims) -> list[tuple[str, tuple[int, ...], str]]:
    """The MLA saved set — COMPRESSED latents, not expanded heads: bwd
    re-expands through w_q_b/w_kv_b, so attention ctx is ~(q_lora +
    kv_lora + rope) wide instead of h*(qk+v). Pre-norm/pre-rope saves
    (the repo convention); attn_out at the TRUE (t, h*v) — the padded
    form is reconstructed from known-zeros at bwd time."""
    t = dims.tokens
    return [
        ("rstd_attn", (t,), "fp32"),
        ("q_a", (t, dims.q_lora_rank), "bf16"),
        ("rstd_qa", (t,), "fp32"),
        ("kv_a", (t, dims.kv_lora_rank + dims.qk_rope_dim), "bf16"),
        ("rstd_kva", (t,), "fp32"),
        _lse_spec(dims, dims.n_heads),
        ("attn_out", (t, dims.n_heads * dims.v_head_dim), "bf16"),
        ("h_mid", (t, dims.d_model), "bf16"),
        ("rstd_ffn", (t,), "fp32"),
    ]


def dsv3_dense_weight_layout(dims: Dsv3Dims, layer: int | None = None) -> PackedLayout:
    d, ff = dims.d_model, dims.d_ff
    return PackedLayout.build(_param_specs(dims, _dsv3_attn_weight_specs(dims) + [
        ("w1", (d, ff)),
        ("w3", (d, ff)),
        ("w2", (ff, d)),
    ], layer=layer))


def dsv3_dense_context_layout(dims: Dsv3Dims) -> PackedLayout:
    t, ff = dims.tokens, dims.d_ff
    return PackedLayout.build(_dsv3_attn_ctx_specs(dims) + [
        ("x1", (t, ff), "bf16"),
        ("x3", (t, ff), "bf16"),
    ])


def dsv3_moe_weight_layout(dims: Dsv3Dims, layer: int | None = None) -> PackedLayout:
    from .moe.spec import moe_weight_specs

    return PackedLayout.build(_param_specs(
        dims, _dsv3_attn_weight_specs(dims) + moe_weight_specs(dims, dims.moe),
        layer=layer,
    ))


def dsv3_moe_context_layout(dims: Dsv3Dims) -> PackedLayout:
    from .moe.spec import moe_context_specs

    return PackedLayout.build(
        _dsv3_attn_ctx_specs(dims) + moe_context_specs(dims, dims.moe)
    )


@dataclass(frozen=True)
class Dsv32Dims(Dsv3Dims):
    """DeepSeek-V3.2 dims: the dsv3 backbone + DSA (lightning indexer +
    fine-grained top-k selection) in EVERY layer's attention. Indexer:
    H_I heads of d_I dims; q^I taps the shared post-norm q_lora latent;
    k^I = rope(LayerNorm(h1 @ w_idx_k)) — ONE shared key per token;
    rope-FIRST head layout (opposite of main MLA); selection = per-token
    top-min(k, prefix) of ReLU-weighted scores, stored (t, k) int32
    (pad-safe short prefixes). sparse_mode=False = the paper's DENSE
    WARM-UP stage: main attention runs dense (dsv3 flash path, no
    selection, no dsa_idx ctx), the MAIN MODEL IS FROZEN (optimizer
    no-ops every non-indexer field, embed/head/bias included), and the
    indexer trains from the full-prefix KL (report formula 3) — its
    only signal. Dense mode requires train_indexer=True."""

    index_n_heads: int = 8
    index_head_dim: int = 64
    index_topk: int = 1024
    sparse_mode: bool = True
    # ablation knob (Shein): False = FREEZE the indexer — no KL loss, no
    # indexer gradients, optimizer skips its five fields entirely (not
    # even weight decay). Default True = paper-faithful sparse training
    # (the KL is the indexer's ONLY training signal).
    train_indexer: bool = True


def _dsv32_attn_weight_specs(dims: Dsv32Dims) -> list[tuple[str, tuple[int, ...]]]:
    specs = _dsv3_attn_weight_specs(dims)
    idx = [
        ("w_idx_q", (dims.q_lora_rank, dims.index_n_heads * dims.index_head_dim)),
        ("w_idx_k", (dims.d_model, dims.index_head_dim)),
        ("idx_k_ln_w", (dims.index_head_dim,)),
        ("idx_k_ln_b", (dims.index_head_dim,)),
        ("w_idx_w", (dims.d_model, dims.index_n_heads)),  # fp32 via policy
    ]
    # insert after wo, before ffn_norm_w
    return specs[:-1] + idx + specs[-1:]


def _dsv32_attn_ctx_specs(dims: Dsv32Dims) -> list[tuple[str, tuple[int, ...], str]]:
    specs = _dsv3_attn_ctx_specs(dims)
    if not dims.sparse_mode:
        # dense warm-up: no selection — ctx is exactly dsv3's (the KL
        # target rebuilds from lse + latents over the full causal prefix)
        return specs
    # the ONLY DSA ctx addition: the selection (indexer q/k/wts recompute
    # from the latents already saved for the MLA backward); emitted
    # before the attention stage, so it sits before lse in layout order
    sel = ("dsa_idx", (dims.tokens, dims.index_topk), "int32")
    i = next(j for j, s in enumerate(specs) if s[0] == "lse")
    return specs[:i] + [sel] + specs[i:]


def dsv32_dense_weight_layout(dims: Dsv32Dims, layer: int | None = None) -> PackedLayout:
    d, ff = dims.d_model, dims.d_ff
    return PackedLayout.build(_param_specs(dims, _dsv32_attn_weight_specs(dims) + [
        ("w1", (d, ff)),
        ("w3", (d, ff)),
        ("w2", (ff, d)),
    ], layer=layer))


def dsv32_dense_context_layout(dims: Dsv32Dims) -> PackedLayout:
    t, ff = dims.tokens, dims.d_ff
    return PackedLayout.build(_dsv32_attn_ctx_specs(dims) + [
        ("x1", (t, ff), "bf16"),
        ("x3", (t, ff), "bf16"),
    ])


def dsv32_moe_weight_layout(dims: Dsv32Dims, layer: int | None = None) -> PackedLayout:
    from .moe.spec import moe_weight_specs

    return PackedLayout.build(_param_specs(
        dims, _dsv32_attn_weight_specs(dims) + moe_weight_specs(dims, dims.moe),
        layer=layer,
    ))


def dsv32_moe_context_layout(dims: Dsv32Dims) -> PackedLayout:
    from .moe.spec import moe_context_specs

    return PackedLayout.build(
        _dsv32_attn_ctx_specs(dims) + moe_context_specs(dims, dims.moe)
    )


@dataclass(frozen=True)
class Qwen35Dims:
    """Qwen3.5-dense dims: hybrid Gated-DeltaNet + gated-attention layers.

    Full-attn: n_heads x head_dim with output gate (w_q projects 2x),
    per-head qk-norm, PARTIAL rope (rot_dim = partial_rotary * head_dim).
    Linear-attn: num_k_heads x head_k_dim (keys/queries), num_v_heads x
    head_v_dim (values, GVA: v-head i reads k-head i // (HV/HK)), causal
    conv (kernel conv_kernel) over [q|k|v], gated RMSNorm over head_v_dim.
    All layers share the dense SwiGLU MLP. Embeddings tied ([table |
    final_norm_w] rides W_embed via head_weight_layout).
    """

    d_model: int
    n_layers: int
    full_attention_interval: int
    # full-attention sub-block
    n_heads: int
    n_kv_heads: int
    head_dim: int
    partial_rotary_factor: float
    # linear-attention sub-block
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int
    # shared
    d_ff: int
    vocab_size: int
    tokens: int
    seq_len: int
    rope_base: float = 10_000_000.0
    dtypes: DTypePolicy = DTypePolicy()
    # explicit per-sequence lengths for ragged packing (sum == tokens);
    # None = uniform sequences of seq_len (varlen-first design note)
    seq_lens: tuple[int, ...] | None = None

    @property
    def seq_spec(self):
        """int (uniform) or tuple (ragged) — the ops-layer seq argument."""
        return self.seq_lens if self.seq_lens is not None else self.seq_len

    @property
    def attn_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def rot_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def qkvz_dim(self) -> int:
        return 2 * self.key_dim + 2 * self.value_dim

    @property
    def ba_dim(self) -> int:
        return 2 * self.num_v_heads

    def kind_of(self, layer: int) -> str:
        return "full" if (layer + 1) % self.full_attention_interval == 0 else "lin"


def _qwen35_lin_attn_specs(dims) -> list[tuple[str, tuple[int, ...]]]:
    """DeltaNet attention-part weight fields (shared by the dense and MoE
    qwen3.5 families — the MLP tail differs, the attention never does)."""
    d = dims.d_model
    return [
        ("attn_norm_w", (d,)),
        ("w_qkvz", (d, dims.qkvz_dim)),
        ("w_ba", (d, dims.ba_dim)),
        ("w_conv", (dims.conv_dim, dims.conv_kernel)),
        ("A_log", (dims.num_v_heads,)),
        ("dt_bias", (dims.num_v_heads,)),
        ("lin_norm_w", (dims.head_v_dim,)),
        ("w_out", (dims.value_dim, d)),
        ("ffn_norm_w", (d,)),
    ]


def _qwen35_lin_attn_ctx(dims) -> list[tuple[str, tuple[int, ...], str]]:
    t, d = dims.tokens, dims.d_model
    hv = dims.num_v_heads
    return [
        ("rstd_attn", (t,), "fp32"),
        ("qkvz", (t, dims.qkvz_dim), "bf16"),
        ("ba", (t, dims.ba_dim), "bf16"),
        ("g_post", (t, hv), "fp32"),
        ("A_int", (t, hv, 64), "bf16"),
        ("core_out", (t, hv, dims.head_v_dim), "bf16"),
        ("rstd_gate", (t * hv,), "fp32"),
        ("xo", (t, d), "bf16"),
        ("rstd_ffn", (t,), "fp32"),
    ]


def _qwen35_attn_attn_specs(dims) -> list[tuple[str, tuple[int, ...]]]:
    """Gated-attention attention-part weight fields (wq = [Q_all | gate_all])."""
    d = dims.d_model
    return [
        ("attn_norm_w", (d,)),
        ("wq", (d, 2 * dims.attn_dim)),
        ("wk", (d, dims.kv_dim)),
        ("wv", (d, dims.kv_dim)),
        ("q_norm_w", (dims.head_dim,)),
        ("k_norm_w", (dims.head_dim,)),
        ("wo", (dims.attn_dim, d)),
        ("ffn_norm_w", (d,)),
    ]


def _qwen35_attn_attn_ctx(dims) -> list[tuple[str, tuple[int, ...], str]]:
    t, d = dims.tokens, dims.d_model
    h, kvh = dims.n_heads, dims.n_kv_heads
    return [
        ("rstd_attn", (t,), "fp32"),
        ("qm", (t, dims.attn_dim), "bf16"),
        ("km", (t, dims.kv_dim), "bf16"),
        ("rstd_q", (t * h,), "fp32"),
        ("rstd_k", (t * kvh,), "fp32"),
        ("gate", (t, dims.attn_dim), "bf16"),
        ("v", (t, dims.kv_dim), "bf16"),
        _lse_spec(dims, h),
        ("attn_out", (t, dims.attn_dim), "bf16"),
        ("xo", (t, d), "bf16"),
        ("rstd_ffn", (t,), "fp32"),
    ]


def _dense_mlp_specs(dims) -> list[tuple[str, tuple[int, ...]]]:
    d, ff = dims.d_model, dims.d_ff
    return [("w1", (d, ff)), ("w3", (d, ff)), ("w2", (ff, d))]


def qwen35_lin_weight_layout(dims: Qwen35Dims, layer: int | None = None) -> PackedLayout:
    """DeltaNet layer weights. Default policy stores A_log/dt_bias bf16
    (golden identical — fla receives fp32 casts at call time; bf16-ULP-vs-
    AdamW caveat recorded in docs/notes/qwen35-design.md); a dtype policy
    override ("A_log"/"dt_bias" -> fp32) lifts that."""
    return PackedLayout.build(_param_specs(
        dims, _qwen35_lin_attn_specs(dims) + _dense_mlp_specs(dims), layer=layer,
    ))


def qwen35_lin_context_layout(dims: Qwen35Dims) -> PackedLayout:
    """DeltaNet saved context (design §3d): projections + fla's saved
    outputs; post-conv and q/k l2norms are recomputed in backward."""
    t, ff = dims.tokens, dims.d_ff
    return PackedLayout.build(_qwen35_lin_attn_ctx(dims) + [
        ("x1", (t, ff), "bf16"),
        ("x3", (t, ff), "bf16"),
    ])


def qwen35_attn_weight_layout(dims: Qwen35Dims, layer: int | None = None) -> PackedLayout:
    """Gated-attention layer weights: w_q projects [Q_all | gate_all]."""
    return PackedLayout.build(_param_specs(
        dims, _qwen35_attn_attn_specs(dims) + _dense_mlp_specs(dims), layer=layer,
    ))


def qwen35_attn_context_layout(dims: Qwen35Dims) -> PackedLayout:
    """Gated-attention saved context: pre-norm q (qm) + per-head rstds
    (qwen3 pattern — backward rebuilds post-norm/rope), k likewise, v,
    pre-sigmoid gate, flash outputs, xo, MLP projections."""
    t, ff = dims.tokens, dims.d_ff
    return PackedLayout.build(_qwen35_attn_attn_ctx(dims) + [
        ("x1", (t, ff), "bf16"),
        ("x3", (t, ff), "bf16"),
    ])


@dataclass(frozen=True)
class Qwen35MoeDims(Qwen35Dims):
    """Qwen3.5-MoE dims: the dense hybrid's attention kinds verbatim, the
    dense SwiGLU replaced by the routed MoE tail (E=256 top-8 F=512 at
    35B-A3B) + ONE sigmoid-gated shared expert. d_ff aliases d_ff_expert
    (metadata only); untied embeddings per the 35B config."""

    moe: MoESpec | None = None


def qwen35moe_lin_weight_layout(dims: Qwen35MoeDims, layer: int | None = None) -> PackedLayout:
    from .moe.spec import moe_weight_specs

    return PackedLayout.build(_param_specs(
        dims, _qwen35_lin_attn_specs(dims) + moe_weight_specs(dims, dims.moe),
        layer=layer,
    ))


def qwen35moe_lin_context_layout(dims: Qwen35MoeDims) -> PackedLayout:
    from .moe.spec import moe_context_specs

    return PackedLayout.build(_qwen35_lin_attn_ctx(dims) + moe_context_specs(dims, dims.moe))


def qwen35moe_attn_weight_layout(dims: Qwen35MoeDims, layer: int | None = None) -> PackedLayout:
    from .moe.spec import moe_weight_specs

    return PackedLayout.build(_param_specs(
        dims, _qwen35_attn_attn_specs(dims) + moe_weight_specs(dims, dims.moe),
        layer=layer,
    ))


def qwen35moe_attn_context_layout(dims: Qwen35MoeDims) -> PackedLayout:
    from .moe.spec import moe_context_specs

    return PackedLayout.build(_qwen35_attn_attn_ctx(dims) + moe_context_specs(dims, dims.moe))


def embed_weight_layout(dims) -> PackedLayout:
    return PackedLayout.build(
        _param_specs(dims, [("w", (dims.vocab_size, dims.d_model))], ns="embed")
    )


def head_weight_layout(dims) -> PackedLayout:
    """LM head object: the projection table PLUS the model's final RMSNorm
    weight (a real learned parameter in llama3/qwen3/qwen3.5 — packed here
    so its gradient and optimizer state ride the head object). Policy names:
    "head.w", "head.final_norm_w" (also matched by "*_norm_w")."""
    return PackedLayout.build(_param_specs(dims, [
        ("w", (dims.vocab_size, dims.d_model)),
        ("final_norm_w", (dims.d_model,)),
    ], ns="head"))

