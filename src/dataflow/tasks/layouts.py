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

from dataflow.core import DTYPE_BITS
from dataflow.runtime.device.base import Buffer

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

    def for_field(self, name: str) -> ParamDTypes:
        for pattern, dts in self.overrides:
            if fnmatchcase(name, pattern):
                return dts
        return self.default


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

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


def _param_specs(dims, names_shapes, ns: str | None = None,
                 ) -> list[tuple[str, tuple[int, ...], str]]:
    """Trainable-field specs at the dims' dtype policy (param role).

    ``ns`` prefixes the POLICY LOOKUP name (not the field name): the embed
    and head tables both pack a field literally named "w", so policies
    address them as "embed.w" / "head.w" / "head.final_norm_w". Block
    fields are looked up bare ("wq", "A_log", "*_norm_w", ...).
    """
    policy = getattr(dims, "dtypes", None) or DTypePolicy()
    key = (lambda n: f"{ns}.{n}") if ns else (lambda n: n)
    return [(n, s, policy.for_field(key(n)).param) for n, s in names_shapes]


def _policy_of(dims) -> DTypePolicy:
    return getattr(dims, "dtypes", None) or DTypePolicy()


def grad_layout(weight: PackedLayout, policy: DTypePolicy,
                ns: str | None = None) -> PackedLayout:
    """dW layout mirroring a weight layout field-by-field at grad dtypes."""
    key = (lambda n: f"{ns}.{n}") if ns else (lambda n: n)
    return PackedLayout.build(
        [(f.name, f.shape, policy.for_field(key(f.name)).grad) for f in weight.fields]
    )


def opt_state_layout(weight: PackedLayout, policy: DTypePolicy,
                     ns: str | None = None) -> PackedLayout:
    """AdamW state for one weight object: per-field first/second moments at
    the policy's opt dtype. Under the all-bf16 default this packs to the
    same total bytes as the historical flat ``adamw_state_layout`` (every
    m/v span is the field's byte span, alignment included) but the interior
    MAPPING is per-field [m_f | v_f] pairs, never covering padding gaps."""
    key = (lambda n: f"{ns}.{n}") if ns else (lambda n: n)
    specs: list[tuple[str, tuple[int, ...], str]] = []
    for f in weight.fields:
        o = policy.for_field(key(f.name)).opt
        specs.append((f"m_{f.name}", f.shape, o))
        specs.append((f"v_{f.name}", f.shape, o))
    return PackedLayout.build(specs)


def weight_layout(dims: LlamaDims) -> PackedLayout:
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
    ]))


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
        ("lse", ((t // dims.seq_len) * h, dims.seq_len), "fp32"),
        ("attn_out", (t, d), "bf16"),
        ("h_mid", (t, d), "bf16"),
        ("rstd_ffn", (t,), "fp32"),
        ("x1", (t, ff), "bf16"),
        ("x3", (t, ff), "bf16"),
    ])


def qwen3_weight_layout(dims: Qwen3Dims) -> PackedLayout:
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
    ]))


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
        ("lse", ((t // dims.seq_len) * h, dims.seq_len), "fp32"),
        ("attn_out", (t, q), "bf16"),
        ("h_mid", (t, d), "bf16"),
        ("rstd_ffn", (t,), "fp32"),
        ("x1", (t, ff), "bf16"),
        ("x3", (t, ff), "bf16"),
    ])


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


def qwen35_lin_weight_layout(dims: Qwen35Dims) -> PackedLayout:
    """DeltaNet layer weights. Default policy stores A_log/dt_bias bf16
    (golden identical — fla receives fp32 casts at call time; bf16-ULP-vs-
    AdamW caveat recorded in docs/notes/qwen35-design.md); a dtype policy
    override ("A_log"/"dt_bias" -> fp32) lifts that."""
    d, ff = dims.d_model, dims.d_ff
    return PackedLayout.build(_param_specs(dims, [
        ("attn_norm_w", (d,)),
        ("w_qkvz", (d, dims.qkvz_dim)),
        ("w_ba", (d, dims.ba_dim)),
        ("w_conv", (dims.conv_dim, dims.conv_kernel)),
        ("A_log", (dims.num_v_heads,)),
        ("dt_bias", (dims.num_v_heads,)),
        ("lin_norm_w", (dims.head_v_dim,)),
        ("w_out", (dims.value_dim, d)),
        ("ffn_norm_w", (d,)),
        ("w1", (d, ff)),
        ("w3", (d, ff)),
        ("w2", (ff, d)),
    ]))


def qwen35_lin_context_layout(dims: Qwen35Dims) -> PackedLayout:
    """DeltaNet saved context (design §3d): projections + fla's saved
    outputs; post-conv and q/k l2norms are recomputed in backward."""
    t, d, ff = dims.tokens, dims.d_model, dims.d_ff
    hv = dims.num_v_heads
    return PackedLayout.build([
        ("rstd_attn", (t,), "fp32"),
        ("qkvz", (t, dims.qkvz_dim), "bf16"),
        ("ba", (t, dims.ba_dim), "bf16"),
        ("g_post", (t, hv), "fp32"),
        ("A_int", (t, hv, 64), "bf16"),
        ("core_out", (t, hv, dims.head_v_dim), "bf16"),
        ("rstd_gate", (t * hv,), "fp32"),
        ("xo", (t, d), "bf16"),
        ("rstd_ffn", (t,), "fp32"),
        ("x1", (t, ff), "bf16"),
        ("x3", (t, ff), "bf16"),
    ])


def qwen35_attn_weight_layout(dims: Qwen35Dims) -> PackedLayout:
    """Gated-attention layer weights: w_q projects [Q_all | gate_all]."""
    d, ff = dims.d_model, dims.d_ff
    return PackedLayout.build(_param_specs(dims, [
        ("attn_norm_w", (d,)),
        ("wq", (d, 2 * dims.attn_dim)),
        ("wk", (d, dims.kv_dim)),
        ("wv", (d, dims.kv_dim)),
        ("q_norm_w", (dims.head_dim,)),
        ("k_norm_w", (dims.head_dim,)),
        ("wo", (dims.attn_dim, d)),
        ("ffn_norm_w", (d,)),
        ("w1", (d, ff)),
        ("w3", (d, ff)),
        ("w2", (ff, d)),
    ]))


def qwen35_attn_context_layout(dims: Qwen35Dims) -> PackedLayout:
    """Gated-attention saved context: pre-norm q (qm) + per-head rstds
    (qwen3 pattern — backward rebuilds post-norm/rope), k likewise, v,
    pre-sigmoid gate, flash outputs, xo, MLP projections."""
    t, d, ff = dims.tokens, dims.d_model, dims.d_ff
    h, kvh = dims.n_heads, dims.n_kv_heads
    return PackedLayout.build([
        ("rstd_attn", (t,), "fp32"),
        ("qm", (t, dims.attn_dim), "bf16"),
        ("km", (t, dims.kv_dim), "bf16"),
        ("rstd_q", (t * h,), "fp32"),
        ("rstd_k", (t * kvh,), "fp32"),
        ("gate", (t, dims.attn_dim), "bf16"),
        ("v", (t, dims.kv_dim), "bf16"),
        ("lse", ((t // dims.seq_len) * h, dims.seq_len), "fp32"),
        ("attn_out", (t, dims.attn_dim), "bf16"),
        ("xo", (t, d), "bf16"),
        ("rstd_ffn", (t,), "fp32"),
        ("x1", (t, ff), "bf16"),
        ("x3", (t, ff), "bf16"),
    ])


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


def adamw_state_layout(param_elems: int) -> PackedLayout:
    """DEPRECATED flat form (bf16 halves over raw element counts, padding
    included). Kept only for byte-size back-compat call sites while the
    per-field ``opt_state_layout`` rollout completes; new code must use
    ``opt_state_layout(weight, policy)``."""
    return PackedLayout.build([
        ("m", (param_elems,), "bf16"),
        ("v", (param_elems,), "bf16"),
    ])
