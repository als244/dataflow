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
    from .modules.moe.spec import MoESpec

_ALIGN = 256


@dataclass(frozen=True)
class ParamDTypes:
    """Storage dtypes for one trainable field: the parameter itself, its
    gradient, and its AdamW moments.
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
        from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view

        f = self.field(name)
        return torch_view(buffer, f.shape, TORCH_DTYPE_BY_NAME[f.dtype], offset_bytes=f.offset_bytes)

    def views(self, buffer: Buffer) -> dict:
        from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME, torch_view

        return {
            f.name: torch_view(buffer, f.shape, TORCH_DTYPE_BY_NAME[f.dtype], offset_bytes=f.offset_bytes)
            for f in self.fields
        }

    def unpack_tensor(self, flat) -> dict:
        """Views into a flat uint8 torch tensor (golden-reference side)."""
        import torch

        from dataflow.runtime.interop import TORCH_DTYPE_BY_NAME

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
    max_tokens: int
    seq_len: int
    rope_base: float = 500_000.0
    dtypes: DTypePolicy = DTypePolicy()
    # explicit per-sequence lengths for ragged packing (sum == tokens);
    # None = uniform sequences of seq_len
    seq_lens: tuple[int, ...] | None = None

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim
    # per-field optimizer assignment (tasks/optim.py): "adamw" (default,
    # historical behavior) | "sgd" | "sgdm" | "muon" | an OptPolicy with
    # fnmatch overrides. update_specials (noaux bias, frozen) stay the
    # highest-priority per-field override on top of this.
    opt_policy: object = "adamw"


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
    max_tokens: int
    seq_len: int
    rope_base: float = 1_000_000.0
    dtypes: DTypePolicy = DTypePolicy()
    # explicit per-sequence lengths for ragged packing (sum == tokens);
    # None = uniform sequences of seq_len
    seq_lens: tuple[int, ...] | None = None

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim
    # per-field optimizer assignment (tasks/optim.py): "adamw" (default,
    # historical behavior) | "sgd" | "sgdm" | "muon" | an OptPolicy with
    # fnmatch overrides. update_specials (noaux bias, frozen) stay the
    # highest-priority per-field override on top of this.
    opt_policy: object = "adamw"


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


def resolve_policy(dims) -> DTypePolicy:
    return getattr(dims, "dtypes", None) or DTypePolicy()


def _lse_spec(dims, n_heads: int) -> tuple[str, tuple[int, ...], str]:
    """Flash lse context field: always the varlen ``(heads, tokens)`` layout.
    Models ALWAYS run varlen (a uniform batch is equal-length segments), so
    ops.flash_fwd emits this shape for every case — never the historical
    batched ``(batch*heads, seq_len)`` (identical element count, but the
    per-batch split reappears only for batch>1 and would mismatch the
    single-launch lse)."""
    return ("lse", (n_heads, dims.max_tokens), "fp32")


def grad_layout(weight: PackedLayout, policy: DTypePolicy,
                ns: str | None = None, layer: int | None = None,
                opt_policy=None) -> PackedLayout:
    """dW layout mirroring a weight layout field-by-field at grad dtypes.
    Fields whose OPTIMIZER rule is "frozen" drop out entirely — frozen
    params need no gradient storage (warm-up phases: dW collapses to the
    trainable fields; a fully-frozen layer's dW sizes to zero and the
    lowering prunes the object and its optimizer task)."""
    key = (lambda n: f"{ns}.{n}") if ns else (lambda n: n)
    fields = weight.fields
    if opt_policy is not None:
        from .optim import resolve_opt_policy

        op = resolve_opt_policy(opt_policy)
        fields = [f for f in fields
                  if op.for_field(key(f.name), layer, f.shape) != "frozen"]
    return PackedLayout.build(
        [(f.name, f.shape, policy.for_field(key(f.name), layer).grad)
         for f in fields]
    )


def opt_state_layout(weight: PackedLayout, policy: DTypePolicy,
                     ns: str | None = None, layer: int | None = None,
                     opt_policy=None, update_regions=None) -> PackedLayout:
    """Optimizer state for one weight object: per-field slots at the
    dtype policy's opt dtype, with the SLOT SET per field decided by the
    optimizer policy (tasks/optim.py): adamw [m_f | v_f] (the default —
    byte-identical to the historical layout), sgdm/muon [m_f], sgd
    nothing. Interior mapping is per-field, never covering padding.

    ``update_regions`` (sharded optimizer state): {field_name ->
    None | (lo, hi)} of the regions THIS RANK updates. Fields absent
    from the map get NO slots; a (lo, hi) dim-0 row range shrinks the
    slot to (hi - lo, *rest). Must match the region map baked into the
    task's block_params — the lowering sizes O with this same call."""
    from .optim import OPTIMIZERS, resolve_opt_policy

    op = resolve_opt_policy(opt_policy)
    key = (lambda n: f"{ns}.{n}") if ns else (lambda n: n)
    specs: list[tuple[str, tuple[int, ...], str]] = []
    for f in weight.fields:
        shape = f.shape
        if update_regions is not None:
            if f.name not in update_regions:
                continue
            rows = update_regions[f.name]
            if rows is not None:
                lo, hi = int(rows[0]), int(rows[1])
                shape = (hi - lo,) + tuple(f.shape[1:])
        o = policy.for_field(key(f.name), layer).opt
        for slot in OPTIMIZERS[op.for_field(key(f.name), layer,
                                            f.shape)].slots:
            specs.append((f"{slot}_{f.name}", shape, o))
    return PackedLayout.build(specs)


def sliced_layout(layout: PackedLayout, slices: dict) -> PackedLayout:
    """Rebuild a packed layout with the given fields narrowed to a
    (dim, lo, hi) slice — the resident-shard (tensor parallel) layout
    transform from a ShardPlan's ``tp_view``. Field order is
    preserved; offsets repack densely, so the result is a normal
    dense layout of the shard."""
    specs = []
    for f in layout.fields:
        shape = f.shape
        if f.name in slices:
            dim, lo, hi = (int(x) for x in slices[f.name])
            shape = tuple(hi - lo if i == dim else s
                          for i, s in enumerate(f.shape))
        specs.append((f.name, shape, f.dtype))
    return PackedLayout.build(specs)


def opt_state_slice_layout(n_slice: int, n_tail: int,
                           opt_dtype: str) -> PackedLayout:
    """Optimizer state for a BYTE-EQUAL shard (the rs/ag fast path):
    flat m/v over this rank's slice elements plus the full
    world-remainder tail (updated redundantly on every rank). Field
    identity is deliberately absent — byte-equal shards require a
    uniform optimizer policy across the root, checked at plan
    derivation."""
    specs = [("m_slice", (n_slice,), opt_dtype),
             ("v_slice", (n_slice,), opt_dtype)]
    if n_tail:
        specs += [("m_tail", (n_tail,), opt_dtype),
                  ("v_tail", (n_tail,), opt_dtype)]
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


def activation_layout(dims: LlamaDims) -> PackedLayout:
    """Saved-for-backward activations for one block forward (the `A_*` object).

    Saves exactly what block backward needs beyond (x, W): post-rope q/k, v,
    flash lse + attention output, the post-attention residual (h_mid), both
    rmsnorm rstds, and the two MLP projections.
    """
    t, d, kv, ff, h = dims.max_tokens, dims.d_model, dims.kv_dim, dims.d_ff, dims.n_heads
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
    from .modules.moe.spec import moe_weight_specs

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


def olmoe_activation_layout(dims: OlmoeDims) -> PackedLayout:
    """Saved-for-backward activations for one OLMoE block: qwen3's save-pre-norm
    convention with FULL-ROW rstds (one per token), plus the MoE tail's
    routing decision + pre-activations (tasks/moe/spec.py)."""
    from .modules.moe.spec import moe_context_specs

    t, d, q, kv, h = dims.max_tokens, dims.d_model, dims.q_dim, dims.kv_dim, dims.n_heads
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
    ] + moe_context_specs(dims, dims.moe, aux_temp=True))


def qwen3_activation_layout(dims: Qwen3Dims) -> PackedLayout:
    """Saved-for-backward activations for one Qwen3 block forward.

    qk-norm changes what is worth saving: instead of post-rope q/k we save
    the PRE-norm projections (qm/km) plus the per-head rstds — backward then
    re-applies norm+rope (cheap elementwise) to rebuild flash-bwd's q/k, and
    has exactly the tensors rmsnorm_bwd needs for the qk-norm gradient. v,
    lse, attn_out, h_mid, both block rstds and the MLP projections are saved
    as in llama."""
    t, d, q, kv, ff = dims.max_tokens, dims.d_model, dims.q_dim, dims.kv_dim, dims.d_ff
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
    from .modules.moe.spec import moe_weight_specs

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


def qwen3moe_activation_layout(dims: Qwen3MoeDims) -> PackedLayout:
    """qwen3's save-pre-norm convention with PER-HEAD rstds, plus the MoE
    tail's routing decision + pre-activations (tasks/moe/spec.py)."""
    from .modules.moe.spec import moe_context_specs

    t, d, q, kv = dims.max_tokens, dims.d_model, dims.q_dim, dims.kv_dim
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
    ] + moe_context_specs(dims, dims.moe, aux_temp=True))


@dataclass(frozen=True)
class Dsv3Dims:
    """DeepSeek-V3 dimensions: MLA attention + hybrid dense/MoE depth.

    MLA: q through the low-rank stack (d -> q_lora_rank -> n_heads *
    (qk_nope_dim + qk_rope_dim), RMSNorm mid-stack); kv through
    d -> (kv_lora_rank + qk_rope_dim) where the LAST qk_rope_dim columns
    are the ONE shared decoupled k_rope per token; latent RMSNorm then
    -> n_heads * (qk_nope_dim + v_head_dim). Per-head attention dims:
    qk = nope + rope, v = v_head_dim (flash runs at shared head_dim=qk
    with zero-padded v — exact; see blocks/modules/mla_forms.py). rope 1e4 on
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
    max_tokens: int
    seq_len: int
    rope_base: float = 10_000.0
    dtypes: DTypePolicy = DTypePolicy()
    seq_lens: tuple[int, ...] | None = None
    moe: MoESpec | None = None

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_dim + self.qk_rope_dim

    @property
    def q_dim(self) -> int:       # full q width after w_q_b
        return self.n_heads * self.qk_head_dim

    @property
    def v_dim(self) -> int:       # attention-output width fed to wo
        return self.n_heads * self.v_head_dim

    # per-field optimizer assignment (tasks/optim.py): "adamw" (default,
    # historical behavior) | "sgd" | "sgdm" | "muon" | an OptPolicy with
    # fnmatch overrides. update_specials (noaux bias, frozen) stay the
    # highest-priority per-field override on top of this.
    opt_policy: object = "adamw"
    # per-layer chain kinds ("dense"/"moe" here), one entry per layer —
    # populated by the family derive_dims (dims alone don't know n_layers). DATA, indexed.
    kinds: tuple[str, ...] = ()


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


# The indexer objective's saved-context inputs (dense warm-up): the KL
# backward re-derives everything it needs from these — h1 via
# (x, rstd_attn), the MLA q/k expansions via the compressed latents,
# targets via lse. dgrad is DEAD in warm-up, so every other ctx field
# has no consumer and is trimmed from the layouts (frozen-plan note §3).
# Too-small here fails LOUDLY (dict KeyError in the warm-up backward,
# caught by the model-step gates); too-large shows as size drift in the
# lowering gate.
DSA_WARMUP_CTX_FIELDS = ("rstd_attn", "q_a", "rstd_qa", "kv_a",
                         "rstd_kva", "lse")


def _warmup_ctx_filter(specs, dims):
    """Under the indexer-only objective (sparse_mode=False on the DSA
    dims), keep only the objective's inputs."""
    if getattr(dims, "sparse_mode", True):
        return specs
    keep = set(DSA_WARMUP_CTX_FIELDS)
    return [s for s in specs if s[0] in keep]


def _dsv3_attn_ctx_specs(dims: Dsv3Dims) -> list[tuple[str, tuple[int, ...], str]]:
    """The MLA saved set — COMPRESSED latents, not expanded heads: bwd
    re-expands through w_q_b/w_kv_b, so attention ctx is ~(q_lora +
    kv_lora + rope) wide instead of h*(qk+v). Pre-norm/pre-rope saves
    (the repo convention); attn_out at the TRUE (t, h*v) — the padded
    form is reconstructed from known-zeros at bwd time."""
    t = dims.max_tokens
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


def dsv3_dense_activation_layout(dims: Dsv3Dims) -> PackedLayout:
    t, ff = dims.max_tokens, dims.d_ff
    return PackedLayout.build(_warmup_ctx_filter(
        _dsv3_attn_ctx_specs(dims) + [
            ("x1", (t, ff), "bf16"),
            ("x3", (t, ff), "bf16"),
        ], dims))


def dsv3_moe_weight_layout(dims: Dsv3Dims, layer: int | None = None) -> PackedLayout:
    from .modules.moe.spec import moe_weight_specs

    return PackedLayout.build(_param_specs(
        dims, _dsv3_attn_weight_specs(dims) + moe_weight_specs(dims, dims.moe),
        layer=layer,
    ))


def dsv3_moe_activation_layout(dims: Dsv3Dims) -> PackedLayout:
    from .modules.moe.spec import moe_context_specs

    return PackedLayout.build(
        _dsv3_attn_ctx_specs(dims) + moe_context_specs(dims, dims.moe, aux_temp=True)
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
    # ablation knob: False = FREEZE the indexer — no KL loss, no
    # indexer gradients, optimizer skips its five fields entirely (not
    # even weight decay). Default True = paper-faithful sparse training
    # (the KL is the indexer's ONLY training signal).
    train_indexer: bool = True


@dataclass(frozen=True)
class Glm52Dims(Dsv32Dims):
    """GLM-5.2 IndexShare dims: dsv32 fields + the per-layer indexer role
    pattern. ``indexer_types[i]`` in {"full", "shared"}: full layers run
    their own lightning indexer and EMIT the selection object S consumed
    by the trailing run of shared layers (nearest-preceding-full rule,
    arXiv 2603.12201 — the pattern is greedy-searched upstream, so it is
    stored explicitly, never derived from a frequency formula). Shared
    layers carry NO indexer weights. Layer 0 must be full. Training: the
    leader's indexer aligns to the AVERAGED attention distributions of
    all layers it serves (paper L^I_multi; dI = sigma - P/N).

    Kinds: gdl (dense FFN + full indexer), gml (MoE + full), gmf (MoE +
    shared). Dense-FFN shared layers are rejected until a real config
    needs them (GLM-5.2's dense layers are all full)."""

    indexer_types: tuple[str, ...] = ()

    def layer_role(self, layer: int) -> str:
        return self.indexer_types[layer]

    def leader_index(self, layer: int) -> int:
        i = layer
        while self.indexer_types[i] != "full":
            i -= 1
        return i

    def group_members(self, leader: int) -> tuple[int, ...]:
        members = [leader]
        i = leader + 1
        while i < len(self.indexer_types) and self.indexer_types[i] == "shared":
            members.append(i)
            i += 1
        return tuple(members)

    def leaders(self) -> tuple[int, ...]:
        return tuple(i for i, r in enumerate(self.indexer_types) if r == "full")



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
    # METADATA GRAMMAR: the dsa selection and the routing pack live in
    # the layer's M object (never recomputed) — the ctx is exactly
    # dsv3-shaped in both modes
    return _dsv3_attn_ctx_specs(dims)


def dsv32_dense_weight_layout(dims: Dsv32Dims, layer: int | None = None) -> PackedLayout:
    d, ff = dims.d_model, dims.d_ff
    return PackedLayout.build(_param_specs(dims, _dsv32_attn_weight_specs(dims) + [
        ("w1", (d, ff)),
        ("w3", (d, ff)),
        ("w2", (ff, d)),
    ], layer=layer))


def dsv32_dense_activation_layout(dims: Dsv32Dims) -> PackedLayout:
    t, ff = dims.max_tokens, dims.d_ff
    return PackedLayout.build(_warmup_ctx_filter(
        _dsv32_attn_ctx_specs(dims) + [
            ("x1", (t, ff), "bf16"),
            ("x3", (t, ff), "bf16"),
        ], dims))


def dsv32_moe_weight_layout(dims: Dsv32Dims, layer: int | None = None) -> PackedLayout:
    from .modules.moe.spec import moe_weight_specs

    return PackedLayout.build(_param_specs(
        dims, _dsv32_attn_weight_specs(dims) + moe_weight_specs(dims, dims.moe),
        layer=layer,
    ))


def dsv32_moe_activation_layout(dims: Dsv32Dims) -> PackedLayout:
    from .modules.moe.spec import moe_context_specs

    return PackedLayout.build(_warmup_ctx_filter(
        _dsv32_attn_ctx_specs(dims)
        + moe_context_specs(dims, dims.moe, aux_temp=True), dims))


def glm52_aux_temp_layout(dims: "Glm52Dims", kind: str) -> PackedLayout:
    """glm52 M objects: leaders (gdl/gml) carry the dsa selection their
    group shares; moe kinds (gml/gmf) carry their own routing pack.
    Followers never carry a selection — they consume the producer's M."""
    from .modules.moe.spec import moe_aux_temp_specs

    specs: list[tuple[str, tuple[int, ...], str]] = []
    if kind in ("gdl", "gml") and getattr(dims, "sparse_mode", True):
        specs.append(("dsa_idx", (dims.max_tokens, dims.index_topk), "int32"))
    if kind in ("gml", "gmf"):
        specs += moe_aux_temp_specs(dims, dims.moe)
    return PackedLayout.build(specs)


def dsv32_aux_temp_layout(dims: Dsv32Dims, kind: str) -> PackedLayout:
    """The layer's M object: ALL its never-recompute metadata in one
    packed layout — the dsa selection (sparse mode) and the discrete
    routing pack (moe kinds)."""
    from .modules.moe.spec import moe_aux_temp_specs

    specs: list[tuple[str, tuple[int, ...], str]] = []
    if dims.sparse_mode:
        specs.append(("dsa_idx", (dims.max_tokens, dims.index_topk), "int32"))
    if kind == "moe":
        specs += moe_aux_temp_specs(dims, dims.moe)
    return PackedLayout.build(specs)


@dataclass(frozen=True)
class Qwen35Dims:
    """Qwen3.5-dense dims: hybrid Gated-DeltaNet + gated-attention layers.

    Full-attn: n_heads x head_dim with output gate (w_q projects 2x),
    per-head qk-norm, PARTIAL rope (rot_dim = partial_rotary * head_dim).
    Linear-attn: lin_k_heads x lin_k_head_dim (keys/queries), lin_v_heads x
    lin_v_head_dim (values, GVA: v-head i reads k-head i // (HV/HK)), causal
    conv (kernel lin_conv_kernel) over [q|k|v], gated RMSNorm over lin_v_head_dim.
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
    lin_k_heads: int
    lin_v_heads: int
    lin_k_head_dim: int
    lin_v_head_dim: int
    lin_conv_kernel: int
    # shared
    d_ff: int
    vocab_size: int
    max_tokens: int
    seq_len: int
    rope_base: float = 10_000_000.0
    dtypes: DTypePolicy = DTypePolicy()
    # explicit per-sequence lengths for ragged packing (sum == tokens);
    # None = uniform sequences of seq_len
    seq_lens: tuple[int, ...] | None = None

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
        return self.lin_k_heads * self.lin_k_head_dim

    @property
    def value_dim(self) -> int:
        return self.lin_v_heads * self.lin_v_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def qkvz_dim(self) -> int:
        return 2 * self.key_dim + 2 * self.value_dim

    @property
    def ba_dim(self) -> int:
        return 2 * self.lin_v_heads

    # per-field optimizer assignment (tasks/optim.py): "adamw" (default,
    # historical behavior) | "sgd" | "sgdm" | "muon" | an OptPolicy with
    # fnmatch overrides. update_specials (noaux bias, frozen) stay the
    # highest-priority per-field override on top of this.
    opt_policy: object = "adamw"
    # per-layer chain kinds ("lin"/"full" here), one entry per layer —
    # populated by the family derive_dims (dims alone don't know n_layers). DATA, indexed.
    kinds: tuple[str, ...] = ()


def _qwen35_lin_attn_specs(dims) -> list[tuple[str, tuple[int, ...]]]:
    """DeltaNet attention-part weight fields (shared by the dense and MoE
    qwen3.5 families — the MLP tail differs, the attention never does)."""
    d = dims.d_model
    return [
        ("attn_norm_w", (d,)),
        ("w_qkvz", (d, dims.qkvz_dim)),
        ("w_ba", (d, dims.ba_dim)),
        ("w_conv", (dims.conv_dim, dims.lin_conv_kernel)),
        ("A_log", (dims.lin_v_heads,)),
        ("dt_bias", (dims.lin_v_heads,)),
        ("lin_norm_w", (dims.lin_v_head_dim,)),
        ("w_out", (dims.value_dim, d)),
        ("ffn_norm_w", (d,)),
    ]


def _qwen35_lin_attn_ctx(dims) -> list[tuple[str, tuple[int, ...], str]]:
    t, d = dims.max_tokens, dims.d_model
    hv = dims.lin_v_heads
    return [
        ("rstd_attn", (t,), "fp32"),
        ("qkvz", (t, dims.qkvz_dim), "bf16"),
        ("ba", (t, dims.ba_dim), "bf16"),
        ("g_post", (t, hv), "fp32"),
        ("A_int", (t, hv, 64), "bf16"),
        ("core_out", (t, hv, dims.lin_v_head_dim), "bf16"),
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
    t, d = dims.max_tokens, dims.d_model
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
    AdamW caveat: bf16 moments round tiny updates); a dtype policy
    override ("A_log"/"dt_bias" -> fp32) lifts that."""
    return PackedLayout.build(_param_specs(
        dims, _qwen35_lin_attn_specs(dims) + _dense_mlp_specs(dims), layer=layer,
    ))


def qwen35_lin_activation_layout(dims: Qwen35Dims) -> PackedLayout:
    """DeltaNet saved context (design §3d): projections + fla's saved
    outputs; post-conv and q/k l2norms are recomputed in backward."""
    t, ff = dims.max_tokens, dims.d_ff
    return PackedLayout.build(_qwen35_lin_attn_ctx(dims) + [
        ("x1", (t, ff), "bf16"),
        ("x3", (t, ff), "bf16"),
    ])


def qwen35_attn_weight_layout(dims: Qwen35Dims, layer: int | None = None) -> PackedLayout:
    """Gated-attention layer weights: w_q projects [Q_all | gate_all]."""
    return PackedLayout.build(_param_specs(
        dims, _qwen35_attn_attn_specs(dims) + _dense_mlp_specs(dims), layer=layer,
    ))


def qwen35_attn_activation_layout(dims: Qwen35Dims) -> PackedLayout:
    """Gated-attention saved context: pre-norm q (qm) + per-head rstds
    (qwen3 pattern — backward rebuilds post-norm/rope), k likewise, v,
    pre-sigmoid gate, flash outputs, xo, MLP projections."""
    t, ff = dims.max_tokens, dims.d_ff
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
    from .modules.moe.spec import moe_weight_specs

    return PackedLayout.build(_param_specs(
        dims, _qwen35_lin_attn_specs(dims) + moe_weight_specs(dims, dims.moe),
        layer=layer,
    ))


def qwen35moe_lin_activation_layout(dims: Qwen35MoeDims) -> PackedLayout:
    from .modules.moe.spec import moe_context_specs

    return PackedLayout.build(
        _qwen35_lin_attn_ctx(dims) + moe_context_specs(dims, dims.moe, aux_temp=True))


def qwen35moe_attn_weight_layout(dims: Qwen35MoeDims, layer: int | None = None) -> PackedLayout:
    from .modules.moe.spec import moe_weight_specs

    return PackedLayout.build(_param_specs(
        dims, _qwen35_attn_attn_specs(dims) + moe_weight_specs(dims, dims.moe),
        layer=layer,
    ))


def qwen35moe_attn_activation_layout(dims: Qwen35MoeDims) -> PackedLayout:
    from .modules.moe.spec import moe_context_specs

    return PackedLayout.build(
        _qwen35_attn_attn_ctx(dims) + moe_context_specs(dims, dims.moe, aux_temp=True))


@dataclass(frozen=True)
class Gpt2Dims:
    """GPT-2 dimensions (the nanogpt-speedrun baseline shape): pre-LN
    blocks with LayerNorm (gain AND bias), fused c_attn QKV (one (d, 3d)
    matrix + bias), full MHA (no GQA, no rope — LEARNED positions),
    GELU-tanh MLP with biases, untied embed/head. ``max_seq_len`` is the
    learned-position table's row count (the model's fixed maximum context,
    independent of the per-program sequence length) — every segment of a
    packed round must fit inside it (positions restart per sequence)."""

    d_model: int
    n_heads: int
    d_ff: int
    vocab_size: int
    max_tokens: int
    seq_len: int
    max_seq_len: int
    # tied embed/head (config option; classic GPT-2 ties, the repo default
    # is untied like the llama3 baselines): ONE W_embed packs
    # [w | wpe | final_norm_w | final_norm_b] and serves both ends
    tied: bool = False
    # biases in Linears AND LayerNorms (the nanoGPT flag): True = classic
    # GPT-2; False = the bias-free variant — every b_*/*_norm_b field
    # drops out of the layouts entirely
    use_bias: bool = True
    dtypes: DTypePolicy = DTypePolicy()
    # explicit per-sequence lengths for ragged packing (sum == tokens);
    # None = uniform sequences of seq_len
    seq_lens: tuple[int, ...] | None = None

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.d_model          # full MHA: kv width == q width

    # per-field optimizer assignment (tasks/optim.py): "adamw" (default,
    # historical behavior) | "sgd" | "sgdm" | "muon" | an OptPolicy with
    # fnmatch overrides. update_specials (frozen) stay the highest-priority
    # per-field override on top of this.
    opt_policy: object = "adamw"


def gpt2_weight_layout(dims: Gpt2Dims, layer: int | None = None) -> PackedLayout:
    d, ff = dims.d_model, dims.d_ff
    specs = [
        ("attn_norm_w", (d,)),
        ("attn_norm_b", (d,)),
        ("w_qkv", (d, 3 * d)),
        ("b_qkv", (3 * d,)),
        ("wo", (d, d)),
        ("b_o", (d,)),
        ("ffn_norm_w", (d,)),
        ("ffn_norm_b", (d,)),
        ("w_fc", (d, ff)),
        ("b_fc", (ff,)),
        ("w_proj", (ff, d)),
        ("b_proj", (d,)),
    ]
    if not dims.use_bias:
        specs = [s for s in specs
                 if not (s[0].startswith("b_") or s[0].endswith("_norm_b"))]
    return PackedLayout.build(_param_specs(dims, specs, layer=layer))


def gpt2_activation_layout(dims: Gpt2Dims) -> PackedLayout:
    """Saved-for-backward activations for one GPT-2 block forward: both
    LayerNorm statistics pairs (mean AND rstd — mean-centered norms, unlike
    the rms families), post-projection q/k/v, flash lse + attention output,
    the post-attention residual, and the pre-GELU MLP projection. The
    normed inputs h1/h2 are recomputed in backward from the statistics."""
    t, d, ff, h = dims.max_tokens, dims.d_model, dims.d_ff, dims.n_heads
    return PackedLayout.build([
        ("mean_attn", (t,), "fp32"),
        ("rstd_attn", (t,), "fp32"),
        ("q", (t, d), "bf16"),
        ("k", (t, d), "bf16"),
        ("v", (t, d), "bf16"),
        _lse_spec(dims, h),
        ("attn_out", (t, d), "bf16"),
        ("h_mid", (t, d), "bf16"),
        ("mean_ffn", (t,), "fp32"),
        ("rstd_ffn", (t,), "fp32"),
        ("x_fc", (t, ff), "bf16"),
    ])


def gpt2_embed_layout(dims: Gpt2Dims) -> PackedLayout:
    """W_embed packs the token table AND the learned-position table; the
    TIED variant appends the final LayerNorm pair so one object serves the
    head too (policy ns follows the serving side: "head" when tied)."""
    specs = [
        ("w", (dims.vocab_size, dims.d_model)),
        ("wpe", (dims.max_seq_len, dims.d_model)),
    ]
    if dims.tied:
        specs.append(("final_norm_w", (dims.d_model,)))
        if dims.use_bias:
            specs.append(("final_norm_b", (dims.d_model,)))
    return PackedLayout.build(
        _param_specs(dims, specs, ns="head" if dims.tied else "embed"))


def gpt2_head_layout(dims: Gpt2Dims) -> PackedLayout:
    """W_head packs the projection table plus the final LayerNorm's gain
    AND (use_bias) bias (policy names "head.w", "head.final_norm_w",
    "head.final_norm_b")."""
    specs = [
        ("w", (dims.vocab_size, dims.d_model)),
        ("final_norm_w", (dims.d_model,)),
    ]
    if dims.use_bias:
        specs.append(("final_norm_b", (dims.d_model,)))
    return PackedLayout.build(_param_specs(dims, specs, ns="head"))


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

