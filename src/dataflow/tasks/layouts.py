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

from dataflow.core import DTYPE_BITS
from dataflow.runtime.device.base import Buffer

_ALIGN = 256


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

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


def weight_layout(dims: LlamaDims) -> PackedLayout:
    d, kv, ff = dims.d_model, dims.kv_dim, dims.d_ff
    return PackedLayout.build([
        ("attn_norm_w", (d,), "bf16"),
        ("wq", (d, d), "bf16"),
        ("wk", (d, kv), "bf16"),
        ("wv", (d, kv), "bf16"),
        ("wo", (d, d), "bf16"),
        ("ffn_norm_w", (d,), "bf16"),
        ("w1", (d, ff), "bf16"),
        ("w3", (d, ff), "bf16"),
        ("w2", (ff, d), "bf16"),
    ])


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
    return PackedLayout.build([
        ("attn_norm_w", (d,), "bf16"),
        ("wq", (d, q), "bf16"),
        ("wk", (d, kv), "bf16"),
        ("wv", (d, kv), "bf16"),
        ("q_norm_w", (hd,), "bf16"),
        ("k_norm_w", (hd,), "bf16"),
        ("wo", (q, d), "bf16"),
        ("ffn_norm_w", (d,), "bf16"),
        ("w1", (d, ff), "bf16"),
        ("w3", (d, ff), "bf16"),
        ("w2", (ff, d), "bf16"),
    ])


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


def embed_weight_layout(dims) -> PackedLayout:
    return PackedLayout.build([("w", (dims.vocab_size, dims.d_model), "bf16")])


def adamw_state_layout(param_elems: int) -> PackedLayout:
    """Optimizer state for one parameter object: bf16 first/second moments."""
    return PackedLayout.build([
        ("m", (param_elems,), "bf16"),
        ("v", (param_elems,), "bf16"),
    ])
