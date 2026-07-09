"""Weight bridge: the engine's packed init bytes -> the reference
``nn.Module`` state_dict, byte-identical.

Parity invariant #2: the initialization is seeded ONCE by the engine
(``initial_values`` / the daemon's ``family_init_all``); this bridge loads
those exact bytes into the isolated ``references.Llama3`` so the ONLY
variable between the two training runs is the execution engine.

The engine stores each block's parameters packed ``(in, out)``; the reference
uses ``nn.Linear`` (weight ``(out, in)``), so projection weights are
TRANSPOSED here (a pure layout change — the values, hence the bits, are the
same). Embedding / LM-head tables ``(vocab, d)`` and 1-D norm gains load
directly. The byte-identity gate compares the loaded parameters against the
unpacked field values.

Source-agnostic: callers supply ``get_bytes(oid) -> flat uint8 CPU tensor``,
which reads either an in-process ``initial_values`` buffer or a service
``get_object(oid)`` payload.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make the top-level references/ package importable when run as a script
# (tests get this from the root conftest; scripts compute it from here).
_ROOT = str(Path(__file__).resolve().parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from references.llama3 import Llama3, Llama3Config  # noqa: E402


def reference_config(cfg) -> Llama3Config:
    """Build the isolated reference config from a ``ShapedLlamaConfig``."""
    from dataflow.training.models.llama3 import dims_of

    dims = dims_of(cfg)
    return Llama3Config(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads, d_ff=cfg.d_ff, vocab_size=cfg.vocab_size,
        rope_base=dims.rope_base,
    )


def build_reference(cfg, *, device="cuda", dtype=torch.bfloat16) -> Llama3:
    """A reference model at the engine's storage dtype (all-bf16), on
    ``device``. Weights are then loaded via ``load_engine_init``."""
    return Llama3(reference_config(cfg)).to(device=device, dtype=dtype)


# -- packed bytes -> state_dict ------------------------------------------------

def bytes_from_buffer(buf) -> torch.Tensor:
    """Flat uint8 CPU copy of a runtime object buffer (in-process init:
    backing buffers are pinned host, so this is a host->host clone)."""
    from dataflow.tasks.interop import torch_view

    return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()


def get_bytes_from_values(values: dict):
    """``get_bytes`` over an in-process ``initial_values`` dict."""
    return lambda oid: bytes_from_buffer(values[oid])


def get_bytes_from_client(client):
    """``get_bytes`` over a service ``EngineClient`` (``get_object`` payload)."""
    def g(oid: str) -> torch.Tensor:
        payload = client.get_object(oid)          # bytes
        return torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    return g


def to_state_dict(dims, n_layers: int, get_bytes) -> dict:
    """Reference ``state_dict`` from the engine's packed weight objects."""
    from dataflow.tasks.layouts import (
        embed_weight_layout,
        head_weight_layout,
        weight_layout,
    )

    sd: dict[str, torch.Tensor] = {}
    ew = embed_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
    sd["embed.weight"] = ew["w"].clone()
    hw = head_weight_layout(dims).unpack_tensor(get_bytes("W_head"))
    sd["lm_head.weight"] = hw["w"].clone()            # (vocab, d) direct
    sd["final_norm.weight"] = hw["final_norm_w"].clone()
    for i in range(n_layers):
        w = weight_layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
        sd[f"blocks.{i}.attn_norm.weight"] = w["attn_norm_w"].clone()
        sd[f"blocks.{i}.attn.wq.weight"] = w["wq"].t().contiguous()
        sd[f"blocks.{i}.attn.wk.weight"] = w["wk"].t().contiguous()
        sd[f"blocks.{i}.attn.wv.weight"] = w["wv"].t().contiguous()
        sd[f"blocks.{i}.attn.wo.weight"] = w["wo"].t().contiguous()
        sd[f"blocks.{i}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        sd[f"blocks.{i}.mlp.w1.weight"] = w["w1"].t().contiguous()
        sd[f"blocks.{i}.mlp.w3.weight"] = w["w3"].t().contiguous()
        sd[f"blocks.{i}.mlp.w2.weight"] = w["w2"].t().contiguous()
    return sd


def load_engine_init(model: Llama3, dims, n_layers: int, get_bytes) -> Llama3:
    """Load the engine's packed init into ``model`` (strict; raises on any
    key/shape mismatch)."""
    sd = to_state_dict(dims, n_layers, get_bytes)
    dev = next(model.parameters()).device
    model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=True)
    return model


def assert_byte_identical(model: Llama3, dims, n_layers: int, get_bytes) -> None:
    """Gate: every reference parameter equals the engine's packed bytes
    (bit-for-bit, up to the projection transpose)."""
    sd = to_state_dict(dims, n_layers, get_bytes)
    params = dict(model.named_parameters())
    assert set(sd) == set(params), (
        f"key mismatch: {set(sd) ^ set(params)}"
    )
    for k, ref in sd.items():
        got = params[k].detach().cpu()
        if not torch.equal(got, ref.cpu()):
            raise AssertionError(f"init not byte-identical at {k}")


# ===================== qwen3.5-dense (hybrid) ================================

def reference_qwen35_config(cfg):
    """Build the isolated qwen3.5 reference config from a ShapedQwen35Config."""
    from references.qwen35 import Qwen35Config

    return Qwen35Config(
        n_layers=cfg.n_layers, d_model=cfg.d_model,
        full_attention_interval=cfg.full_attention_interval,
        n_heads=cfg.n_heads, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        partial_rotary_factor=cfg.partial_rotary_factor,
        lin_k_heads=cfg.lin_k_heads, lin_v_heads=cfg.lin_v_heads,
        lin_k_head_dim=cfg.lin_k_head_dim, lin_v_head_dim=cfg.lin_v_head_dim,
        lin_conv_kernel=cfg.lin_conv_kernel, d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size, rope_base=cfg.rope_base,
        tied_embeddings=cfg.tied_embeddings,
    )


def build_qwen35_reference(cfg, *, device="cuda", dtype=torch.bfloat16):
    from references.qwen35 import Qwen35

    return Qwen35(reference_qwen35_config(cfg)).to(device=device, dtype=dtype)


def _T(x: torch.Tensor) -> torch.Tensor:
    return x.t().contiguous()


def to_qwen35_state_dict(cfg, get_bytes) -> dict:
    """Reference qwen3.5 state_dict from the engine's packed weight objects.
    Projections transpose (in,out)->(out,in); the depthwise conv reshapes
    (D,W)->(D,1,W); 1-D params (A_log/dt_bias/norm gains) load direct;
    embed/head tables are (vocab,d) direct (tied packs both into W_embed)."""
    from dataflow.tasks.layouts import (
        embed_weight_layout,
        head_weight_layout,
        qwen35_attn_weight_layout,
        qwen35_lin_weight_layout,
    )
    from dataflow.training.models.qwen35 import dims_of_qwen35

    dims = dims_of_qwen35(cfg)
    sd: dict[str, torch.Tensor] = {}
    if cfg.tied_embeddings:
        ew = head_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
        sd["embed.weight"] = ew["w"].clone()
        sd["lm_head.weight"] = ew["w"].clone()          # tied (shared param)
        sd["final_norm.weight"] = ew["final_norm_w"].clone()
    else:
        ew = embed_weight_layout(dims).unpack_tensor(get_bytes("W_embed"))
        sd["embed.weight"] = ew["w"].clone()
        hw = head_weight_layout(dims).unpack_tensor(get_bytes("W_head"))
        sd["lm_head.weight"] = hw["w"].clone()
        sd["final_norm.weight"] = hw["final_norm_w"].clone()
    for i in range(cfg.n_layers):
        p = f"blocks.{i}"
        if dims.kind_of(i) == "lin":
            w = qwen35_lin_weight_layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
            sd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
            sd[f"{p}.mixer.w_qkvz.weight"] = _T(w["w_qkvz"])
            sd[f"{p}.mixer.w_ba.weight"] = _T(w["w_ba"])
            sd[f"{p}.mixer.conv.weight"] = w["w_conv"].unsqueeze(1).contiguous()
            sd[f"{p}.mixer.A_log"] = w["A_log"].clone()
            sd[f"{p}.mixer.dt_bias"] = w["dt_bias"].clone()
            sd[f"{p}.mixer.lin_norm.weight"] = w["lin_norm_w"].clone()
            sd[f"{p}.mixer.w_out.weight"] = _T(w["w_out"])
            sd[f"{p}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        else:
            w = qwen35_attn_weight_layout(dims, layer=i).unpack_tensor(get_bytes(f"W_{i}"))
            sd[f"{p}.attn_norm.weight"] = w["attn_norm_w"].clone()
            sd[f"{p}.mixer.wq.weight"] = _T(w["wq"])
            sd[f"{p}.mixer.wk.weight"] = _T(w["wk"])
            sd[f"{p}.mixer.wv.weight"] = _T(w["wv"])
            sd[f"{p}.mixer.q_norm.weight"] = w["q_norm_w"].clone()
            sd[f"{p}.mixer.k_norm.weight"] = w["k_norm_w"].clone()
            sd[f"{p}.mixer.wo.weight"] = _T(w["wo"])
            sd[f"{p}.ffn_norm.weight"] = w["ffn_norm_w"].clone()
        for nm in ("w1", "w3", "w2"):
            sd[f"{p}.mlp.{nm}.weight"] = _T(w[nm])
    return sd


def load_qwen35_init(model, cfg, get_bytes):
    sd = to_qwen35_state_dict(cfg, get_bytes)
    dev = next(model.parameters()).device
    model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=True)
    return model


# ===================== family dispatch (driver seam) ========================

def build_reference_model(cfg, *, device="cuda", dtype=torch.bfloat16):
    """Build the reference nn.Module for ``cfg``'s family."""
    if type(cfg).__name__ == "ShapedQwen35Config":
        return build_qwen35_reference(cfg, device=device, dtype=dtype)
    return build_reference(cfg, device=device, dtype=dtype)


def load_reference_init(model, cfg, dims, get_bytes):
    """Load the engine's packed init into ``model`` (family-dispatched)."""
    if type(cfg).__name__ == "ShapedQwen35Config":
        return load_qwen35_init(model, cfg, get_bytes)
    return load_engine_init(model, dims, cfg.n_layers, get_bytes)
