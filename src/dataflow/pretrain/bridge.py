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
