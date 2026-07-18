"""Shared bridge plumbing: packed-byte access and state_dict load/verify.

Source-agnostic byte access: callers supply ``get_bytes(oid) -> flat uint8
CPU tensor``, built here over either an in-process ``initial_values`` buffer
dict or a service ``EngineClient`` (``get_object`` payload).
"""
from __future__ import annotations

import functools

import torch


def bytes_from_buffer(buf) -> torch.Tensor:
    """Flat uint8 CPU copy of a runtime object buffer (in-process init:
    backing buffers are pinned host, so this is a host->host clone)."""
    from dataflow.runtime.interop import torch_view

    return torch_view(buf, (buf.size_bytes,), torch.uint8).clone()


def bytes_for_object(values: dict, oid: str) -> torch.Tensor:
    """One object's packed bytes from an in-process ``initial_values`` dict."""
    return bytes_from_buffer(values[oid])


def get_bytes_from_values(values: dict):
    """``get_bytes`` over an in-process ``initial_values`` dict."""
    return functools.partial(bytes_for_object, values)


def bytes_from_client_object(client, oid: str) -> torch.Tensor:
    """One object's packed bytes fetched from a service ``EngineClient``."""
    payload = client.get_object(oid)
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()


def get_bytes_from_client(client):
    """``get_bytes`` over a service ``EngineClient`` (``get_object`` payload)."""
    return functools.partial(bytes_from_client_object, client)


def transposed(w: torch.Tensor) -> torch.Tensor:
    """Projection orientation change: engine packed ``(in, out)`` ->
    ``nn.Linear`` weight ``(out, in)``. Pure layout — same values, same bits."""
    return w.t().contiguous()


def load_state_dict_strict(model, sd: dict):
    """Load ``sd`` into ``model`` on its device (strict; raises on any
    key/shape mismatch)."""
    dev = next(model.parameters()).device
    model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=True)
    return model


def assert_state_dict_byte_identical(model, sd: dict) -> None:
    """Gate: every loaded reference tensor equals the engine's packed bytes
    (bit-for-bit, up to the documented orientation changes). Compares the
    STATE_DICT — not named_parameters — so tied configs (one tensor behind
    two keys) are verified at both keys."""
    msd = model.state_dict()
    assert set(sd) == set(msd), f"key mismatch: {set(sd) ^ set(msd)}"
    for k, v in sd.items():
        if not torch.equal(msd[k].detach().cpu(), v.cpu()):
            raise AssertionError(f"init not byte-identical at {k}")
